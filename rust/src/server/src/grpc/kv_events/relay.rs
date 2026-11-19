// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Duration;

use thiserror_ext::AsReport as _;
use tokio::sync::watch;
use tokio_util::task::AbortOnDropHandle;
use tracing::warn;
use uuid::Uuid;
use vllm_engine_core_client::kv_events::{KvEventSubscriber, SequencedKvEventBatch};
use vllm_engine_core_client::protocol::handshake::KvEventsConfig;

use super::{batch_to_proto, pb};

/// Most batches one read hands to a reader, so that a reader far behind does
/// not keep a whole history's worth of evicted batches alive.
const MAX_READ_BATCHES: usize = 256;

/// Pause after a failed read from the publisher, which fails repeatedly while
/// the connection is down.
const RETRY_DELAY: Duration = Duration::from_secs(1);

/// The cursor is not in the retained history.
#[derive(Debug)]
pub(super) struct HistoryMiss;

/// Why a reader cannot continue.
#[derive(Debug)]
pub(super) enum ReadError {
    /// Batches after the reader's cursor are gone.
    Miss,
    /// The relay was dropped with its service.
    RelayStopped,
}

impl From<HistoryMiss> for ReadError {
    fn from(_: HistoryMiss) -> Self {
        Self::Miss
    }
}

/// The retained tail of one rank's event stream: consecutive batches of one
/// stream ID.
struct History {
    stream_id: String,
    batches: VecDeque<Arc<pb::KvEventBatch>>,
}

impl History {
    fn new() -> Self {
        Self {
            stream_id: Uuid::new_v4().to_string(),
            batches: VecDeque::new(),
        }
    }

    /// Append a batch, starting a new stream when it does not continue the
    /// retained ones.
    fn push(&mut self, batch: SequencedKvEventBatch, capacity: usize) {
        if batch.follows_gap {
            *self = Self::new();
        }
        if self.batches.len() == capacity {
            self.batches.pop_front();
        }
        self.batches.push_back(Arc::new(batch_to_proto(batch)));
    }

    /// Index of the retained batch with sequence number `next`, which may be
    /// the one after the newest; of the oldest one for `None`.
    fn position(&self, stream_id: &str, next: Option<u64>) -> Result<usize, HistoryMiss> {
        if stream_id != self.stream_id {
            return Err(HistoryMiss);
        }
        match (next, self.batches.front()) {
            (Some(next), Some(oldest)) => next
                .checked_sub(oldest.sequence_number)
                .filter(|position| *position <= self.batches.len() as u64)
                .map(|position| position as usize)
                .ok_or(HistoryMiss),
            // Nothing to check the cursor against until the first batch.
            _ => Ok(0),
        }
    }
}

/// Follows one engine's KV-event publisher and retains its recent batches for
/// any number of readers.
pub(super) struct KvEventRelay {
    history: watch::Receiver<History>,
    _task: AbortOnDropHandle<()>,
}

impl KvEventRelay {
    /// Connect to the publisher and start following it. Retains as many
    /// batches as the publisher keeps for replay.
    pub async fn connect(config: &KvEventsConfig) -> vllm_engine_core_client::Result<Self> {
        let subscriber = KvEventSubscriber::connect(config).await?;
        let capacity = (config.buffer_steps as usize).max(1);
        let (sender, history) = watch::channel(History::new());
        let task = tokio::spawn(follow(subscriber, sender, capacity));
        Ok(Self {
            history,
            _task: AbortOnDropHandle::new(task),
        })
    }

    /// A reader positioned at `resume`, or at the oldest retained batch.
    pub fn reader(&self, resume: Option<pb::KvEventCursor>) -> Result<KvEventReader, HistoryMiss> {
        let mut history = self.history.clone();
        let (stream_id, next) = {
            let retained = history.borrow_and_update();
            let (stream_id, next) = match resume {
                Some(cursor) => (cursor.stream_id, Some(cursor.next_sequence_number)),
                None => (retained.stream_id.clone(), None),
            };
            retained.position(&stream_id, next)?;
            (stream_id, next)
        };
        // The first read must not wait for a change.
        history.mark_changed();
        Ok(KvEventReader {
            history,
            stream_id,
            next,
        })
    }
}

async fn follow(
    mut subscriber: KvEventSubscriber,
    history: watch::Sender<History>,
    capacity: usize,
) {
    loop {
        match subscriber.next().await {
            Ok(batch) => history.send_modify(|history| history.push(batch, capacity)),
            Err(error) => {
                warn!(error = %error.as_report(), "failed to read KV events from the engine");
                tokio::time::sleep(RETRY_DELAY).await;
            }
        }
    }
}

/// One subscriber's position in a relay's history.
pub(super) struct KvEventReader {
    history: watch::Receiver<History>,
    stream_id: String,
    /// Sequence number after the last batch read; `None` before the first.
    next: Option<u64>,
}

impl KvEventReader {
    pub fn stream_id(&self) -> &str {
        &self.stream_id
    }

    /// Up to [`MAX_READ_BATCHES`] batches after the last ones read, waiting
    /// for at least one.
    pub async fn next_batches(&mut self) -> Result<Vec<Arc<pb::KvEventBatch>>, ReadError> {
        loop {
            self.history.changed().await.map_err(|_| ReadError::RelayStopped)?;
            let batches: Vec<_> = {
                let retained = self.history.borrow_and_update();
                let position = retained.position(&self.stream_id, self.next)?;
                retained.batches.iter().skip(position).take(MAX_READ_BATCHES).cloned().collect()
            };
            if let Some(newest) = batches.last() {
                self.next = Some(newest.sequence_number + 1);
                if batches.len() == MAX_READ_BATCHES {
                    // More may be retained already.
                    self.history.mark_changed();
                }
                return Ok(batches);
            }
        }
    }
}
