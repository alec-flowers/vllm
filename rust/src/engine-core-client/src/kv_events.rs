// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Ordered subscription to one engine's KV-event publisher.

use std::collections::VecDeque;
use std::time::Duration;

use bytes::Bytes;
use thiserror_ext::AsReport as _;
use tracing::warn;
use zeromq::prelude::{Socket, SocketRecv, SocketSend};
use zeromq::{DealerSocket, SubSocket, ZmqMessage};

use crate::error::{Error, Result};
use crate::protocol::handshake::KvEventsConfig;
use crate::protocol::kv_events::KvEventBatch;

/// How long one replay may take before the missing batches count as lost.
const REPLAY_TIMEOUT: Duration = Duration::from_secs(5);
/// Sequence number of the frame that ends a replay (`ZmqEventPublisher.END_SEQ`).
const REPLAY_END_SEQUENCE_NUMBER: u64 = u64::MAX;

/// A batch with its position in the publisher's stream.
#[derive(Debug, Clone, PartialEq)]
pub struct SequencedKvEventBatch {
    /// The publisher's batch counter, starting at 0.
    pub sequence_number: u64,
    /// Whether batches between the previously yielded one and this one are
    /// missing.
    pub follows_gap: bool,
    pub batch: KvEventBatch,
}

/// Subscriber to the `ZmqEventPublisher` of one engine.
///
/// Yields each batch once, in sequence-number order. Batches the PUB socket
/// dropped are fetched from the publisher's replay socket when it has one.
// TODO: a batch dropped right before the publisher goes idle stays unnoticed
// until the next one arrives; probe the replay socket while idle.
pub struct KvEventSubscriber {
    events: SubSocket,
    replay: Option<DealerSocket>,
    /// Sequence number after the last accepted batch; `None` before the first.
    next_sequence_number: Option<u64>,
    /// Whether a batch was lost since the last yielded one.
    gap: bool,
    /// Replayed batches not yet yielded.
    replayed: VecDeque<(u64, Bytes)>,
}

impl KvEventSubscriber {
    /// Connect to the publisher described by an engine's ready response.
    ///
    /// A wildcard bind address is reached over loopback, so such a publisher
    /// must run on this host.
    pub async fn connect(config: &KvEventsConfig) -> Result<Self> {
        let mut events = SubSocket::new();
        events.connect(&connect_endpoint(&config.endpoint)?).await?;
        events.subscribe(&config.topic).await?;

        let replay = match &config.replay_endpoint {
            Some(endpoint) => {
                let mut replay = DealerSocket::new();
                replay.connect(&connect_endpoint(endpoint)?).await?;
                Some(replay)
            }
            None => None,
        };

        let mut subscriber = Self {
            events,
            replay,
            next_sequence_number: None,
            gap: false,
            replayed: VecDeque::new(),
        };
        // Batches published before the subscription took effect.
        subscriber.request_replay(0).await;
        Ok(subscriber)
    }

    /// The next batch in sequence-number order.
    pub async fn next(&mut self) -> Result<SequencedKvEventBatch> {
        loop {
            while let Some((sequence_number, payload)) = self.replayed.pop_front() {
                if let Some(batch) = self.accept(sequence_number, &payload) {
                    return Ok(batch);
                }
            }

            let (sequence_number, payload) = published_batch(self.events.recv().await?, 0)?;
            if let Some(next) = self.next_sequence_number
                && sequence_number > next
            {
                self.request_replay(next).await;
            }
            // Behind anything replayed, which covers it unless the replay failed.
            self.replayed.push_back((sequence_number, payload));
        }
    }

    /// Decode a batch unless it was already accepted, recording what it
    /// reveals as lost.
    fn accept(&mut self, sequence_number: u64, payload: &[u8]) -> Option<SequencedKvEventBatch> {
        match self.next_sequence_number {
            // A replay overlapping what the PUB socket already delivered. The
            // publisher lives as long as its engine, which outlives this
            // client, so its counter never restarts.
            Some(next) if sequence_number < next => return None,
            Some(next) if sequence_number > next => self.gap = true,
            _ => {}
        }
        self.next_sequence_number = Some(sequence_number + 1);

        match KvEventBatch::decode(payload) {
            Ok(batch) => Some(SequencedKvEventBatch {
                sequence_number,
                follows_gap: std::mem::take(&mut self.gap),
                batch,
            }),
            Err(error) => {
                warn!(
                    sequence_number,
                    error = %error.as_report(),
                    "dropping undecodable KV event batch"
                );
                self.gap = true;
                None
            }
        }
    }

    /// Queue the publisher's buffered batches from `start` on. Whatever a
    /// failed replay leaves out shows up as a gap.
    async fn request_replay(&mut self, start: u64) {
        if let Err(error) = self.replay_from(start).await {
            warn!(start, error = %error.as_report(), "KV event replay failed");
        }
    }

    async fn replay_from(&mut self, start: u64) -> Result<()> {
        let Some(replay) = &mut self.replay else {
            return Ok(());
        };
        // The empty frame is the REQ envelope the publisher's ROUTER expects.
        let request = vec![Bytes::new(), Bytes::copy_from_slice(&start.to_be_bytes())];
        replay
            .send(ZmqMessage::try_from(request).expect("replay request has frames"))
            .await?;

        loop {
            let message =
                tokio::time::timeout(REPLAY_TIMEOUT, replay.recv()).await.map_err(|_| {
                    Error::KvEventReplayTimeout {
                        timeout: REPLAY_TIMEOUT,
                    }
                })??;
            let (sequence_number, payload) = published_batch(message, 1)?;
            if sequence_number == REPLAY_END_SEQUENCE_NUMBER {
                return Ok(());
            }
            self.replayed.push_back((sequence_number, payload));
        }
    }
}

/// Split a `(topic, sequence number, payload)` message that follows
/// `envelope_frames` envelope frames.
fn published_batch(message: ZmqMessage, envelope_frames: usize) -> Result<(u64, Bytes)> {
    let mut frames = message.into_vec();
    let sequence_number = match frames.get(envelope_frames + 1).map(|frame| frame.as_ref()) {
        Some(&[b0, b1, b2, b3, b4, b5, b6, b7]) if frames.len() == envelope_frames + 3 => {
            u64::from_be_bytes([b0, b1, b2, b3, b4, b5, b6, b7])
        }
        _ => {
            return Err(Error::UnexpectedKvEventMessage {
                message: format!(
                    "expected {} frames with an 8-byte sequence number, got frame lengths {:?}",
                    envelope_frames + 3,
                    frames.iter().map(Bytes::len).collect::<Vec<_>>()
                ),
            });
        }
    };
    Ok((sequence_number, frames.swap_remove(envelope_frames + 2)))
}

/// The address that reaches a publisher bound to `endpoint`.
fn connect_endpoint(endpoint: &str) -> Result<String> {
    if endpoint.starts_with("inproc://") {
        return Err(Error::UnsupportedKvEventEndpoint {
            endpoint: endpoint.to_string(),
            message: "an inproc endpoint is only reachable inside the engine process".to_string(),
        });
    }
    let Some((host, port)) =
        endpoint.strip_prefix("tcp://").and_then(|address| address.rsplit_once(':'))
    else {
        return Ok(endpoint.to_string());
    };
    let host = match host {
        "*" | "0.0.0.0" => "127.0.0.1",
        "[::]" => "[::1]",
        host => host,
    };
    Ok(format!("tcp://{host}:{port}"))
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use tokio::sync::mpsc;
    use tokio_util::task::AbortOnDropHandle;

    use super::{KvEventSubscriber, SequencedKvEventBatch, connect_endpoint};
    use crate::protocol::kv_events::{BlockStored, KvCacheEvent, KvEventBatch};
    use crate::test_utils::{IpcNamespace, MockKvEventPublisher};

    fn batch(token_id: u32) -> KvEventBatch {
        KvEventBatch {
            ts: 0.0,
            events: vec![KvCacheEvent::BlockStored(BlockStored {
                token_ids: vec![token_id],
                ..Default::default()
            })],
            data_parallel_rank: Some(0),
        }
    }

    /// Yield a subscriber's batches as `(sequence number, follows gap)`.
    async fn follow(
        publisher: &MockKvEventPublisher,
    ) -> (mpsc::UnboundedReceiver<(u64, bool)>, AbortOnDropHandle<()>) {
        let mut subscriber = KvEventSubscriber::connect(&publisher.config()).await.unwrap();
        let (sender, receiver) = mpsc::unbounded_channel();
        let task = tokio::spawn(async move {
            loop {
                let SequencedKvEventBatch {
                    sequence_number,
                    follows_gap,
                    ..
                } = subscriber.next().await.unwrap();
                let _ = sender.send((sequence_number, follows_gap));
            }
        });
        (receiver, AbortOnDropHandle::new(task))
    }

    /// Publish a batch until the subscriber yields something. A PUB socket
    /// drops what it sends before the subscription reaches it, and the
    /// subscriber ignores the repeats.
    async fn publish_until_received(
        publisher: &mut MockKvEventPublisher,
        sequence_number: u64,
        received: &mut mpsc::UnboundedReceiver<(u64, bool)>,
    ) -> (u64, bool) {
        loop {
            publisher
                .publish_payload(
                    sequence_number,
                    rmp_serde::to_vec_named(&batch(0)).unwrap().into(),
                )
                .await;
            if let Ok(batch) =
                tokio::time::timeout(Duration::from_millis(20), received.recv()).await
            {
                return batch.unwrap();
            }
        }
    }

    #[tokio::test]
    async fn replay_supplies_batches_the_pub_socket_did_not_deliver() {
        let ipc = IpcNamespace::new().unwrap();
        let mut publisher = MockKvEventPublisher::bind(&ipc, true).await;
        // Published before the subscriber existed.
        publisher.retain(0, &batch(0));
        publisher.retain(1, &batch(1));
        let (mut received, _task) = follow(&publisher).await;
        assert_eq!(received.recv().await, Some((0, false)));
        assert_eq!(received.recv().await, Some((1, false)));

        // Dropped by the PUB socket and noticed when the next batch arrives.
        publisher.retain(2, &batch(2));
        publisher.retain(3, &batch(3));
        assert_eq!(
            publish_until_received(&mut publisher, 3, &mut received).await,
            (2, false)
        );
        assert_eq!(received.recv().await, Some((3, false)));
    }

    #[tokio::test]
    async fn lost_and_undecodable_batches_mark_the_next_one() {
        let ipc = IpcNamespace::new().unwrap();
        let mut publisher = MockKvEventPublisher::bind(&ipc, false).await;
        let (mut received, _task) = follow(&publisher).await;
        // The first batch only sets the starting point.
        assert_eq!(
            publish_until_received(&mut publisher, 5, &mut received).await,
            (5, false)
        );

        publisher.publish(7, &batch(7)).await;
        assert_eq!(received.recv().await, Some((7, true)));
        publisher.publish(8, &batch(8)).await;
        assert_eq!(received.recv().await, Some((8, false)));

        publisher.publish_payload(9, vec![0xc1].into()).await;
        publisher.publish(10, &batch(10)).await;
        assert_eq!(received.recv().await, Some((10, true)));
    }

    #[test]
    fn wildcard_bind_addresses_connect_over_loopback() {
        assert_eq!(
            connect_endpoint("tcp://*:5557").unwrap(),
            "tcp://127.0.0.1:5557"
        );
        assert_eq!(
            connect_endpoint("tcp://0.0.0.0:5557").unwrap(),
            "tcp://127.0.0.1:5557"
        );
        assert_eq!(
            connect_endpoint("tcp://[::]:5557").unwrap(),
            "tcp://[::1]:5557"
        );
        assert_eq!(
            connect_endpoint("tcp://10.0.0.7:5557").unwrap(),
            "tcp://10.0.0.7:5557"
        );
        assert_eq!(connect_endpoint("ipc:///tmp/kv").unwrap(), "ipc:///tmp/kv");
        assert!(connect_endpoint("inproc://kv").is_err());
    }
}
