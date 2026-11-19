// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

mod relay;

use std::collections::BTreeMap;
use std::pin::Pin;
use std::sync::Arc;

use asynk_strim_attr::{TryYielder, try_stream};
use futures::Stream;
use thiserror_ext::AsReport as _;
use tokio::sync::Mutex;
use tokio_util::sync::CancellationToken;
use tonic::{Request, Response, Status};
use vllm_engine_core_client::kv_events::SequencedKvEventBatch;
use vllm_engine_core_client::protocol::handshake::{EngineCoreReadyResponse, KvEventsConfig};
use vllm_engine_core_client::protocol::kv_events::{
    BlockExtraKey, BlockLocality, BlockRemoved, BlockStored, ExternalBlockHash, KvCacheEvent,
};

use self::relay::{KvEventReader, KvEventRelay, ReadError};
use super::{KvEventsServer, pb};
use crate::state::AppState;

pub(crate) type KvEventsGrpcService = KvEventsServer<KvEventsServiceImpl>;

/// gRPC KV event service: re-publishes each engine's ZMQ KV events as typed
/// streams.
pub struct KvEventsServiceImpl {
    state: Arc<AppState>,
    /// Relays by data-parallel rank, each started by its first subscriber.
    relays: Mutex<BTreeMap<u32, Arc<KvEventRelay>>>,
    /// Ends the streams, which would otherwise hold up a graceful shutdown.
    shutdown: CancellationToken,
}

impl KvEventsServiceImpl {
    pub fn new(state: Arc<AppState>, shutdown: CancellationToken) -> Self {
        Self {
            state,
            relays: Mutex::new(BTreeMap::new()),
            shutdown,
        }
    }

    /// The engine a request addresses: the given rank, or the only one.
    fn engine(&self, data_parallel_rank: Option<u32>) -> Result<&EngineCoreReadyResponse, Status> {
        let engines = self.state.engine_core_client().ready_responses();
        let connected_ranks =
            || engines.iter().map(|ready| ready.data_parallel_rank).collect::<Vec<_>>();
        match (data_parallel_rank, engines.as_slice()) {
            (None, [engine]) => Ok(engine),
            (None, _) => Err(Status::invalid_argument(format!(
                "data_parallel_rank is required; connected ranks: {:?}",
                connected_ranks()
            ))),
            (Some(rank), _) => engines
                .iter()
                .find(|ready| ready.data_parallel_rank == rank)
                .copied()
                .ok_or_else(|| {
                    Status::invalid_argument(format!(
                        "data parallel rank {rank} is not connected to this frontend; \
                             connected ranks: {:?}",
                        connected_ranks()
                    ))
                }),
        }
    }

    /// The relay of one rank, connecting it on first use.
    async fn relay(&self, rank: u32, config: &KvEventsConfig) -> Result<Arc<KvEventRelay>, Status> {
        let mut relays = self.relays.lock().await;
        if let Some(relay) = relays.get(&rank) {
            return Ok(relay.clone());
        }
        let relay = KvEventRelay::connect(config).await.map_err(|error| {
            Status::unavailable(format!(
                "failed to connect to the KV event publisher of rank {rank}: {}",
                error.as_report()
            ))
        })?;
        Ok(relays.entry(rank).or_insert(Arc::new(relay)).clone())
    }
}

#[tonic::async_trait]
impl pb::kv_events_server::KvEvents for KvEventsServiceImpl {
    type SubscribeKvEventsStream =
        Pin<Box<dyn Stream<Item = Result<pb::SubscribeKvEventsResponse, Status>> + Send>>;

    async fn subscribe_kv_events(
        &self,
        request: Request<pb::SubscribeKvEventsRequest>,
    ) -> Result<Response<Self::SubscribeKvEventsStream>, Status> {
        let request = request.into_inner();
        let engine = self.engine(request.data_parallel_rank)?;
        let rank = engine.data_parallel_rank;
        let config = engine
            .kv_events_config
            .as_ref()
            .filter(|config| config.enable_kv_cache_events && config.publisher == "zmq")
            .ok_or_else(|| {
                Status::failed_precondition(format!(
                    "the engine of rank {rank} does not publish KV cache events; \
                     enable them with --kv-events-config"
                ))
            })?;

        let reader = self.relay(rank, config).await?.reader(request.resume).map_err(|_| {
            Status::out_of_range(
                "the cursor is not in the retained KV event history; \
                 subscribe without `resume` and rebuild",
            )
        })?;
        Ok(Response::new(Box::pin(kv_event_responses(
            reader,
            rank,
            self.shutdown.clone(),
        ))))
    }
}

#[try_stream]
async fn kv_event_responses(
    mut reader: KvEventReader,
    data_parallel_rank: u32,
    shutdown: CancellationToken,
    mut y: TryYielder<pb::SubscribeKvEventsResponse, Status>,
) -> Result<(), Status> {
    use pb::subscribe_kv_events_response::Item;

    let start = pb::KvEventStreamStart {
        stream_id: reader.stream_id().to_string(),
        data_parallel_rank,
    };
    y.yield_ok(pb::SubscribeKvEventsResponse {
        item: Some(Item::Start(start)),
    })
    .await;

    loop {
        let batches = tokio::select! {
            _ = shutdown.cancelled() => {
                return Err(Status::unavailable("the server is shutting down"));
            }
            batches = reader.next_batches() => batches.map_err(|error| match error {
                ReadError::Miss => Status::data_loss(
                    "KV event batches were lost or this subscriber fell behind the retained \
                     history; subscribe without `resume` and rebuild",
                ),
                ReadError::RelayStopped => Status::unavailable("the KV event relay stopped"),
            })?,
        };
        for batch in batches {
            y.yield_ok(pb::SubscribeKvEventsResponse {
                item: Some(Item::Batch(Arc::unwrap_or_clone(batch))),
            })
            .await;
        }
    }
}

fn batch_to_proto(batch: SequencedKvEventBatch) -> pb::KvEventBatch {
    pb::KvEventBatch {
        sequence_number: batch.sequence_number,
        timestamp: batch.batch.ts,
        events: batch.batch.events.into_iter().map(event_to_proto).collect(),
    }
}

fn event_to_proto(event: KvCacheEvent) -> pb::KvCacheEvent {
    use pb::kv_cache_event::Event;

    let event = match event {
        KvCacheEvent::BlockStored(stored) => Event::Stored(stored_to_proto(stored)),
        KvCacheEvent::BlockRemoved(removed) => Event::Removed(removed_to_proto(removed)),
        KvCacheEvent::AllBlocksCleared => Event::Cleared(pb::KvAllBlocksCleared {}),
    };
    pb::KvCacheEvent { event: Some(event) }
}

fn stored_to_proto(stored: BlockStored) -> pb::KvBlocksStored {
    let extra_keys =
        stored
            .extra_keys
            .unwrap_or_default()
            .into_iter()
            .map(|keys| pb::KvBlockExtraKeys {
                keys: keys.unwrap_or_default().into_iter().map(extra_key_to_proto).collect(),
            });
    let lookahead_token_ids = stored
        .lookahead_token_ids
        .unwrap_or_default()
        .into_iter()
        .map(|token_ids| pb::KvLookaheadTokenIds { token_ids });

    pb::KvBlocksStored {
        block_hashes: hashes_to_proto(stored.block_hashes),
        parent_block_hash: stored.parent_block_hash.map(|hash| hash.0),
        token_ids: stored.token_ids,
        block_size: stored.block_size,
        medium: stored.medium,
        lora_name: stored.lora_name,
        extra_keys: extra_keys.collect(),
        group_idx: stored.group_idx,
        kv_cache_spec_kind: stored.kv_cache_spec_kind,
        kv_cache_spec_sliding_window: stored.kv_cache_spec_sliding_window,
        locality: locality_to_proto(stored.locality).into(),
        ownership: stored.ownership,
        session_id: stored.session_id,
        lookahead_token_ids: lookahead_token_ids.collect(),
    }
}

fn removed_to_proto(removed: BlockRemoved) -> pb::KvBlocksRemoved {
    pb::KvBlocksRemoved {
        block_hashes: hashes_to_proto(removed.block_hashes),
        medium: removed.medium,
        group_idx: removed.group_idx,
        locality: locality_to_proto(removed.locality).into(),
        ownership: removed.ownership,
    }
}

fn hashes_to_proto(hashes: Vec<ExternalBlockHash>) -> Vec<Vec<u8>> {
    hashes.into_iter().map(|hash| hash.0).collect()
}

fn locality_to_proto(locality: Option<BlockLocality>) -> pb::KvBlockLocality {
    match locality {
        None => pb::KvBlockLocality::Unspecified,
        Some(BlockLocality::Local) => pb::KvBlockLocality::Local,
        Some(BlockLocality::Remote) => pb::KvBlockLocality::Remote,
    }
}

fn extra_key_to_proto(key: BlockExtraKey) -> pb::KvBlockExtraKey {
    use pb::kv_block_extra_key::Key;

    let key = match key {
        BlockExtraKey::Text(text) => Key::Text(text),
        BlockExtraKey::Digest(digest) => Key::Digest(digest),
        BlockExtraKey::Multimodal(identifier, offset) => {
            Key::Multimodal(pb::KvMultimodalKey { identifier, offset })
        }
    };
    pb::KvBlockExtraKey { key: Some(key) }
}
