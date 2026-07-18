// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! gRPC Generate service backed by the shared [`vllm_text::TextLlm`] facade.

mod convert;
mod health;

use std::pin::Pin;
use std::sync::Arc;

use futures::{Stream, StreamExt as _};
use thiserror_ext::AsReport as _;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Request, Response, Status};
use tracing::info;
use vllm_engine_core_client::protocol::handshake::EngineCoreReadyResponse;
use vllm_text::{DecodedTextEvent, TextOutputStreamExt as _};

use self::convert::ResponseOpts;
use crate::state::AppState;

/// Generated protobuf/gRPC types for the `vllm` package.
pub mod pb {
    tonic::include_proto!("vllm");
}

pub(crate) use health::monitor_health;
pub use pb::control_server::ControlServer;
pub use pb::generate_server::GenerateServer;

pub(crate) type ControlGrpcService = ControlServer<ControlServiceImpl>;
pub(crate) type GenerateGrpcService = GenerateServer<GenerateServiceImpl>;

#[cfg(test)]
mod tests;

/// gRPC Generate service implementation backed by the shared application state.
pub struct GenerateServiceImpl {
    state: Arc<AppState>,
}

impl GenerateServiceImpl {
    pub fn new(state: Arc<AppState>) -> Self {
        Self { state }
    }
}

/// gRPC control service backed by the shared application state.
pub struct ControlServiceImpl {
    state: Arc<AppState>,
}

impl ControlServiceImpl {
    pub fn new(state: Arc<AppState>) -> Self {
        Self { state }
    }
}

#[tonic::async_trait]
impl pb::control_server::Control for ControlServiceImpl {
    async fn abort(
        &self,
        request: Request<pb::AbortRequest>,
    ) -> Result<Response<pb::AbortResponse>, Status> {
        self.state
            .engine_core_client()
            .abort(&request.into_inner().request_ids)
            .await
            .map_err(|error| Status::internal(error.to_report_string()))?;
        Ok(Response::new(pb::AbortResponse {}))
    }

    async fn get_kv_event_sources(
        &self,
        _request: Request<pb::GetKvEventSourcesRequest>,
    ) -> Result<Response<pb::GetKvEventSourcesResponse>, Status> {
        let client = self.state.engine_core_client();
        let sources = client
            .indexed_ready_responses()
            .into_iter()
            .filter_map(|(rank, response)| kv_event_source(response, rank))
            .collect();
        Ok(Response::new(pb::GetKvEventSourcesResponse { sources }))
    }
}

fn kv_event_source(
    response: &EngineCoreReadyResponse,
    data_parallel_rank: Option<u32>,
) -> Option<pb::KvEventSource> {
    if response.kv_events_publisher.as_deref() != Some("zmq") {
        return None;
    }

    let rank = data_parallel_rank.unwrap_or_default();
    let endpoint = offset_endpoint_port(response.kv_events_endpoint.as_deref()?, rank);
    let replay_endpoint = response
        .kv_events_replay_endpoint
        .as_deref()
        .map(|endpoint| offset_endpoint_port(endpoint, rank))
        .unwrap_or_default();

    Some(pb::KvEventSource {
        transport: "zmq".to_string(),
        endpoint_addr: Some(kv_endpoint_from_zmq(&endpoint)?),
        topic: response.kv_events_topic.clone().unwrap_or_default(),
        replay_endpoint,
        data_parallel_rank,
        encoding: "msgpack".to_string(),
        schema_version: 1,
        buffer_steps: response.kv_events_buffer_steps,
        hwm: response.kv_events_hwm,
        max_queue_size: response.kv_events_max_queue_size,
    })
}

fn offset_endpoint_port(endpoint: &str, data_parallel_rank: u32) -> String {
    if data_parallel_rank == 0 || endpoint.is_empty() {
        return endpoint.to_string();
    }
    if endpoint.contains("inproc") {
        return format!("{endpoint}_dp{data_parallel_rank}");
    }
    if endpoint.contains("tcp")
        && let Some((base_addr, port)) = endpoint.rsplit_once(':')
        && let Ok(base_port) = port.parse::<u32>()
    {
        return format!("{base_addr}:{}", base_port + data_parallel_rank);
    }
    endpoint.to_string()
}

fn kv_endpoint_from_zmq(endpoint: &str) -> Option<pb::KvEventEndpoint> {
    let rest = endpoint.strip_prefix("tcp://").unwrap_or(endpoint);
    let (host, port) = rest.rsplit_once(':')?;
    let port = port.parse().ok()?;
    let host = match host.trim_matches(|character| character == '[' || character == ']') {
        "*" | "0.0.0.0" | "::" | "" => advertise_host(),
        concrete => concrete.to_string(),
    };
    Some(pb::KvEventEndpoint {
        host,
        port,
        protocol: "tcp".to_string(),
    })
}

fn advertise_host() -> String {
    std::net::UdpSocket::bind("0.0.0.0:0")
        .and_then(|socket| {
            socket.connect("10.255.255.255:1")?;
            Ok(socket.local_addr()?.ip().to_string())
        })
        .unwrap_or_else(|_| "127.0.0.1".to_string())
}

#[tonic::async_trait]
impl pb::generate_server::Generate for GenerateServiceImpl {
    type GenerateStreamStream =
        Pin<Box<dyn Stream<Item = Result<pb::GenerateResponse, Status>> + Send>>;

    /// Unary generate: collect all output and return a single response.
    async fn generate(
        &self,
        request: Request<pb::GenerateRequest>,
    ) -> Result<Response<pb::GenerateResponse>, Status> {
        let proto_req = request.into_inner();
        let response_opts = ResponseOpts::from_proto(proto_req.response.as_ref());
        let text_request =
            convert::to_text_request(proto_req, false, self.state.served_model_names())?;

        let request_id = text_request.request_id.clone();
        info!(%request_id, "grpc generate (unary)");

        let stream = self.state.chat.text().generate(text_request).await;
        let stream = stream.map_err(text_error_to_status)?;

        let collected = stream.collect_output().await.map_err(text_error_to_status)?;

        // Build the single aggregated response.
        let prompt_info = convert::to_prompt_info(
            &collected.prompt_token_ids,
            collected.prompt_logprobs.as_ref(),
            &response_opts,
        );

        let finish_info = vllm_text::Finished {
            usage: collected.usage,
            finish_reason: collected.finish_reason,
            kv_transfer_params: collected.kv_transfer_params,
            ec_transfer_params: collected.ec_transfer_params,
        };

        let outputs = convert::to_sequence_output(
            &collected.text,
            &collected.token_ids,
            collected.logprobs.as_ref(),
            Some(&finish_info),
            &response_opts,
        );

        Ok(Response::new(pb::GenerateResponse {
            prompt_info: Some(prompt_info),
            outputs: Some(outputs),
        }))
    }

    /// Streaming generate: yield incremental responses as tokens are produced.
    async fn generate_stream(
        &self,
        request: Request<pb::GenerateRequest>,
    ) -> Result<Response<Self::GenerateStreamStream>, Status> {
        let proto_req = request.into_inner();
        let response_opts = ResponseOpts::from_proto(proto_req.response.as_ref());
        let text_request =
            convert::to_text_request(proto_req, true, self.state.served_model_names())?;

        let request_id = text_request.request_id.clone();
        info!(%request_id, "grpc generate (stream)");

        let stream = self.state.chat.text().generate(text_request).await;
        let stream = stream.map_err(text_error_to_status)?;

        let (tx, rx) = mpsc::channel(32);

        tokio::spawn(async move {
            futures::pin_mut!(stream);
            while let Some(event) = stream.next().await {
                let response = match event {
                    Err(e) => Err(text_error_to_status(e)),
                    Ok(DecodedTextEvent::Start {
                        prompt_token_ids,
                        prompt_logprobs,
                    }) => {
                        let prompt_info = convert::to_prompt_info(
                            &prompt_token_ids,
                            prompt_logprobs.as_ref(),
                            &response_opts,
                        );
                        Ok(pb::GenerateResponse {
                            prompt_info: Some(prompt_info),
                            outputs: None,
                        })
                    }
                    Ok(DecodedTextEvent::TextDelta {
                        delta,
                        token_ids,
                        logprobs,
                        finished,
                    }) => Ok(pb::GenerateResponse {
                        prompt_info: None,
                        outputs: Some(convert::to_sequence_output(
                            &delta,
                            &token_ids,
                            logprobs.as_ref(),
                            finished.as_ref(),
                            &response_opts,
                        )),
                    }),
                };

                if tx.send(response).await.is_err() {
                    // Client disconnected.
                    break;
                }
            }
        });

        let response_stream = ReceiverStream::new(rx);
        Ok(Response::new(Box::pin(response_stream)))
    }
}

fn text_error_to_status(error: vllm_text::Error) -> Status {
    let message = error.to_report_string();
    if error.is_request_validation_error() {
        Status::invalid_argument(message)
    } else {
        Status::internal(message)
    }
}
