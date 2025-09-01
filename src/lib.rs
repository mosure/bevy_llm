//! bevy_llm (minimal): a thin bevy wrapper over the `llm` crate.
//!
//! - re-exports `llm` chat/types so you don't duplicate data models.
//! - streams deltas and tool-calls as bevy events.
//! - lets the `llm` provider manage history (via builder memory).
//! - never blocks the main thread: on native we spawn onto a tiny tokio
//!   runtime (no bevy pool blocking); on wasm we use bevy's async pool,
//!   which yields to the browser/event loop.
//!
//! api docs (types & traits): https://docs.rs/llm
//!   - chat provider:             `llm::chat::ChatProvider`
//!   - message builder/roles:     `llm::chat::{ChatMessage, ChatRole, MessageType}`
//!   - streaming:                 `llm::chat::{StreamResponse, StreamChoice, StreamDelta}`
//!   - tools / tool calls:        `llm::builder::FunctionBuilder`, `llm::chat::ToolChoice`, `llm::ToolCall`

use bevy::prelude::*;
use bevy::tasks::futures_lite::StreamExt;
use bevy::tasks::AsyncComputeTaskPool;
use std::any::type_name_of_val;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use flume::{Receiver, Sender, TryRecvError};

/// re-export the llm types so downstream code can use the same structs/enums.
pub use llm::{
    builder::{FunctionBuilder, LLMBackend, LLMBuilder},
    chat::{
        ChatMessage, ChatProvider, ChatRole, MessageType, StreamChoice, StreamDelta,
        StreamResponse, ToolChoice,
    },
    error::LLMError,
    LLMProvider,
    ToolCall,
};

/// a map of ready-to-use `llm` providers.
///
/// - `default`: used when a `ChatSession` doesn't specify a `key`
/// - `per_key`: named providers if you want multiple backends/models
#[derive(Resource, Clone)]
pub struct Providers {
    pub default: Arc<dyn LLMProvider>,
    pub per_key: HashMap<String, Arc<dyn LLMProvider>>,
}

impl Providers {
    pub fn new(default: Arc<dyn LLMProvider>) -> Self {
        Self { default, per_key: HashMap::new() }
    }
    pub fn with(mut self, key: impl Into<String>, provider: Arc<dyn LLMProvider>) -> Self {
        self.per_key.insert(key.into(), provider);
        self
    }
    fn get(&self, key: Option<&String>) -> Arc<dyn LLMProvider> {
        if let Some(k) = key {
            self.per_key.get(k).cloned().unwrap_or_else(|| self.default.clone())
        } else {
            self.default.clone()
        }
    }
}

/// on native we keep a tiny tokio runtime to drive `llm` futures.
/// we spawn onto this rt from compute tasks so neither the main thread
/// nor bevy's compute pools block.
#[cfg(not(target_arch = "wasm32"))]
#[derive(Resource, Clone)]
pub struct TokioRt(pub Arc<tokio::runtime::Runtime>);

#[cfg(not(target_arch = "wasm32"))]
impl Default for TokioRt {
    fn default() -> Self {
        info!(target: "bevy_llm", "BevyLlm: initializing Tokio multi-thread runtime (native)");
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .expect("tokio runtime");
        Self(Arc::new(rt))
    }
}

/// system ordering so uis can run after we emit events
#[derive(SystemSet, Debug, Hash, PartialEq, Eq, Clone)]
pub enum LlmSet {
    /// bevy_llm emits Chat* events here (in `Update`)
    Drain,
}

/// attach this to an entity you want to chat with a provider.
#[derive(Component, Clone, Debug, Default)]
pub struct ChatSession {
    /// optional key to pick a provider from `Providers::per_key`.
    pub key: Option<String>,
    /// whether to use streaming (`chat_stream_struct`) or one-shot (`chat`).
    pub stream: bool,
}

/// insert this component to trigger a chat request for the session entity.
/// the provider manages the history; you only provide the *new* messages.
#[derive(Component, Clone, Debug)]
pub struct ChatRequest {
    pub messages: Vec<ChatMessage>,
}

/// helper to enqueue a text user message on a session entity.
pub fn send_user_text(commands: &mut Commands, target: Entity, text: impl Into<String>) {
    let text = text.into();
    info!(target: "bevy_llm", "send_user_text -> '{}' (len={})", text, text.len());
    let msg = ChatMessage::user().content(text).build();
    commands.entity(target).insert(ChatRequest { messages: vec![msg] });
}

/// events emitted by the wrapper during/after chat.
#[derive(Event, Debug)]
pub struct ChatStarted {
    pub entity: Entity,
}
#[derive(Event, Debug)]
pub struct ChatDeltaEvt {
    pub entity: Entity,
    pub text: String,
}
#[derive(Event, Debug)]
pub struct ChatToolCallsEvt {
    pub entity: Entity,
    pub calls: Vec<ToolCall>,
}
#[derive(Event, Debug)]
pub struct ChatCompletedEvt {
    pub entity: Entity,
    /// the final assistant text if available (for non-stream or after stream).
    pub final_text: Option<String>,
    /// latest provider memory snapshot (if provider has memory configured).
    pub memory: Option<Vec<ChatMessage>>,
}
#[derive(Event, Debug)]
pub struct ChatErrorEvt {
    pub entity: Entity,
    pub error: String,
}

/// cross-thread inbox for streaming; producers send, main thread drains.
/// bounded to avoid unbounded growth when the frame stalls briefly.
#[derive(Resource, Clone)]
struct StreamInbox {
    tx: Sender<StreamMsg>,
    rx: Receiver<StreamMsg>,
}

impl Default for StreamInbox {
    fn default() -> Self {
        let (tx, rx) = flume::bounded(2048);
        Self { tx, rx }
    }
}


#[derive(Debug)]
pub enum StreamMsg {
    Begin { entity: Entity },
    Delta { entity: Entity, text: String },
    Tool  { entity: Entity, calls: Vec<ToolCall> },
    Done  { entity: Entity, final_text: Option<String>, memory: Option<Vec<ChatMessage>> },
    Err   { entity: Entity, error: String },
}

/// send to inbox (ignore full/disconnected)
fn push_inbox(tx: &Sender<StreamMsg>, msg: StreamMsg) {
    let _ = tx.send(msg);
}

/// ensure a memory snapshot includes the just-produced assistant text.
/// some providers update their internal memory *after* the stream ends,
/// so a snapshot taken immediately can miss the final assistant message.
fn merge_memory_with_final(
    mem: Option<Vec<ChatMessage>>,
    final_text: Option<&str>,
) -> Option<Vec<ChatMessage>> {
    let mut mem = match mem {
        Some(m) if !m.is_empty() => m,
        _ => return None, // keep ui state; don't replace with empty
    };
    if let Some(t) = final_text
        && !t.is_empty() {
            let need_append = match mem.last() {
                Some(last) => !(matches!(last.role, ChatRole::Assistant) && last.content == t),
                None => true,
            };
            if need_append {
                mem.push(ChatMessage::assistant().content(t.to_string()).build());
            }
    }
    Some(mem)
}

/// bevy plugin: wires systems, events, resources.
/// requires you to insert a `Providers` resource before/after adding the plugin.
/// on native, also inserts a tiny tokio runtime resource by default.
pub struct BevyLlmPlugin;

impl Plugin for BevyLlmPlugin {
    fn build(&self, app: &mut App) {
        info!(target: "bevy_llm", "BevyLlmPlugin: build()");
        app.init_resource::<StreamInbox>()
            .add_event::<ChatStarted>()
            .add_event::<ChatDeltaEvt>()
            .add_event::<ChatToolCallsEvt>()
            .add_event::<ChatCompletedEvt>()
            .add_event::<ChatErrorEvt>()
            // write + read events in the same schedule (Update)
            .configure_sets(Update, LlmSet::Drain)
            .add_systems(Update, drain_stream_inbox.in_set(LlmSet::Drain))
            // spawn requests in Update; work continues off-thread/tokio
            .add_systems(Update, spawn_chat_requests);

        #[cfg(not(target_arch = "wasm32"))]
        if app.world().get_resource::<TokioRt>().is_none() {
            app.insert_resource(TokioRt::default());
        }
    }
}

/// spawns async tasks to fulfill pending requests (compute-tasks-first).
fn spawn_chat_requests(
    mut commands: Commands,
    providers: Res<Providers>,
    inbox: Res<StreamInbox>,
    mut q: Query<(Entity, &ChatSession, &ChatRequest)>,
    mut ev_start: EventWriter<ChatStarted>,

    // native-only: small runtime to drive network futures from `llm`
    #[cfg(not(target_arch = "wasm32"))] rt: Res<TokioRt>,
) {
    for (e, session, req) in q.iter_mut() {
        let provider = providers.get(session.key.as_ref());
        let inbox_tx = inbox.tx.clone();
        let messages = req.messages.clone();
        let stream = session.stream;

        // logging: provider type + msg stats
        let pty = type_name_of_val(provider.as_ref());
        let user_msgs = messages.iter().filter(|m| matches!(m.role, ChatRole::User)).count();
        let assistant_msgs = messages.iter().filter(|m| matches!(m.role, ChatRole::Assistant)).count();
        info!(target: "bevy_llm",
            "spawn_chat_requests: entity={:?} provider={} stream={} msgs={} (user={}, assistant={})",
            e, pty, stream, messages.len(), user_msgs, assistant_msgs
        );

        // one-shot marker removal
        commands.entity(e).remove::<ChatRequest>();
        ev_start.write(ChatStarted { entity: e });

        let pool = AsyncComputeTaskPool::get();
        #[cfg(not(target_arch = "wasm32"))]
        let rt = rt.0.clone();

        // spawn an async compute task; internally we hand off to tokio (native).
        pool.spawn(async move {
            let run = async move {
                if stream {
                    // try structured streaming first.
                    match provider.chat_stream_struct(&messages).await {
                        Err(err) => {
                            warn!(target: "bevy_llm",
                                "structured streaming failed for provider {}: {err}. falling back to one-shot chat()",
                                pty
                            );
                            // fall back to one-shot
                            match provider.chat(&messages).await {
                                Err(err2) => {
                                    error!(target: "bevy_llm", "chat error: {}", err2);
                                    push_inbox(&inbox_tx, StreamMsg::Err { entity: e, error: err2.to_string() });
                                }
                                Ok(resp) => {
                                    let text = resp.text().unwrap_or_default().to_string();
                                    // only emit a snapshot when it’s non-empty; otherwise leave
                                    // memory as none so uis don’t clear their local view.
                                    let mem = provider
                                        .memory_contents()
                                        .await
                                        .and_then(|m| (!m.is_empty()).then_some(m));
                                    push_inbox(&inbox_tx, StreamMsg::Begin { entity: e });
                                    if !text.is_empty() {
                                        push_inbox(&inbox_tx, StreamMsg::Delta { entity: e, text: text.clone() });
                                    }
                                    info!(target: "bevy_llm", "chat (fallback) completed: final_len={}", text.len());
                                    let final_text = if text.is_empty() { None } else { Some(text.clone()) };
                                    let memory = merge_memory_with_final(mem, final_text.as_deref());
                                    push_inbox(&inbox_tx, StreamMsg::Done { entity: e, final_text, memory });
                                }
                            }
                        }
                        Ok(mut s) => {
                            push_inbox(&inbox_tx, StreamMsg::Begin { entity: e });
                            let mut last_text = String::new();
                            // coalesce tiny deltas to ~60hz or >=64 chars
                            const MIN_CHARS: usize = 64;
                            const MAX_LATENCY: Duration = Duration::from_millis(16);
                            let mut buf = String::new();
                            let mut last_flush = Instant::now();
                            while let Some(item) = s.next().await {
                                match item {
                                    Ok(StreamResponse { choices, .. }) => {
                                        for StreamChoice { delta: StreamDelta { content, tool_calls } } in choices {
                                            if let Some(txt) = content
                                                && !txt.is_empty() {
                                                    last_text.push_str(&txt);
                                                    buf.push_str(&txt);
                                                    let now = Instant::now();
                                                    if buf.len() >= MIN_CHARS || now.duration_since(last_flush) >= MAX_LATENCY {
                                                        let chunk = std::mem::take(&mut buf);
                                                        push_inbox(&inbox_tx, StreamMsg::Delta { entity: e, text: chunk });
                                                        last_flush = now;
                                                    }
                                            }
                                            if let Some(calls) = tool_calls
                                                && !calls.is_empty() {
                                                    debug!(target: "bevy_llm", "tool calls (chunk): {}", calls.len());
                                                    push_inbox(&inbox_tx, StreamMsg::Tool { entity: e, calls });
                                            }
                                        }
                                    }
                                    Err(err) => {
                                        error!(target: "bevy_llm", "streaming error: {}", err);
                                        // flush whatever we buffered before error
                                        if !buf.is_empty() {
                                            let chunk = std::mem::take(&mut buf);
                                            push_inbox(&inbox_tx, StreamMsg::Delta { entity: e, text: chunk });
                                        }
                                        push_inbox(&inbox_tx, StreamMsg::Err { entity: e, error: err.to_string() });
                                        return;
                                    }
                                }
                            }
                            // flush tail
                            if !buf.is_empty() {
                                let chunk = std::mem::take(&mut buf);
                                push_inbox(&inbox_tx, StreamMsg::Delta { entity: e, text: chunk });
                            }
                            let mem = provider
                                .memory_contents()
                                .await
                                .and_then(|m| (!m.is_empty()).then_some(m));
                            info!(target: "bevy_llm", "stream completed: final_len={}", last_text.len());
                            let final_text = if last_text.is_empty() { None } else { Some(last_text.clone()) };
                            let memory = merge_memory_with_final(mem, final_text.as_deref());
                            push_inbox(&inbox_tx, StreamMsg::Done { entity: e, final_text, memory });
                        }
                    }
                } else {
                    // one-shot response.
                    match provider.chat(&messages).await {
                        Err(err) => {
                            error!(target: "bevy_llm", "chat error: {}", err);
                            push_inbox(&inbox_tx, StreamMsg::Err { entity: e, error: err.to_string() });
                        }
                        Ok(resp) => {
                            let text = resp.text().unwrap_or_default().to_string();
                            let mem = provider
                                .memory_contents()
                                .await
                                .and_then(|m| (!m.is_empty()).then_some(m));
                            push_inbox(&inbox_tx, StreamMsg::Begin { entity: e });
                            if !text.is_empty() {
                                push_inbox(&inbox_tx, StreamMsg::Delta { entity: e, text: text.clone() });
                            }
                            info!(target: "bevy_llm", "chat completed: final_len={}", text.len());
                            let final_text = if text.is_empty() { None } else { Some(text.clone()) };
                            let memory = merge_memory_with_final(mem, final_text.as_deref());
                            push_inbox(&inbox_tx, StreamMsg::Done { entity: e, final_text, memory });
                        }
                    }
                }
            };

            #[cfg(target_arch = "wasm32")]
            {
                // wasm path: just await directly (no tokio).
                run.await;
            }
            #[cfg(not(target_arch = "wasm32"))]
            {
                // native: hand off to tokio so bevy pools stay free.
                let _ = rt.spawn(run).await;
            }
        })
        .detach();
    }
}

/// drains the inbox and emits user-facing events.
fn drain_stream_inbox(
    inbox: Res<StreamInbox>,
    mut ev_delta: EventWriter<ChatDeltaEvt>,
    mut ev_tool: EventWriter<ChatToolCallsEvt>,
    mut ev_done: EventWriter<ChatCompletedEvt>,
    mut ev_err: EventWriter<ChatErrorEvt>,
) {
    // drain up to a cap per frame to avoid long frames on bursty streams
    const MAX_PER_FRAME: usize = 512;
    let mut drained = Vec::with_capacity(64);
    for _ in 0..MAX_PER_FRAME {
        match inbox.rx.try_recv() {
            Ok(m) => drained.push(m),
            Err(TryRecvError::Empty) => break,
            Err(TryRecvError::Disconnected) => break,
        }
    }
    if drained.is_empty() { return; }

    // aggregate deltas per entity so ui applies a single push per entity per frame
    let mut delta_map: HashMap<Entity, String> = HashMap::new();
    let mut tools: Vec<(Entity, Vec<ToolCall>)> = Vec::new();
    let mut dones: Vec<(Entity, Option<String>, Option<Vec<ChatMessage>>)> = Vec::new();
    let mut errs: Vec<(Entity, String)> = Vec::new();

    for ev in drained {
        match ev {
            StreamMsg::Begin { .. } => { /* optional: debug */ }
            StreamMsg::Delta { entity, text } => {
                delta_map.entry(entity).or_default().push_str(&text);
            }
            StreamMsg::Tool { entity, calls } => tools.push((entity, calls)),
            StreamMsg::Done { entity, final_text, memory } => dones.push((entity, final_text, memory)),
            StreamMsg::Err { entity, error } => errs.push((entity, error)),
        }
    }

    for (entity, text) in delta_map {
        ev_delta.write(ChatDeltaEvt { entity, text });
    }
    for (entity, calls) in tools {
        ev_tool.write(ChatToolCallsEvt { entity, calls });
    }
    // ensure deltas land before "done" for the same frame
    for (entity, final_text, memory) in dones {
        ev_done.write(ChatCompletedEvt { entity, final_text, memory });
    }
    for (entity, error) in errs {
        ev_err.write(ChatErrorEvt { entity, error });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::app::AppExit;

    #[test]
    fn attach_request_via_send_user_text() {
        let mut app = App::new();
        app.add_plugins(MinimalPlugins);
        app.add_event::<AppExit>();

        let e = app.world_mut().spawn(ChatSession { key: None, stream: false }).id();

        {
            let mut commands = app.world_mut().commands();
            super::send_user_text(&mut commands, e, "hello world");
        }
        app.world_mut().flush();

        let req = app.world().entity(e).get::<ChatRequest>().expect("ChatRequest exists");
        assert_eq!(req.messages.len(), 1);
        let m = &req.messages[0];
        match m.role {
            ChatRole::User => {}
            _ => panic!("expected ChatRole::User"),
        }
        assert_eq!(m.content, "hello world");
    }

    #[test]
    fn drain_stream_emits_events() {
        let mut app = App::new();
        app.add_plugins(MinimalPlugins);
        app.add_event::<ChatDeltaEvt>();
        app.add_event::<ChatToolCallsEvt>();
        app.add_event::<ChatCompletedEvt>();
        app.add_event::<ChatErrorEvt>();
        app.insert_resource(StreamInbox::default());
        app.add_systems(Update, super::drain_stream_inbox);

        let e = app.world_mut().spawn_empty().id();

        {
            // send via bounded channel (new inbox api)
            let tx = app.world().resource::<StreamInbox>().tx.clone();
            tx.send(super::StreamMsg::Delta {
                entity: e,
                text: "hi ".into(),
            })
            .unwrap();
            tx.send(super::StreamMsg::Done {
                entity: e,
                final_text: Some("hi".into()),
                memory: None,
            })
            .unwrap();
        }

        // run the system once to drain inbox and emit events this frame
        app.update();

        // IMPORTANT CHANGE: use `drain()` (not `update_drain()`)
        {
            let mut ev = app.world_mut().resource_mut::<Events<ChatDeltaEvt>>();
            let deltas: Vec<_> = ev.drain().collect();
            assert!(!deltas.is_empty(), "expected at least one delta");
            assert_eq!(deltas[0].text, "hi ");
        }
        {
            let mut ev = app.world_mut().resource_mut::<Events<ChatCompletedEvt>>();
            let done: Vec<_> = ev.drain().collect();
            assert_eq!(done.len(), 1);
            assert_eq!(done[0].final_text.as_deref(), Some("hi"));
        }
        {
            let mut ev = app.world_mut().resource_mut::<Events<ChatErrorEvt>>();
            let errs: Vec<_> = ev.drain().collect();
            assert!(errs.is_empty(), "no errors expected");
        }
    }
}
