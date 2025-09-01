# bevy_llm 🤖

[![test](https://github.com/mosure/bevy_llm/workflows/test/badge.svg)](https://github.com/mosure/bevy_llm/actions?query=workflow%3Atest)
[![GitHub License](https://img.shields.io/github/license/mosure/bevy_llm)](https://github.com/mosure/bevy_llm)
[![crates.io](https://img.shields.io/crates/v/bevy_llm.svg)](https://crates.io/crates/bevy_llm)

bevy llm plugin (native + wasm). minimal wrapper over the `llm` crate that:

- re-exports `llm` chat/types, so you don’t duplicate models
- streams assistant deltas and tool-calls as Bevy events
- keeps history inside the provider (optional sliding-window memory)
- never blocks the main thread (tiny Tokio RT on native; async pool on wasm)


## install crate

```bash
cargo add bevy_llm
# or in Cargo.toml: bevy_llm = "0.2"
```


## capabilities

- [X] Bevy plugin with non-blocking async chat
- [X] Structured streaming with coalesced deltas (~60hz or >=64 chars)
- [X] Fallback to one-shot chat when streaming unsupported
- [X] Tool-calls surfaced via `ChatToolCallsEvt`
- [X] Provider-managed memory with `sliding_window_memory`
- [X] Multiple providers via `Providers` + optional `ChatSession.key`
- [X] Native + wasm (wasm uses `gloo-net`)
- [X] Helper `send_user_text()` API
- [ ] Built-in UI widgets
- [ ] Persisted conversation storage
- [ ] Additional backends convenience builders


## usage

```rust
use bevy::prelude::*;
use bevy_llm::{
    BevyLlmPlugin, Providers, ChatSession, ChatDeltaEvt, ChatCompletedEvt,
    LLMBackend, LLMBuilder, send_user_text,
};

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(BevyLlmPlugin)
        .add_systems(Startup, setup)
        // read chat events after the plugin drains its inbox
        .add_systems(Update, on_events.after(bevy_llm::LlmSet::Drain))
        .run();
}

fn setup(mut commands: Commands) {
    // OpenAI-compatible backend (point base_url at your server)
    let provider = LLMBuilder::new()
        .backend(LLMBackend::OpenAI)
        .base_url("https://api.openai.com/v1/responses")
        .model("gpt-5")
        .sliding_window_memory(16)
        .build().expect("provider");

    commands.insert_resource(Providers::new(provider.into()));

    // start a streaming chat session and send a message
    let session = commands.spawn(ChatSession { key: None, stream: true }).id();
    send_user_text(&mut commands, session, "hello from bevy_llm!");
}

fn on_events(
    mut deltas: EventReader<ChatDeltaEvt>,
    mut done: EventReader<ChatCompletedEvt>,
){
    for e in deltas.read() { println!("delta: {}", e.text); }
    for e in done.read()   { println!("final: {:?}", e.final_text); }
}
```


## examples

- `chat`: simple text streaming UI with base url / key / model fields
- `tool`: demonstrates parsing JSON-as-text and handling `ChatToolCallsEvt`

run (native):

```bash
# optional env for examples
export OPENAI_API_KEY=sk-...
export LLM_BASE_URL=https://api.openai.com
export LLM_MODEL=gpt-5

cargo run --example chat
cargo run --example tool
```

wasm is supported; integrate with your preferred bundler and target `wasm32-unknown-unknown`.


## backends

Configured via the upstream `llm` crate. This plugin works great with OpenAI‑compatible servers
(set `base_url` to your `/v1/responses` endpoint). Additional convenience builders may land later.


## compatible bevy versions

| `bevy_llm` | `bevy` |
| :--        | :--    |
| `0.2`      | `0.16` |


## license
licensed under either of

 - Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
 - MIT license (http://opensource.org/licenses/MIT)

at your option.


## contribution

unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any
additional terms or conditions.
