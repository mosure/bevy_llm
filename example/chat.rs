//! minimal bevy + bevy_llm example with openai-compatible backends.
//! - text boxes for base url, api key, model (with model discovery button).
//! - non-structured streaming by default (lib falls back automatically).
//! - lots of logging to help diagnose http 404 / streaming support issues.
//!
//! visible ui shows ONLY the most recent dialogue turn (npc-style).
//! persistent history is kept inside the llm provider and hidden.

use bevy::input::keyboard::{KeyCode, KeyboardInput};
use bevy::prelude::*;
use bevy_llm::{
    BevyLlmPlugin, ChatCompletedEvt, ChatDeltaEvt, ChatErrorEvt, ChatSession, LLMBackend, LLMBuilder,
    LLMProvider, Providers, send_user_text,
};
use std::sync::Arc;

// ---------------------- helpers: openai base url & models url ----------------------

fn normalize_oai_base(base: &str) -> String {
    // provider requires base to include `/v1` (this avoids 404s on chat endpoints).
    let b = base.trim_end_matches('/');
    if b.ends_with("/v1") {
        b.to_string()
    } else {
        format!("{}/v1", b)
    }
}

fn oai_models_url(base: &str) -> String {
    // models endpoint is `{base-with-/v1}/models`.
    format!("{}/models", normalize_oai_base(base))
}

fn responses_url(base: &str) -> String {
    format!("{}/responses", normalize_oai_base(base))
}

// ---------------------- ui tags ----------------------

#[derive(Component)]
struct HistoryText;
#[derive(Component)]
struct StreamText;
#[derive(Component)]
struct PromptText;

#[derive(Component)]
struct BaseUrlText;
#[derive(Component)]
struct ApiKeyText;
#[derive(Component)]
struct ModelText;

#[derive(Component)]
struct BtnFetchModels;
#[derive(Component)]
struct BtnApply;
#[derive(Component)]
struct BtnPrevModel;
#[derive(Component)]
struct BtnNextModel;

#[derive(Component, Copy, Clone)]
struct TargetSession(Entity);

// store the last user message for this session so we can render only the latest turn
#[derive(Component, Default, Clone)]
struct LastUserText(String);

// ---------------------- app state ----------------------

#[derive(Resource, Default, Clone)]
struct UiConfig {
    base_url: String,
    api_key: String, // visible for demo simplicity
    model: String,
}

#[derive(Resource, Default)]
struct PromptBuf(String);

#[derive(Resource, Default)]
struct ModelList {
    items: Vec<String>,
    loading: bool,
    error: Option<String>,
    selected: usize, // index into items
}

#[derive(Resource, Default)]
struct PendingModelTask(Option<bevy::tasks::Task<Result<Vec<String>, String>>>);

#[derive(Resource)]
struct Focus(FocusField);
impl Default for Focus {
    fn default() -> Self {
        Self(FocusField::Prompt)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum FocusField {
    BaseUrl,
    ApiKey,
    Prompt,
}

// ---------------------- provider helpers (dedup) ----------------------

const SYSTEM_PROMPT: &str = "you are the assistant for bevy_llm's example application. you have access to bevy via mcp. perform user actions against the bevy environment, e.g. modify properties, spawn entities, etc.";

fn build_provider(ui: &UiConfig) -> Arc<dyn LLMProvider> {
    info!(
        target: "minimal",
        "build_provider: base_url='{}', model='{}', key_present={}",
        ui.base_url, ui.model, !ui.api_key.is_empty()
    );

    let mut b = LLMBuilder::new()
        .backend(LLMBackend::OpenAI) // openai-compatible
        .base_url(responses_url(&ui.base_url))
        .model(if !ui.model.is_empty() {
            ui.model.clone()
        } else {
            "gpt-5".to_string()
        })
        .system(SYSTEM_PROMPT);
    if !ui.api_key.is_empty() {
        b = b.api_key(ui.api_key.clone());
    }
    b.build().expect("build provider").into()
}

fn apply_provider(commands: &mut Commands, ui: &UiConfig) {
    info!(target: "minimal", "apply_provider (re)installing provider");
    let provider = build_provider(ui);
    commands.insert_resource(Providers::new(provider));
}

// ---------------------- main ----------------------

fn main() {
    // seed ui config from env (users might paste "/v1"; we normalize for provider)
    let base_url =
        std::env::var("LLM_BASE_URL").unwrap_or_else(|_| "https://api.openai.com".to_string());
    let api_key = std::env::var("OPENAI_API_KEY").unwrap_or_default();
    let model = std::env::var("LLM_MODEL").unwrap_or_else(|_| "gpt-5".to_string());

    App::new()
        .insert_resource(ClearColor(Color::srgb_u8(18, 18, 20)))
        .insert_resource(UiConfig {
            base_url,
            api_key,
            model,
        })
        .insert_resource(PromptBuf::default())
        .insert_resource(ModelList::default())
        .insert_resource(Focus::default())
        .insert_resource(PendingModelTask::default())
        .add_plugins(DefaultPlugins)
        .add_plugins(BevyLlmPlugin)
        .add_systems(
            Startup,
            (bootstrap_provider, fetch_models_startup, setup).chain(),
        )
        // non-event ui + housekeeping can run anytime in Update
        .add_systems(
            Update,
            (
                handle_text_input,
                // button handlers split so we don't need "contains::<T>()"
                btn_apply,
                btn_fetch_models,
                btn_prev_model,
                btn_next_model,
                poll_model_fetch_task, // auto-apply a valid model after fetch
                refresh_config_texts,
                refresh_prompt_text,
            ),
        )
        // event readers should run after bevy_llm emits events
        .add_systems(
            Update,
            (on_delta, on_done, on_error).after(bevy_llm::LlmSet::Drain),
        )
        .run();
}

// build & insert the initial provider from UiConfig once at startup
fn bootstrap_provider(mut commands: Commands, ui: Res<UiConfig>) {
    info!(target: "minimal", "bootstrap_provider");
    apply_provider(&mut commands, &ui);
}

// also fetch models immediately at startup (to avoid 404 from invalid model ids)
fn fetch_models_startup(
    mut commands: Commands,
    ui: Res<UiConfig>,
    mut models: ResMut<ModelList>,
) {
    info!(target: "minimal", "fetch_models_startup -> {}", ui.base_url);
    if !models.loading {
        spawn_fetch_models(
            &mut commands,
            &ui.base_url,
            (!ui.api_key.is_empty()).then_some(ui.api_key.clone()),
        );
        models.loading = true;
        models.error = None;
    }
}

// ---------------------- setup ui ----------------------

fn setup(mut commands: Commands, assets: Res<AssetServer>) {
    // 0.16: camera2d (bundle-free)
    commands.spawn(Camera2d::default());

    // chat session entity (streaming on; provider may fall back)
    let session = commands
        .spawn((ChatSession { key: None, stream: true }, LastUserText::default()))
        .id();

    // ui
    let font: Handle<Font> = assets.load("fonts/Caveat-Regular.ttf");
    let style_18 = TextFont {
        font: font.clone(),
        font_size: 18.0,
        ..default()
    };
    let style_14 = TextFont {
        font: font.clone(),
        font_size: 14.0,
        ..default()
    };

    // root
    commands
        .spawn((
            Node {
                width: Val::Percent(100.0),
                height: Val::Percent(100.0),
                flex_direction: FlexDirection::Column,
                row_gap: Val::Px(8.0),
                padding: UiRect::all(Val::Px(12.0)),
                ..default()
            },
            BackgroundColor(Color::NONE),
        ))
        .with_children(|p| {
            // --- config box ---
            p.spawn((
                Node {
                    width: Val::Percent(100.0),
                    height: Val::Auto,
                    flex_direction: FlexDirection::Column,
                    row_gap: Val::Px(6.0),
                    padding: UiRect::all(Val::Px(8.0)),
                    ..default()
                },
                BackgroundColor(Color::srgb(0.10, 0.10, 0.12)),
            ))
            .with_children(|c| {
                // base url (textbox-like)
                c.spawn((
                    Text::new(""),
                    style_14.clone(),
                    TextColor(Color::WHITE),
                    BaseUrlText,
                ));
                // api key (textbox-like)
                // c.spawn((Text::new(""), style_14.clone(), TextColor(Color::WHITE), ApiKeyText));

                // row: [fetch models] [<] model [>] [apply]
                c.spawn((
                    Node {
                        width: Val::Percent(100.0),
                        height: Val::Auto,
                        column_gap: Val::Px(8.0),
                        flex_direction: FlexDirection::Row,
                        ..default()
                    },
                    BackgroundColor(Color::NONE),
                ))
                .with_children(|row| {
                    // fetch models
                    row.spawn((
                        Button,
                        Node {
                            width: Val::Px(130.0),
                            height: Val::Px(28.0),
                            align_items: AlignItems::Center,
                            justify_content: JustifyContent::Center,
                            ..default()
                        },
                        BackgroundColor(Color::srgb(0.2, 0.2, 0.25)),
                        BtnFetchModels,
                    ))
                    .with_children(|b| {
                        b.spawn((
                            Text::new("fetch models"),
                            style_14.clone(),
                            TextColor(Color::WHITE),
                        ));
                    });

                    // prev
                    row.spawn((
                        Button,
                        Node {
                            width: Val::Px(28.0),
                            height: Val::Px(28.0),
                            align_items: AlignItems::Center,
                            justify_content: JustifyContent::Center,
                            ..default()
                        },
                        BackgroundColor(Color::srgb(0.2, 0.2, 0.25)),
                        BtnPrevModel,
                    ))
                    .with_children(|b| {
                        b.spawn((Text::new("<"), style_14.clone(), TextColor(Color::WHITE)));
                    });

                    // model text
                    row.spawn((
                        Text::new("model: "),
                        style_14.clone(),
                        TextColor(Color::WHITE),
                        ModelText,
                    ));

                    // next
                    row.spawn((
                        Button,
                        Node {
                            width: Val::Px(28.0),
                            height: Val::Px(28.0),
                            align_items: AlignItems::Center,
                            justify_content: JustifyContent::Center,
                            ..default()
                        },
                        BackgroundColor(Color::srgb(0.2, 0.2, 0.25)),
                        BtnNextModel,
                    ))
                    .with_children(|b| {
                        b.spawn((Text::new(">"), style_14.clone(), TextColor(Color::WHITE)));
                    });

                    // apply
                    row.spawn((
                        Button,
                        Node {
                            width: Val::Px(90.0),
                            height: Val::Px(28.0),
                            align_items: AlignItems::Center,
                            justify_content: JustifyContent::Center,
                            ..default()
                        },
                        BackgroundColor(Color::srgb(0.2, 0.2, 0.25)),
                        BtnApply,
                        TargetSession(session),
                    ))
                    .with_children(|b| {
                        b.spawn((
                            Text::new("apply"),
                            style_14.clone(),
                            TextColor(Color::WHITE),
                        ));
                    });
                });
            });

            // --- conversation box ---
            p.spawn((
                Node {
                    width: Val::Percent(100.0),
                    height: Val::Percent(100.0),
                    flex_direction: FlexDirection::Column,
                    row_gap: Val::Px(8.0),
                    padding: UiRect::axes(Val::Px(8.0), Val::Px(12.0)),
                    ..default()
                },
                BackgroundColor(Color::NONE),
            ))
            .with_children(|c| {
                c.spawn((
                    Text::new(""),
                    style_18.clone(),
                    TextColor(Color::WHITE),
                    HistoryText,
                    TargetSession(session),
                ));
                c.spawn((
                    Text::new(""),
                    style_18.clone(),
                    TextColor(Color::srgb_u8(200, 200, 200)),
                    StreamText,
                    TargetSession(session),
                ));
                c.spawn((
                    Text::new("> "),
                    style_14.clone(),
                    TextColor(Color::WHITE),
                    PromptText,
                    TargetSession(session),
                ));
            });
        });

    // initial prompt
    let initial = "hello from bevy_llm".to_string();
    info!(target: "minimal", "sending initial prompt");
    commands.entity(session).insert(LastUserText(initial.clone()));
    send_user_text(&mut commands, session, initial);
}

// ---------------------- input & buttons ----------------------

fn handle_text_input(
    mut commands: Commands,
    mut ev_kbd: EventReader<KeyboardInput>,
    keys: Res<ButtonInput<KeyCode>>,
    mut focus: ResMut<Focus>,
    mut ui: ResMut<UiConfig>,
    mut prompt: ResMut<PromptBuf>,
    q_prompt_target: Query<&TargetSession, With<PromptText>>,
    mut q_hist: Query<(&TargetSession, &mut Text), With<HistoryText>>,
) {
    // switch focus with tab
    if keys.just_pressed(KeyCode::Tab) {
        focus.0 = match focus.0 {
            FocusField::BaseUrl => FocusField::ApiKey,
            FocusField::ApiKey => FocusField::Prompt,
            FocusField::Prompt => FocusField::BaseUrl,
        };
        info!(target: "minimal", "focus -> {:?}", focus.0);
    }

    // collect text per-focused field
    for ev in ev_kbd.read() {
        if ev.state.is_pressed() {
            if let Some(txt) = &ev.text {
                let s = txt.replace('\r', "").replace('\n', "");
                match focus.0 {
                    FocusField::BaseUrl => ui.base_url.push_str(&s),
                    FocusField::ApiKey => ui.api_key.push_str(&s),
                    FocusField::Prompt => prompt.0.push_str(&s),
                }
            }
        }
    }

    // backspace on focused field
    if keys.just_pressed(KeyCode::Backspace) {
        match focus.0 {
            FocusField::BaseUrl => {
                ui.base_url.pop();
            }
            FocusField::ApiKey => {
                ui.api_key.pop();
            }
            FocusField::Prompt => {
                prompt.0.pop();
            }
        }
    }

    // enter applies in config fields; sends message in prompt field
    if keys.just_pressed(KeyCode::Enter) {
        match focus.0 {
            FocusField::Prompt => {
                if let Ok(TargetSession(e)) = q_prompt_target.single() {
                    if !prompt.0.trim().is_empty() {
                        let msg = std::mem::take(&mut prompt.0);
                        info!(target: "minimal", "send_user_text -> '{}' (len={})", msg, msg.len());
                        // remember the last user message for this session
                        commands.entity(*e).insert(LastUserText(msg.clone()));
                        // prefill history with the user line so the ui shows the latest turn while streaming
                        for (TargetSession(t), mut h) in q_hist.iter_mut() {
                            if *t == *e {
                                h.0 = format!("history:\nuser: {}\n", msg);
                            }
                        }
                        send_user_text(&mut commands, *e, msg);
                    }
                }
            }
            _ => {
                // apply provider with current base_url/api_key/model (builder will normalize base)
                info!(target: "minimal", "enter (config) -> rebuild provider");
                rebuild_provider(&mut commands, &ui);
            }
        }
    }
}

// separate button handlers (no contains::<T>() calls)

fn btn_apply(
    mut commands: Commands,
    mut q: Query<
        (&Interaction, &TargetSession, &mut BackgroundColor),
        (Changed<Interaction>, With<BtnApply>),
    >,
    ui: Res<UiConfig>,
) {
    for (i, TargetSession(_), mut bg) in &mut q {
        match *i {
            Interaction::Pressed => {
                bg.0 = Color::srgb(0.3, 0.3, 0.35);
                info!(target: "minimal", "apply clicked -> rebuild provider");
                rebuild_provider(&mut commands, &ui);
            }
            Interaction::Hovered => bg.0 = Color::srgb(0.25, 0.25, 0.3),
            Interaction::None => bg.0 = Color::srgb(0.2, 0.2, 0.25),
        }
    }
}

fn btn_fetch_models(
    mut commands: Commands,
    mut q: Query<(&Interaction, &mut BackgroundColor), (Changed<Interaction>, With<BtnFetchModels>)>,
    mut models: ResMut<ModelList>,
    ui: Res<UiConfig>,
) {
    for (i, mut bg) in &mut q {
        match *i {
            Interaction::Pressed => {
                bg.0 = Color::srgb(0.3, 0.3, 0.35);
                if !models.loading {
                    spawn_fetch_models(
                        &mut commands,
                        &ui.base_url,
                        (!ui.api_key.is_empty()).then_some(ui.api_key.clone()),
                    );
                    models.loading = true;
                    models.error = None;
                }
            }
            Interaction::Hovered => bg.0 = Color::srgb(0.25, 0.25, 0.3),
            Interaction::None => bg.0 = Color::srgb(0.2, 0.2, 0.25),
        }
    }
}

fn btn_prev_model(
    mut commands: Commands,
    mut q: Query<(&Interaction, &mut BackgroundColor), (Changed<Interaction>, With<BtnPrevModel>)>,
    mut models: ResMut<ModelList>,
    mut ui: ResMut<UiConfig>,
) {
    for (i, mut bg) in &mut q {
        match *i {
            Interaction::Pressed => {
                bg.0 = Color::srgb(0.3, 0.3, 0.35);
                if !models.items.is_empty() {
                    models.selected = (models.selected + models.items.len() - 1) % models.items.len();
                    ui.model = models.items[models.selected].clone();
                    info!(target: "minimal", "prev model -> '{}'", ui.model);
                    rebuild_provider(&mut commands, &ui);
                }
            }
            Interaction::Hovered => bg.0 = Color::srgb(0.25, 0.25, 0.3),
            Interaction::None => bg.0 = Color::srgb(0.2, 0.2, 0.25),
        }
    }
}

fn btn_next_model(
    mut commands: Commands,
    mut q: Query<(&Interaction, &mut BackgroundColor), (Changed<Interaction>, With<BtnNextModel>)>,
    mut models: ResMut<ModelList>,
    mut ui: ResMut<UiConfig>,
) {
    for (i, mut bg) in &mut q {
        match *i {
            Interaction::Pressed => {
                bg.0 = Color::srgb(0.3, 0.3, 0.35);
                if !models.items.is_empty() {
                    models.selected = (models.selected + 1) % models.items.len();
                    ui.model = models.items[models.selected].clone();
                    info!(target: "minimal", "next model -> '{}'", ui.model);
                    rebuild_provider(&mut commands, &ui);
                }
            }
            Interaction::Hovered => bg.0 = Color::srgb(0.25, 0.25, 0.3),
            Interaction::None => bg.0 = Color::srgb(0.2, 0.2, 0.25),
        }
    }
}

// ---------------------- provider & models ----------------------

fn rebuild_provider(commands: &mut Commands, ui: &UiConfig) {
    info!(
        target: "minimal",
        "rebuild_provider: base='{}', model='{}', key_present={}",
        ui.base_url, ui.model, !ui.api_key.is_empty()
    );

    let mut b = LLMBuilder::new()
        .backend(LLMBackend::OpenAI)
        .base_url(responses_url(&ui.base_url))
        .model(if !ui.model.is_empty() {
            ui.model.clone()
        } else {
            "gpt-5".to_string()
        })
        .system(SYSTEM_PROMPT);

    if !ui.api_key.is_empty() {
        b = b.api_key(ui.api_key.clone());
    }

    let provider: Arc<dyn LLMProvider> = b.build().expect("build provider").into();
    commands.insert_resource(Providers::new(provider));
}

fn spawn_fetch_models(commands: &mut Commands, base_url: &str, api_key: Option<String>) {
    let url = oai_models_url(base_url);
    info!(target: "minimal", "spawn_fetch_models -> {}", url);

    #[cfg(not(target_arch = "wasm32"))]
    {
        use bevy::tasks::IoTaskPool;
        let api_key = api_key.clone();
        let task = IoTaskPool::get().spawn(async move {
            // ureq is blocking; do it inside this worker
            let agent = ureq::Agent::new_with_defaults();
            let mut req = agent.get(&url).header("accept", "application/json");
            if let Some(k) = api_key.as_ref() {
                req = req.header("authorization", &format!("Bearer {}", k));
            }
            let res = req.call().map_err(|e| e.to_string())?;
            // ureq 3.1: read body via Body::read_to_string()
            let text = res.into_body().read_to_string().map_err(|e| e.to_string())?;
            parse_model_ids(&text)
        });
        commands.insert_resource(PendingModelTask(Some(task)));
    }

    #[cfg(target_arch = "wasm32")]
    {
        use bevy::tasks::IoTaskPool;
        use gloo_net::http::Request;

        let api_key = api_key.clone();
        let task = IoTaskPool::get().spawn(async move {
            let mut req = Request::get(&url).header("accept", "application/json");
            if let Some(k) = api_key.as_ref() {
                req = req.header("authorization", &format!("Bearer {}", k));
            }
            let resp = req.send().await.map_err(|e| e.to_string())?;
            let text = resp.text().await.map_err(|e| e.to_string())?;
            parse_model_ids(&text)
        });
        commands.insert_resource(PendingModelTask(Some(task)));
    }
}

fn parse_model_ids(text: &str) -> Result<Vec<String>, String> {
    // expect openai-style: { "data": [ { "id": "...", ... }, ... ] }
    let v: serde_json::Value = serde_json::from_str(text).map_err(|e| e.to_string())?;
    let mut out = Vec::new();
    if let Some(arr) = v.get("data").and_then(|d| d.as_array()) {
        for item in arr {
            if let Some(id) = item.get("id").and_then(|s| s.as_str()) {
                out.push(id.to_string());
            }
        }
    }
    if out.is_empty() {
        return Err("no models found".into());
    }
    Ok(out)
}

fn poll_model_fetch_task(
    mut commands: Commands,
    mut task_res: ResMut<PendingModelTask>,
    mut models: ResMut<ModelList>,
    mut ui: ResMut<UiConfig>,
) {
    use bevy::tasks::futures_lite::future;

    if let Some(task) = task_res.0.as_mut() {
        if let Some(result) = future::block_on(future::poll_once(task)) {
            models.loading = false;
            match result {
                Ok(items) => {
                    info!(target: "minimal", "models fetched: {}", items.len());
                    models.items = items;
                    models.error = None;

                    // choose a valid model:
                    // - if user-picked model exists in list, keep it and snap selected index.
                    // - otherwise, pick first item from the list as default and re-apply provider.
                    if let Some(idx) = models.items.iter().position(|m| m == &ui.model) {
                        info!(target: "minimal", "keeping user model '{}'", ui.model);
                        models.selected = idx;
                    } else if !models.items.is_empty() {
                        models.selected = 0;
                        ui.model = models.items[0].clone();
                        info!(target: "minimal", "auto-select model '{}'", ui.model);
                        apply_provider(&mut commands, &ui);
                    }
                }
                Err(e) => {
                    warn!(target: "minimal", "model fetch error: {}", e);
                    models.error = Some(e);
                    models.items.clear();
                    models.selected = 0;
                }
            }
            task_res.0 = None;
        }
    }
}

// ---------------------- text refresh ----------------------

fn refresh_config_texts(
    ui: Res<UiConfig>,
    models: Res<ModelList>,
    focus: Res<Focus>,
    mut sets: ParamSet<(
        Query<&mut Text, With<BaseUrlText>>,
        Query<&mut Text, With<ApiKeyText>>,
        Query<&mut Text, With<ModelText>>,
    )>,
) {
    if ui.is_changed() || models.is_changed() || focus.is_changed() {
        // base url (caret shows focus) -- ascii only
        if let Ok(mut t) = sets.p0().single_mut() {
            let caret = if matches!(focus.0, FocusField::BaseUrl) {
                " |"
            } else {
                ""
            };
            t.0 = format!("base url: {}{}", ui.base_url, caret);
        }
        // api key
        if let Ok(mut t) = sets.p1().single_mut() {
            let caret = if matches!(focus.0, FocusField::ApiKey) {
                " |"
            } else {
                ""
            };
            let key = if ui.api_key.is_empty() {
                "<empty>".to_string()
            } else {
                ui.api_key.clone()
            };
            t.0 = format!("api key: {}{}", key, caret);
        }
        // model (from fetched list if present, else current ui.model)
        if let Ok(mut t) = sets.p2().single_mut() {
            let label = if models.loading {
                "model: (loading...)".to_string()
            } else if let Some(err) = &models.error {
                format!("model: [error: {err}]")
            } else if !models.items.is_empty() {
                format!("model: {}", models.items[models.selected])
            } else if !ui.model.is_empty() {
                format!("model: {}", ui.model)
            } else {
                "model: <none>".to_string()
            };
            t.0 = label;
        }
    }
}

fn refresh_prompt_text(
    prompt: Res<PromptBuf>,
    focus: Res<Focus>,
    mut q_prompt: Query<&mut Text, With<PromptText>>,
) {
    if prompt.is_changed() || focus.is_changed() {
        if let Ok(mut t) = q_prompt.single_mut() {
            let caret = if matches!(focus.0, FocusField::Prompt) {
                " |"
            } else {
                ""
            };
            t.0 = format!("> {}{}", prompt.0, caret);
        }
    }
}

// ---------------------- chat events ----------------------

fn on_delta(
    mut ev: EventReader<ChatDeltaEvt>,
    mut q: Query<(&TargetSession, &mut Text), With<StreamText>>,
) {
    use std::collections::HashMap;
    // group all deltas per-entity so we touch Text once per frame
    let mut per_entity: HashMap<Entity, String> = HashMap::new();
    for ChatDeltaEvt { entity, text } in ev.read() {
        per_entity.entry(*entity).or_default().push_str(text);
    }
    for (TargetSession(t), mut ui) in q.iter_mut() {
        if let Some(buf) = per_entity.remove(t) {
            ui.0.push_str(&buf);
        }
    }
}

fn on_done(
    mut ev: EventReader<ChatCompletedEvt>,
    mut q_hist: Query<(&TargetSession, &mut Text), With<HistoryText>>,
    mut q_stream: Query<(&TargetSession, &mut Text), (With<StreamText>, Without<HistoryText>)>,
    q_last: Query<&LastUserText>,
    mut ui: ResMut<UiConfig>,
    models: Res<ModelList>,
) {
    for ChatCompletedEvt {
        entity,
        final_text,
        memory: _,
    } in ev.read()
    {
        // grab streamed text and clear the stream line
        let mut streamed = String::new();
        for (TargetSession(t), mut s) in q_stream.iter_mut() {
            if *t == *entity {
                streamed = std::mem::take(&mut s.0);
            }
        }

        // determine assistant line (prefer final_text, else streamed)
        let assistant_line = final_text
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| streamed.trim());

        // fetch last user line for this session
        let user_line = q_last.get(*entity).ok().map(|c| c.0.as_str()).unwrap_or("");

        // render ONLY the most recent turn
        let mut out = String::from("history:\n");
        if !user_line.is_empty() {
            out.push_str("user: ");
            out.push_str(user_line);
            out.push('\n');
        }
        if !assistant_line.is_empty() {
            out.push_str("assistant: ");
            out.push_str(assistant_line);
            out.push('\n');
        }

        for (TargetSession(t), mut h) in q_hist.iter_mut() {
            if *t == *entity {
                h.0 = out.clone();
            }
        }

        // keep ui model aligned with selection if we have a list
        if !models.items.is_empty() {
            ui.model = models.items[models.selected].clone();
        }
    }
}

fn on_error(
    mut ev: EventReader<ChatErrorEvt>,
    mut q: Query<(&TargetSession, &mut Text), With<StreamText>>,
) {
    for ChatErrorEvt { entity, error } in ev.read() {
        error!(target: "minimal", "chat error (entity={:?}): {}", entity, error);
        for (TargetSession(t), mut ui) in q.iter_mut() {
            if *t == *entity {
                ui.0 = format!("ERROR: {}", error);
            }
        }
    }
}
