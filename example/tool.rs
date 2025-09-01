// examples/tool.rs
//
// spawn cubes via LLM "tool":
// - model is instructed to output JSON like:
//     {"spawn_cube":{"translation":[0,0,0],"rotation_euler_deg":[0,45,0],"scale":[1,1,1],"color_rgba":[1,0,0,1]}}
//   or an array:
//     [{"spawn_cube":{...}}, {"spawn_cube":{...}}]
//   or {"actions":[{"spawn_cube":{...}}, ...]}
//
// we also handle ChatToolCallsEvt if your provider emits real tool calls.
//
// env:
//   OPENAI_API_KEY   (key)
//   LLM_BASE_URL     (default https://api.openai.com)
//   LLM_MODEL        (default gpt-5)

use bevy::input::keyboard::{KeyCode, KeyboardInput};
use bevy::prelude::*;
use bevy_llm::{
    BevyLlmPlugin, ChatCompletedEvt, ChatDeltaEvt, ChatErrorEvt, ChatSession, ChatToolCallsEvt,
    LLMBackend, LLMBuilder, LLMProvider, Providers, ToolCall, send_user_text,
};
use serde::Deserialize;
use serde_json::Value;
use std::sync::Arc;

// ------------ config helpers ------------

fn normalize_oai_base(base: &str) -> String {
    let b = base.trim_end_matches('/');
    if b.ends_with("/v1") { b.to_string() } else { format!("{}/v1", b) }
}
fn responses_url(base: &str) -> String { format!("{}/responses", normalize_oai_base(base)) }

// ------------ ui resources ------------

#[derive(Resource, Default)] struct PromptBuf(String);
#[derive(Resource, Default)] struct StreamBuf(String);

#[derive(Component)] struct PromptText;
#[derive(Component)] struct StatusText;
#[derive(Component, Copy, Clone)] struct TargetSession(Entity);

// ------------ tool arg schema ------------

#[derive(Debug, Deserialize, Clone)]
struct SpawnCubeArgs {
    #[serde(default = "zero3")]
    translation: [f32; 3],
    #[serde(default)]
    rotation_euler_deg: Option<[f32; 3]>,
    #[serde(default)]
    scale: Option<[f32; 3]>,
    #[serde(default = "white4")]
    color_rgba: [f32; 4],
}
fn zero3() -> [f32; 3] { [0.0, 0.0, 0.0] }
fn white4() -> [f32; 4] { [1.0, 1.0, 1.0, 1.0] }

// payload variants we will accept from text
#[derive(Deserialize)]
struct One { spawn_cube: SpawnCubeArgs }
#[derive(Deserialize)]
struct Many { actions: Vec<One> }

// ------------ app ------------

fn main() {
    let base_url = std::env::var("LLM_BASE_URL").unwrap_or_else(|_| "https://api.openai.com".to_string());
    let api_key  = std::env::var("OPENAI_API_KEY").unwrap_or_default();
    let model    = std::env::var("LLM_MODEL").unwrap_or_else(|_| "gpt-5".to_string());

    App::new()
        .insert_resource(ClearColor(Color::srgb_u8(18, 18, 20)))
        .insert_resource(PromptBuf::default())
        .insert_resource(StreamBuf::default())
        .insert_resource(UiCfg { base_url, api_key, model })
        .add_plugins(DefaultPlugins)
        .add_plugins(BevyLlmPlugin)
        .add_systems(Startup, (setup_scene, setup_ui, install_provider).chain())
        .add_systems(Update, (handle_input, ui_refresh))
        .add_systems(Update, (on_delta, on_done, on_error, on_tool_calls).after(bevy_llm::LlmSet::Drain))
        .run();
}

// ------------ provider ------------

#[derive(Resource, Clone)]
struct UiCfg { base_url: String, api_key: String, model: String }

fn install_provider(mut commands: Commands, cfg: Res<UiCfg>) {
    // Instruct the model to output JSON "tool calls" in text
    let sys = "\
You are a scene assistant. When the user asks for cubes, output ONLY JSON with one of these shapes:
1) {\"spawn_cube\": {translation:[x,y,z], rotation_euler_deg:[rx,ry,rz], scale:[sx,sy,sz], color_rgba:[r,g,b,a]}}
2) [{\"spawn_cube\":{...}}, {\"spawn_cube\":{...}}]
3) {\"actions\": [{\"spawn_cube\":{...}}, ...]}
Numbers are floats. Degrees for rotation. Color channels are 0..1. No prose.";

    // IMPORTANT: enable built-in memory so the provider tracks BOTH user and assistant turns.
    // We keep the last 16 messages (adjust as you like).
    let mut b = LLMBuilder::new()
        .backend(LLMBackend::OpenAI)
        .base_url(responses_url(&cfg.base_url))
        .model(cfg.model.clone())
        .system(sys)
        .sliding_window_memory(16);

    if !cfg.api_key.is_empty() { b = b.api_key(cfg.api_key.clone()); }

    let provider: Arc<dyn LLMProvider> = b.build().expect("build provider").into();
    commands.insert_resource(Providers::new(provider));

    // Start a session
    let session = commands.spawn(ChatSession { key: None, stream: true }).id();
    commands.spawn(TargetSession(session));

    // Kick off with an example
    send_user_text(&mut commands, session, "spawn a red cube at (0,0,0) and a green cube at (2,0,0) rotated 45 deg around y");
}

// ------------ scene ------------

fn setup_scene(mut commands: Commands, mut meshes: ResMut<Assets<Mesh>>, mut mats: ResMut<Assets<StandardMaterial>>) {
    // Camera
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(8.0, 6.0, 10.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
    // Light
    commands.spawn((
        DirectionalLight::default(),
        Transform::from_xyz(10.0, 12.0, 8.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
    // Ground (bundle-free)
    let ground = meshes.add(Plane3d::default().mesh().size(20.0, 20.0));
    let mat = mats.add(StandardMaterial { base_color: Color::srgb(0.12, 0.12, 0.13), perceptual_roughness: 0.9, ..default() });
    commands.spawn((
        Mesh3d(ground),
        MeshMaterial3d(mat),
        Transform::from_translation(Vec3::new(0.0, -0.51, 0.0)),
    ));
}

// ------------ ui ------------

fn setup_ui(mut commands: Commands) {
    let style = TextFont { font_size: 16.0, ..default() };

    commands.spawn((
        Node {
            width: Val::Percent(100.0),
            height: Val::Auto,
            flex_direction: FlexDirection::Column,
            row_gap: Val::Px(6.0),
            padding: UiRect::all(Val::Px(8.0)),
            ..default()
        },
        BackgroundColor(Color::NONE),
    ))
    .with_children(|p| {
        p.spawn((Text::new("prompt:"), style.clone(), TextColor(Color::WHITE)));
        p.spawn((Text::new("> "), style.clone(), TextColor(Color::WHITE), PromptText));
        p.spawn((Text::new("status: ready"), style, TextColor(Color::WHITE), StatusText));
    });
}

fn handle_input(
    mut commands: Commands,
    mut ev_kbd: EventReader<KeyboardInput>,
    keys: Res<ButtonInput<KeyCode>>,
    mut prompt: ResMut<PromptBuf>,
    q_target: Query<&TargetSession>,
) {
    for ev in ev_kbd.read() {
        if ev.state.is_pressed() {
            if let Some(txt) = &ev.text {
                let s = txt.replace('\r', "").replace('\n', "");
                prompt.0.push_str(&s);
            }
        }
    }
    if keys.just_pressed(KeyCode::Backspace) { prompt.0.pop(); }
    if keys.just_pressed(KeyCode::Enter) {
        if let Ok(TargetSession(e)) = q_target.single() {
            if !prompt.0.trim().is_empty() {
                let msg = std::mem::take(&mut prompt.0);
                send_user_text(&mut commands, *e, msg);
            }
        }
    }
}

fn ui_refresh(
    prompt: Res<PromptBuf>,
    stream: Res<StreamBuf>,
    mut sets: ParamSet<(
        // disjoint: prompt text must NOT also be status text
        Query<&mut Text, (With<PromptText>, Without<StatusText>)>,
        // disjoint: status text must NOT also be prompt text
        Query<&mut Text, (With<StatusText>, Without<PromptText>)>,
    )>,
) {
    if prompt.is_changed() {
        if let Ok(mut t) = sets.p0().single_mut() {
            t.0 = format!("> {}", prompt.0);
        }
    }
    if stream.is_changed() {
        if let Ok(mut t) = sets.p1().single_mut() {
            t.0 = format!("status: {}", stream.0);
        }
    }
}

// ------------ event handlers ------------

fn on_delta(mut ev: EventReader<ChatDeltaEvt>, mut stream: ResMut<StreamBuf>) {
    for ChatDeltaEvt { text, .. } in ev.read() {
        stream.0.push_str(text);
        if stream.0.len() > 240 {
            let cut = stream.0.len() - 240;
            stream.0.drain(..cut);
        }
    }
}

fn on_done(
    mut ev: EventReader<ChatCompletedEvt>,
    mut stream: ResMut<StreamBuf>,
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut mats: ResMut<Assets<StandardMaterial>>,
) {
    for ChatCompletedEvt { final_text, .. } in ev.read() {
        let txt = final_text.as_deref().unwrap_or("");
        if txt.is_empty() {
            stream.0 = "done".to_string();
            continue;
        }
        let mut spawned = 0usize;
        for args in parse_spawn_args_from_text(txt) {
            if spawn_cube_from_args(&mut commands, &mut meshes, &mut mats, args).is_ok() {
                spawned += 1;
            }
        }
        stream.0 = format!("done: spawned {} cube(s)", spawned);
    }
}

fn on_error(mut ev: EventReader<ChatErrorEvt>, mut stream: ResMut<StreamBuf>) {
    for ChatErrorEvt { error, .. } in ev.read() {
        stream.0 = format!("error: {}", error);
    }
}

// Optional: if your provider emits real tool calls, handle them here.
fn on_tool_calls(
    mut ev: EventReader<ChatToolCallsEvt>,
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut mats: ResMut<Assets<StandardMaterial>>,
) {
    for ChatToolCallsEvt { calls, .. } in ev.read() {
        for call in calls {
            if let Some(args) = tool_args_json(call) {
                let _ = spawn_cube_from_args(&mut commands, &mut meshes, &mut mats, args);
            }
        }
    }
}

// ------------ tool parsing helpers ------------

fn tool_args_json(call: &ToolCall) -> Option<SpawnCubeArgs> {
    let v = serde_json::to_value(call).ok()?;
    let raw = v.get("function").and_then(|f| f.get("arguments"))
        .or_else(|| v.get("arguments"))?;
    let args_val = match raw {
        Value::String(s) => serde_json::from_str::<Value>(s).unwrap_or(Value::Null),
        other => other.clone(),
    };
    serde_json::from_value(args_val).ok()
}

// Try to recover one or more SpawnCubeArgs from assistant text.
fn parse_spawn_args_from_text(s: &str) -> Vec<SpawnCubeArgs> {
    let mut out = Vec::new();

    // 1) attempt whole string as One / Vec<One> / Many / SpawnCubeArgs
    if let Ok(One { spawn_cube }) = serde_json::from_str::<One>(s) {
        out.push(spawn_cube);
        return out;
    }
    if let Ok(v) = serde_json::from_str::<Vec<One>>(s) {
        for One { spawn_cube } in v { out.push(spawn_cube); }
        return out;
    }
    if let Ok(Many { actions }) = serde_json::from_str::<Many>(s) {
        for One { spawn_cube } in actions { out.push(spawn_cube); }
        return out;
    }
    if let Ok(args) = serde_json::from_str::<SpawnCubeArgs>(s) {
        out.push(args);
        return out;
    }

    // 2) try to extract JSON objects from free text (balanced braces)
    for obj in find_json_objects(s) {
        if let Ok(One { spawn_cube }) = serde_json::from_str::<One>(&obj) {
            out.push(spawn_cube);
            continue;
        }
        if let Ok(Many { actions }) = serde_json::from_str::<Many>(&obj) {
            for One { spawn_cube } in actions { out.push(spawn_cube); }
            continue;
        }
        if let Ok(args) = serde_json::from_str::<SpawnCubeArgs>(&obj) {
            out.push(args);
        }
    }

    out
}

fn find_json_objects(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut depth = 0usize;
    let mut start: Option<usize> = None;
    for (i, ch) in s.char_indices() {
        match ch {
            '{' => {
                if depth == 0 { start = Some(i); }
                depth += 1;
            }
            '}' => {
                if depth > 0 {
                    depth -= 1;
                    if depth == 0 {
                        if let Some(st) = start {
                            out.push(s[st..=i].to_string());
                        }
                        start = None;
                    }
                }
            }
            _ => {}
        }
    }
    out
}

// ------------ cube spawn ------------

fn spawn_cube_from_args(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    mats: &mut Assets<StandardMaterial>,
    args: SpawnCubeArgs,
) -> Result<(), String> {
    let t = vec3_from(args.translation);
    let rot = if let Some(euler_deg) = args.rotation_euler_deg {
        let r = vec3_from(euler_deg);
        let rx = r.x.to_radians();
        let ry = r.y.to_radians();
        let rz = r.z.to_radians();
        Quat::from_euler(EulerRot::XYZ, rx, ry, rz)
    } else {
        Quat::IDENTITY
    };
    let scl = if let Some(s) = args.scale { vec3_from(s) } else { Vec3::ONE };

    let c = args.color_rgba;
    let color = Color::srgba(c[0], c[1], c[2], c[3]);

    let mesh = meshes.add(Cuboid::new(1.0, 1.0, 1.0));
    let mat  = mats.add(StandardMaterial { base_color: color, ..default() });

    commands.spawn((
        Mesh3d(mesh),
        MeshMaterial3d(mat),
        Transform { translation: t, rotation: rot, scale: scl },
    ));
    Ok(())
}

fn vec3_from(a: [f32; 3]) -> Vec3 { Vec3::new(a[0], a[1], a[2]) }
