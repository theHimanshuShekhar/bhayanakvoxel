//! `client bench`: a scripted flight over the Planet that times every GPU pass and writes a
//! report. See `docs/agents/benchmarking.md`.
//!
//! It draws offscreen, through [`Offscreen`], not in a window. That is the same tracer and the
//! same pass timers as the window's, minus the window's own blit and overlay passes, and it
//! brings three things a window would not: nothing waits on the display (no vsync, no
//! compositor), a frame's GPU times are in by the time it returns (no 1 to 2 frames of lag, no
//! frames still in flight when the run ends), and it needs no display at all. `Offscreen` asks
//! Vulkan for the high-performance adapter like the window does, so on the dev's machine it is the
//! RTX 4090, and the report says which GPU it got.

mod args;
mod budget;
mod path;
mod report;
mod stats;

use std::error::Error;
use std::fs::{self, File};
use std::io::BufWriter;
use std::path::Path;
use std::process::ExitCode;
use std::time::Instant;

use brick_store::{BrickCoord, BrickStore};
use render::{Camera, Image, Mode, Offscreen, Resolution, Scene};

use args::{Args, Diag, Parsed, USAGE};
use budget::{Budgets, Status};
use path::{CameraPath, Keyframe};
use report::{
    Altitude, Build, Config, KeyframeLod, Machine, Measured, ModeReport, Modes, Report,
    SCHEMA_VERSION,
};

const REPORT_FILE: &str = "report.json";

/// How much of the path to draw, how, and on what.
struct Plan {
    /// The traced (internal) resolution. The image presented is twice as big in each direction:
    /// 2560x1440 for the default 1280x720.
    width: u32,
    height: u32,
    /// The test Planet, or `--short`'s patch of it.
    patch: bool,
    path_name: &'static str,
    path: CameraPath,
    primary: ModePlan,
    reference: ModePlan,
}

/// One mode's share of the run.
#[derive(Clone, Copy)]
struct ModePlan {
    /// Frames of the path that are measured.
    frames: u32,
    /// Frames drawn first, at the end of the path, and left out. They let the GPU's clocks, caches
    /// and the driver settle, and they keep the mode switch from leaking into the numbers.
    warmup: u32,
    /// Samples per frame (reference mode).
    spp: u32,
}

impl Plan {
    /// The measurement: the whole Planet at the resolution that the frame budgets are for.
    fn full() -> Self {
        Self {
            width: 1280,
            height: 720,
            patch: false,
            path_name: "planet-flyover-v1",
            path: path::planet_flyover(),
            primary: ModePlan {
                frames: 240,
                warmup: 10,
                spp: 1,
            },
            reference: ModePlan {
                frames: 60,
                warmup: 3,
                spp: 4,
            },
        }
    }

    /// A smoke test that takes seconds even on a software driver. Its times mean nothing.
    ///
    /// Generating the whole Planet takes about 22 s (release), so this makes the patch of it that
    /// the golden image test uses instead, which takes a fraction of a second.
    fn short() -> Self {
        Self {
            width: 160,
            height: 90,
            patch: true,
            path_name: "patch-flyover-v1",
            path: path::patch_flyover(),
            primary: ModePlan {
                frames: 6,
                warmup: 1,
                spp: 1,
            },
            reference: ModePlan {
                frames: 3,
                warmup: 1,
                spp: 2,
            },
        }
    }
}

/// Runs `client bench` with the arguments after `bench`. The exit status is 0 if all is well,
/// [`budget::EXIT_OVER_BUDGET`] if a pass is over budget and 1 if the bench could not run.
pub fn main(args: impl IntoIterator<Item = String>) -> ExitCode {
    match args::parse(args) {
        Ok(Parsed::Help) => {
            println!("{USAGE}");
            ExitCode::SUCCESS
        }
        Ok(Parsed::Run(args)) => match run(&args) {
            Ok(code) => ExitCode::from(code),
            Err(err) => {
                eprintln!("bench failed: {err}");
                ExitCode::FAILURE
            }
        },
        Err(err) => {
            eprintln!("{err}\n\n{USAGE}");
            ExitCode::FAILURE
        }
    }
}

fn run(args: &Args) -> Result<u8, Box<dyn Error>> {
    let (budgets, source) = match &args.budgets {
        Some(file) => {
            let text = fs::read_to_string(file)
                .map_err(|e| format!("cannot read budgets {}: {e}", file.display()))?;
            let budgets =
                Budgets::parse(&text).map_err(|e| format!("budgets {}: {e}", file.display()))?;
            (budgets, file.display().to_string())
        }
        None => (
            Budgets::parse(budget::EMBEDDED).expect("the built-in budgets parse"),
            "embedded".to_string(),
        ),
    };
    let mut plan = if args.short {
        Plan::short()
    } else {
        Plan::full()
    };
    if let Some((width, height)) = args.size {
        (plan.width, plan.height) = (width, height);
    }
    fs::create_dir_all(&args.out)
        .map_err(|e| format!("cannot create {}: {e}", args.out.display()))?;

    let started = Instant::now();
    let mut flight = if plan.patch {
        Flight {
            scene: patch_scene(),
            store: None,
            diag: Diag::default(),
        }
    } else {
        eprintln!("generating the Planet (this takes a while)...");
        let mut store = planet_gen::generate_planet(crate::SEED);
        store.build_lod(crate::RING_LEVELS - 1);
        // The bands are for the image that is presented: twice the traced height.
        let bands = render::lod::Bands::new(plan.height * 2, render::lod::VFOV, args.bias);
        let scene = crate::planet_scene(
            &store,
            plan.path.camera_at(0, 2).position,
            bands,
            planet_gen::MAX_RADIUS,
        )?;
        Flight {
            scene,
            store: Some(store),
            diag: Diag::default(),
        }
    };
    eprintln!("scene ready after {:.1?}", started.elapsed());
    let resolution = Resolution {
        internal: (plan.width, plan.height),
        output: (plan.width * 2, plan.height * 2),
    };
    let mut gpu = Offscreen::with_resolution(resolution, &flight.scene)?;
    eprintln!(
        "the Planet takes {:.1} MiB of GPU memory",
        gpu.scene_bytes() as f64 / (1024.0 * 1024.0)
    );
    let info = gpu.gpu_info().clone();
    eprintln!(
        "GPU: {} ({}, {}, {})",
        info.name, info.device_type, info.backend, info.driver
    );
    if info.device_type == "Cpu" {
        eprintln!("warning: this is a software renderer; the timings are not those of a GPU");
    }
    if !gpu.timers_supported() {
        eprintln!("warning: the GPU has no timestamp queries; passes cannot be timed");
    }
    gpu.set_keep_history(true);

    flight.diag = args.diag;
    gpu.set_flush_scene_updates(!args.diag.no_flush);
    let mut screenshots = Vec::new();
    let primary = run_mode(
        &mut gpu,
        &mut flight,
        &plan.path,
        plan.primary,
        Mode::Primary,
        &budgets,
        Some((&args.out, &mut screenshots)),
    )?;
    let reference = run_mode(
        &mut gpu,
        &mut flight,
        &plan.path,
        plan.reference,
        Mode::Reference,
        &budgets,
        Some((&args.out, &mut screenshots)),
    )?;
    let modes = Modes { primary, reference };
    let altitudes = if plan.patch {
        Vec::new()
    } else {
        sweep(&mut gpu, &mut flight, &budgets)?
    };

    let report = Report {
        schema_version: SCHEMA_VERSION,
        build: Build {
            version: env!("CARGO_PKG_VERSION").to_string(),
            commit: option_env!("GITHUB_SHA").map(str::to_string),
        },
        machine: Machine {
            gpu: info.name,
            device_type: info.device_type,
            driver: info.driver,
            driver_info: info.driver_info,
            backend: info.backend,
            os: std::env::consts::OS.to_string(),
            arch: std::env::consts::ARCH.to_string(),
            timers_supported: gpu.timers_supported(),
        },
        config: Config {
            width: plan.width,
            height: plan.height,
            output_width: resolution.output.0,
            output_height: resolution.output.1,
            scene: if plan.patch { "patch" } else { "planet" }.to_string(),
            path: plan.path_name.to_string(),
            keyframes: plan.path.keyframes(),
            bias: args.bias,
        },
        altitudes,
        screenshots,
        budgets: report::summarize(&source, &budgets, &modes),
        modes,
    };
    let json = serde_json::to_string_pretty(&report)?;
    let file = args.out.join(REPORT_FILE);
    fs::write(&file, &json).map_err(|e| format!("cannot write {}: {e}", file.display()))?;
    print_summary(&report);
    eprintln!("report and screenshots are in {}", args.out.display());
    // The report is also the only thing on stdout, to paste or to pipe.
    println!("{json}");

    let code = budget::exit_code(&report.modes.statuses(), args.no_fail);
    if code != 0 {
        let (no_data, over) = failures(&report.modes);
        if !over.is_empty() {
            eprintln!("over budget: {}", over.join(", "));
        }
        if !no_data.is_empty() {
            eprintln!("could not be timed: {}", no_data.join(", "));
        }
    }
    Ok(code)
}

/// `mode/name` of the entries with no times to check their budget, and of those over budget.
fn failures(modes: &Modes) -> (Vec<String>, Vec<String>) {
    let named = |status| modes.names_with(status);
    (named(Status::NoData), named(Status::Over))
}

/// The patch of the test Planet near the +y pole that the golden image test also renders:
/// Voxels x -440..=447, y 1360..=1575, z -360..=263.
fn patch_scene() -> Scene {
    let store = planet_gen::generate_region(
        crate::SEED,
        BrickCoord([-55, 170, -45]),
        BrickCoord([55, 196, 32]),
    );
    Scene::new(&store, None)
}

/// The Planet being flown over: its packed bricks and, for rings, the bricks they are made from.
struct Flight {
    scene: Scene,
    store: Option<BrickStore>,
    /// What `--diag` asked for.
    diag: Diag,
}

impl Flight {
    /// Moves the rings to the camera, if the scene has rings, and copies what that changed to
    /// the GPU. A window does the same before it draws a frame; the time it takes is not part
    /// of any pass.
    fn follow(&mut self, gpu: &mut Offscreen, camera: &Camera) -> Result<(), Box<dyn Error>> {
        if self.diag.gap_ms > 0.0 {
            std::thread::sleep(std::time::Duration::from_secs_f64(self.diag.gap_ms / 1e3));
        }
        if self.diag.no_follow {
            return Ok(());
        }
        if let Some(store) = &self.store
            && let Some(update) = self.scene.recentre(store, camera.position)?
            && !self.diag.no_upload
        {
            gpu.update_scene(&self.scene, &update)?;
        }
        Ok(())
    }
}

/// Draws a frame, and returns its image if `read_back`. Primary mode only reads the image back
/// for a screenshot: at 1440p it is 15 MB a frame, which would be most of the CPU time of the
/// frame and leave the GPU idle between frames (and so at lower clocks than a real frame has).
fn draw(
    gpu: &mut Offscreen,
    mode: Mode,
    camera: &Camera,
    spp: u32,
    read_back: bool,
) -> Option<Image> {
    match mode {
        Mode::Primary if read_back => Some(gpu.render_upscaled(camera)),
        Mode::Primary => {
            gpu.draw_upscaled(camera);
            None
        }
        Mode::Reference => Some(gpu.render_reference(camera, spp)),
    }
}

/// Flies `path` in one mode: warm-up frames, then the measured ones. With `shots`, a screenshot of
/// each keyframe goes to the folder, and its name to the list, and the bricks drawn at each
/// keyframe are noted; without, there are none of either.
fn run_mode(
    gpu: &mut Offscreen,
    flight: &mut Flight,
    path: &CameraPath,
    share: ModePlan,
    mode: Mode,
    budgets: &Budgets,
    mut shots: Option<(&Path, &mut Vec<String>)>,
) -> Result<ModeReport, Box<dyn Error>> {
    let name = match mode {
        Mode::Primary => "primary",
        Mode::Reference => "reference",
    };
    eprintln!(
        "{name}: {} warm-up and {} measured frames...",
        share.warmup, share.frames
    );
    let passes_before = gpu.timings().passes().len();
    // Warm up at the end of the path, away from where the measured frames start, so that frame 0
    // of the reference mode starts a new image.
    let end = path.camera_at(share.frames.saturating_sub(1), share.frames);
    flight.follow(gpu, &end)?;
    for _ in 0..share.warmup {
        draw(gpu, mode, &end, share.spp, false);
    }
    // The timers' histories hold everything since the start of the run, warm-up and the other
    // mode's frames included. From here on is this mode's.
    let timings = gpu.timings();
    let starts: Vec<usize> = timings
        .passes()
        .iter()
        .map(|p| p.times.history().len())
        .collect();
    let total_start = timings.total().history().len();

    let keyframes = path.keyframe_frames(share.frames);
    let mut wall = Vec::new();
    let mut scene_update = Vec::new();
    let mut lod_at_keyframes = Vec::new();
    for frame in 0..share.frames {
        let camera = path.camera_at(frame, share.frames);
        let following = Instant::now();
        flight.follow(gpu, &camera)?;
        scene_update.push(following.elapsed().as_secs_f64() * 1e3);
        if shots.is_some() && keyframes.contains(&frame) && flight.scene.is_rings() {
            let radius = f64::from(planet_gen::PLANET_RADIUS);
            let distance = camera.position.iter().map(|c| c * c).sum::<f64>().sqrt();
            lod_at_keyframes.push(KeyframeLod {
                frame,
                // 10 cm Voxels.
                height_m: (distance - radius) * 0.1,
                bricks: (0..render::lod::LEVELS)
                    .map(|level| flight.scene.level_bricks(level))
                    .collect(),
            });
        }
        let began = Instant::now();
        let shot = shots.is_some() && keyframes.contains(&frame);
        let image = draw(gpu, mode, &camera, share.spp, shot);
        wall.push(began.elapsed().as_secs_f64() * 1e3);
        if let (true, Some(image), Some((out, names))) = (shot, image, shots.as_mut()) {
            let file = format!("{name}-frame{frame:04}.png");
            write_png(&out.join(&file), &image)?;
            names.push(file);
        }
    }

    let timings = gpu.timings();
    let passes: Vec<(String, &[f64])> = timings
        .passes()
        .iter()
        .enumerate()
        .filter_map(|(i, pass)| {
            let new = &pass.times.history()[starts.get(i).copied().unwrap_or(0)..];
            // A pass belongs to the mode that ran it: it has times from it, or it was first used
            // in it (when the GPU cannot time passes, there are no times to tell by).
            (!new.is_empty() || i >= passes_before).then(|| (pass.name.clone(), new))
        })
        .collect();
    Ok(report::mode_report(
        name,
        &Measured {
            frames: share.frames,
            warmup_frames: share.warmup,
            spp: share.spp,
            passes,
            gpu_total: &timings.total().history()[total_start..],
            wall: &wall,
            scene_update: &scene_update,
            lod_at_keyframes,
        },
        budgets,
    ))
}

/// Heights above the Planet's mean radius that the sweep looks from, in Voxels: 4.8 m (the ground),
/// 100 m, 500 m and 2 km.
const SWEEP_HEIGHTS: [f64; 4] = [48.0, 1000.0, 5000.0, 20000.0];

/// The test Planet from a still camera at each of [`SWEEP_HEIGHTS`], looking at the ground that
/// slopes away towards the horizon: both modes' passes timed there, over a few dozen frames, and
/// the bricks the bands select. It follows the flight and does not change it.
fn sweep(
    gpu: &mut Offscreen,
    flight: &mut Flight,
    budgets: &Budgets,
) -> Result<Vec<Altitude>, Box<dyn Error>> {
    let primary = ModePlan {
        frames: 24,
        warmup: 4,
        spp: 1,
    };
    let reference = ModePlan {
        frames: 6,
        warmup: 2,
        spp: 2,
    };
    let radius = f64::from(planet_gen::PLANET_RADIUS);
    let mut altitudes = Vec::new();
    for height in SWEEP_HEIGHTS {
        let at = Keyframe {
            position: [0.0, radius + height, 250.0],
            target: [0.0, radius, -1500.0],
        };
        let path = CameraPath::new(vec![at, at]);
        eprintln!("sweep: {:.0} m", height * 0.1);
        let by_mode = |gpu: &mut Offscreen, flight: &mut Flight, share, mode| {
            run_mode(gpu, flight, &path, share, mode, budgets, None)
        };
        let primary = by_mode(gpu, flight, primary, Mode::Primary)?;
        let bricks = (0..render::lod::LEVELS)
            .map(|level| flight.scene.level_bricks(level))
            .collect();
        let reference = by_mode(gpu, flight, reference, Mode::Reference)?;
        let pass = |report: &ModeReport, name: &str| {
            report
                .passes
                .iter()
                .find(|e| e.name == name)
                .map(|e| e.stats.clone())
                .unwrap_or_default()
        };
        altitudes.push(Altitude {
            height_m: height * 0.1,
            bricks,
            primary: pass(&primary, "primary"),
            upscale: pass(&primary, "upscale"),
            reference: pass(&reference, "reference"),
        });
    }
    Ok(altitudes)
}

fn write_png(path: &Path, image: &Image) -> Result<(), Box<dyn Error>> {
    let file = BufWriter::new(
        File::create(path).map_err(|e| format!("cannot write {}: {e}", path.display()))?,
    );
    let mut encoder = png::Encoder::new(file, image.width, image.height);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    encoder.write_header()?.write_image_data(&image.rgba)?;
    Ok(())
}

/// The report in brief, for whoever is watching the run.
fn print_summary(report: &Report) {
    for a in &report.altitudes {
        let ms = |v: Option<f64>| v.map_or("n/a".to_string(), |v| format!("{v:.3}"));
        eprintln!(
            "at {:>7.1} m: primary {} (p95 {}), upscale {}, reference {} (p95 {}) ms; bricks {:?}",
            a.height_m,
            ms(a.primary.mean_ms),
            ms(a.primary.p95_ms),
            ms(a.upscale.mean_ms),
            ms(a.reference.mean_ms),
            ms(a.reference.p95_ms),
            a.bricks
        );
    }
    let ms = |v: Option<f64>| v.map_or("n/a".to_string(), |v| format!("{v:.3}"));
    for (mode, run) in [
        ("primary", &report.modes.primary),
        ("reference", &report.modes.reference),
    ] {
        eprintln!("\n{mode} ({} frames)", run.frames);
        eprintln!(
            "  {:<12} {:>9} {:>9} {:>9} {:>9}  status",
            "ms", "mean", "p95", "max", "budget"
        );
        for e in run.passes.iter().chain([&run.gpu_total]) {
            eprintln!(
                "  {:<12} {:>9} {:>9} {:>9} {:>9}  {}",
                e.name,
                ms(e.stats.mean_ms),
                ms(e.stats.p95_ms),
                ms(e.stats.max_ms),
                ms(e.budget_ms),
                serde_json::to_value(e.status)
                    .ok()
                    .and_then(|s| s.as_str().map(str::to_string))
                    .unwrap_or_default()
            );
        }
        eprintln!(
            "  {:<12} {:>9} {:>9} {:>9}",
            "frame (cpu)",
            ms(run.frame_time.mean_ms),
            ms(run.frame_time.p95_ms),
            ms(run.frame_time.max_ms)
        );
        eprintln!(
            "  {:<12} {:>9} {:>9} {:>9}",
            "scene (cpu)",
            ms(run.scene_update_ms.mean_ms),
            ms(run.scene_update_ms.p95_ms),
            ms(run.scene_update_ms.max_ms)
        );
        for k in &run.lod_at_keyframes {
            eprintln!(
                "  frame {:>4} at {:>8.1} m: bricks per level {:?}",
                k.frame, k.height_m, k.bricks
            );
        }
    }
    eprintln!();
}
