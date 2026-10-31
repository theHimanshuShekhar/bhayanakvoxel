//! The JSON report: what the bench measured, where, how, and how it fared against the budgets.
//! The shape of these types is the report's schema; bump [`SCHEMA_VERSION`] when it changes.

use serde::Serialize;

use super::budget::{Budgets, Status, status};
use super::stats::Stats;

pub const SCHEMA_VERSION: u32 = 3;

/// One timed thing (a pass, or the GPU total of a profiler frame) in one mode, against its
/// budget. In reference mode a profiler frame is one sample, so the entries of that mode have
/// `frames * spp` samples.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Entry {
    pub name: String,
    #[serde(flatten)]
    pub stats: Stats,
    pub budget_ms: Option<f64>,
    pub status: Status,
}

/// One mode's run over the path.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ModeReport {
    /// Frames of the path that were measured.
    pub frames: u32,
    /// Frames drawn before those and left out.
    pub warmup_frames: u32,
    /// Samples taken per frame of the path (more than 1 in reference mode only). Each sample is
    /// timed on its own, so `passes` and `gpu_total` are per sample, not per frame of the path.
    pub spp: u32,
    pub passes: Vec<Entry>,
    pub gpu_total: Entry,
    /// CPU time to draw one frame of the path and read the image back; in reference mode, a
    /// frame is all of its samples.
    pub frame_time: Stats,
    /// CPU time per frame to follow the camera with the Planet: select the bricks the LOD bands
    /// ask for, pack the ones that changed, and hand them to the GPU. It is done before the
    /// frame is timed, so it is in no pass and not in `frame_time`.
    pub scene_update_ms: Stats,
    /// What the bands selected at each keyframe of the path. Empty for a scene that is not
    /// drawn in levels (`--short`).
    pub lod_at_keyframes: Vec<KeyframeLod>,
}

/// A still camera at one height above the test Planet: the times of the passes there, and the
/// bricks the bands select for it.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Altitude {
    /// Above the Planet's mean radius, in metres.
    pub height_m: f64,
    /// Bricks drawn at each level, level 0 first.
    pub bricks: Vec<usize>,
    /// The `primary` and `upscale` passes of primary mode, and the `reference` pass of reference
    /// mode (one sample): no budget is held against them.
    pub primary: Stats,
    pub upscale: Stats,
    pub reference: Stats,
}

/// The bricks drawn at each level of detail from one keyframe of the path.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct KeyframeLod {
    /// The frame of the run the camera is at the keyframe.
    pub frame: u32,
    /// How high the camera is above the Planet's mean radius, in metres.
    pub height_m: f64,
    /// Bricks drawn at each level, level 0 first.
    pub bricks: Vec<usize>,
}

/// The times that one mode's measured frames recorded, as the GPU timers kept them.
pub struct Measured<'a> {
    pub frames: u32,
    pub warmup_frames: u32,
    pub spp: u32,
    /// `(pass, times)` for each pass that ran.
    pub passes: Vec<(String, &'a [f64])>,
    pub gpu_total: &'a [f64],
    pub wall: &'a [f64],
    /// CPU milliseconds to follow the camera, for each frame.
    pub scene_update: &'a [f64],
    pub lod_at_keyframes: Vec<KeyframeLod>,
}

fn entry(name: &str, times: &[f64], budget_ms: Option<f64>) -> Entry {
    let stats = Stats::of(times);
    Entry {
        name: name.to_string(),
        status: status(budget_ms, stats.mean_ms),
        stats,
        budget_ms,
    }
}

/// `mode` is `primary` or `reference`, which picks the budget of the whole frame.
pub fn mode_report(mode: &str, measured: &Measured, budgets: &Budgets) -> ModeReport {
    ModeReport {
        frames: measured.frames,
        warmup_frames: measured.warmup_frames,
        spp: measured.spp,
        passes: measured
            .passes
            .iter()
            .map(|(name, times)| entry(name, times, budgets.passes.get(name).copied()))
            .collect(),
        gpu_total: entry(
            "gpu_total",
            measured.gpu_total,
            budgets.gpu_total.get(mode).copied(),
        ),
        frame_time: Stats::of(measured.wall),
        scene_update_ms: Stats::of(measured.scene_update),
        lod_at_keyframes: measured.lod_at_keyframes.clone(),
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Modes {
    pub primary: ModeReport,
    pub reference: ModeReport,
}

impl Modes {
    /// Every entry, with the name of its mode.
    fn entries(&self) -> impl Iterator<Item = (&'static str, &Entry)> {
        [("primary", &self.primary), ("reference", &self.reference)]
            .into_iter()
            .flat_map(|(mode, report)| {
                report
                    .passes
                    .iter()
                    .chain([&report.gpu_total])
                    .map(move |entry| (mode, entry))
            })
    }

    /// `mode/name` of every entry with `status`.
    pub fn names_with(&self, status: Status) -> Vec<String> {
        self.entries()
            .filter(|(_, e)| e.status == status)
            .map(|(mode, e)| format!("{mode}/{}", e.name))
            .collect()
    }

    pub fn statuses(&self) -> Vec<Status> {
        self.entries().map(|(_, e)| e.status).collect()
    }
}

/// How the budgets came out, over both modes.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct BudgetSummary {
    /// `embedded`, or the path of the file.
    pub source: String,
    /// Whether nothing failed: no pass over budget, and none that could not be checked.
    pub ok: bool,
    /// `mode/name` of every entry that failed.
    pub failed: Vec<String>,
    /// Budgets that matched nothing that ran, so held nothing to account.
    pub unmatched: Vec<String>,
}

pub fn summarize(source: &str, budgets: &Budgets, modes: &Modes) -> BudgetSummary {
    let failed: Vec<String> = modes
        .entries()
        .filter(|(_, e)| e.status.fails())
        .map(|(mode, e)| format!("{mode}/{}", e.name))
        .collect();
    let unmatched = budgets
        .passes
        .keys()
        .filter(|name| !modes.entries().any(|(_, e)| &e.name == *name))
        .map(|name| format!("passes.{name}"))
        .collect();
    BudgetSummary {
        source: source.to_string(),
        ok: failed.is_empty(),
        failed,
        unmatched,
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Machine {
    pub gpu: String,
    /// `Discrete`, `Integrated`, `Virtual`, `Cpu` (software, such as lavapipe) or `Other`.
    pub device_type: String,
    pub driver: String,
    pub driver_info: String,
    pub backend: String,
    pub os: String,
    pub arch: String,
    /// Without GPU timestamp queries there are no pass times, only frame times.
    pub timers_supported: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Config {
    /// The traced (internal) resolution: what primary and reference mode trace.
    pub width: u32,
    pub height: u32,
    /// The resolution of the image primary mode presents, upscaled from the traced one.
    pub output_width: u32,
    pub output_height: u32,
    /// `planet` is the whole test Planet; `patch` the cut of it near the pole that `--short` uses.
    pub scene: String,
    /// The camera path, which has a version in its name: the same name is the same flight.
    pub path: String,
    pub keyframes: usize,
    /// The LOD bias: output pixels one Voxel may cover (`--bias`).
    pub bias: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Build {
    pub version: String,
    /// The commit the executable was built from, when CI built it.
    pub commit: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Report {
    pub schema_version: u32,
    pub build: Build,
    pub machine: Machine,
    pub config: Config,
    pub modes: Modes,
    /// The Planet from a still camera at several heights, one after the other, after the flight;
    /// empty for `--short`.
    pub altitudes: Vec<Altitude>,
    /// File names, in the output folder.
    pub screenshots: Vec<String>,
    pub budgets: BudgetSummary,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn budgets() -> Budgets {
        Budgets::parse(
            "[passes]\nprimary = 1.0\nreference = 25.0\nshadows = 0.4\n[gpu_total]\nprimary = 8.6\n",
        )
        .unwrap()
    }

    fn measured<'a>(passes: Vec<(String, &'a [f64])>, total: &'a [f64]) -> Measured<'a> {
        Measured {
            frames: 4,
            warmup_frames: 2,
            spp: 1,
            passes,
            gpu_total: total,
            wall: &[16.0, 18.0],
            scene_update: &[2.0, 4.0],
            lod_at_keyframes: vec![KeyframeLod {
                frame: 0,
                height_m: 12.5,
                bricks: vec![10, 0, 4],
            }],
        }
    }

    #[test]
    fn a_mode_lists_each_pass_with_its_stats_and_how_it_fared_against_its_budget() {
        let primary: &[f64] = &[0.5, 0.7, 0.9, 1.7];
        let blit: &[f64] = &[0.1, 0.1];
        let report = mode_report(
            "primary",
            &measured(
                vec![("primary".into(), primary), ("blit".into(), blit)],
                &[1.0, 2.0],
            ),
            &budgets(),
        );
        assert_eq!((report.frames, report.warmup_frames, report.spp), (4, 2, 1));
        let [p, b] = &report.passes[..] else {
            panic!("{:?}", report.passes)
        };
        assert_eq!(p.name, "primary");
        assert_eq!(p.stats.mean_ms, Some(0.95));
        assert_eq!((p.budget_ms, p.status), (Some(1.0), Status::Within));
        assert_eq!(b.name, "blit");
        assert_eq!((b.budget_ms, b.status), (None, Status::NoBudget));
        // The whole frame is held to the budget of this mode.
        assert_eq!(report.gpu_total.name, "gpu_total");
        assert_eq!(report.gpu_total.stats.mean_ms, Some(1.5));
        assert_eq!(report.gpu_total.budget_ms, Some(8.6));
        assert_eq!(report.frame_time.mean_ms, Some(17.0));
        assert_eq!(report.scene_update_ms.mean_ms, Some(3.0));
        assert_eq!(report.lod_at_keyframes[0].bricks, [10, 0, 4]);
    }

    #[test]
    fn a_pass_slower_than_its_budget_is_marked_over() {
        let slow: &[f64] = &[1.5, 1.5];
        let report = mode_report(
            "primary",
            &measured(vec![("primary".into(), slow)], &[1.5, 1.5]),
            &budgets(),
        );
        assert_eq!(report.passes[0].status, Status::Over);
    }

    #[test]
    fn a_pass_that_ran_without_times_is_no_data() {
        let report = mode_report(
            "primary",
            &measured(vec![("primary".into(), &[])], &[]),
            &budgets(),
        );
        assert_eq!(report.passes[0].status, Status::NoData);
        assert_eq!(report.passes[0].stats.samples, 0);
        assert_eq!(report.gpu_total.status, Status::NoData);
    }

    fn modes(primary_ms: f64) -> Modes {
        let primary: &[f64] = &[primary_ms];
        let reference: &[f64] = &[5.0];
        Modes {
            primary: mode_report(
                "primary",
                &measured(vec![("primary".into(), primary)], primary),
                &budgets(),
            ),
            reference: mode_report(
                "reference",
                &measured(vec![("reference".into(), reference)], reference),
                &budgets(),
            ),
        }
    }

    #[test]
    fn the_summary_is_ok_when_nothing_is_over() {
        let summary = summarize("embedded", &budgets(), &modes(0.5));
        assert!(summary.ok, "{summary:?}");
        assert!(summary.failed.is_empty());
        assert_eq!(summary.source, "embedded");
    }

    #[test]
    fn the_summary_names_what_failed_and_which_budgets_matched_nothing() {
        let summary = summarize("b.toml", &budgets(), &modes(3.0));
        assert!(!summary.ok);
        // 3.0 ms is over the pass's 1.0 ms, but within the 8.6 ms of the whole frame.
        assert_eq!(summary.failed, ["primary/primary"]);
        // `shadows` is budgeted but no such pass ran.
        assert_eq!(summary.unmatched, ["passes.shadows"]);
    }

    #[test]
    fn the_report_serialises_to_the_documented_shape() {
        let report = Report {
            schema_version: SCHEMA_VERSION,
            build: Build {
                version: "0.1.0".into(),
                commit: None,
            },
            machine: Machine {
                gpu: "Test GPU".into(),
                device_type: "Cpu".into(),
                driver: "d".into(),
                driver_info: "i".into(),
                backend: "Vulkan".into(),
                os: "linux".into(),
                arch: "x86_64".into(),
                timers_supported: true,
            },
            config: Config {
                width: 64,
                height: 36,
                output_width: 128,
                output_height: 72,
                scene: "patch".into(),
                path: "flyover".into(),
                keyframes: 3,
                bias: 1.0,
            },
            modes: modes(0.5),
            altitudes: vec![Altitude {
                height_m: 100.0,
                bricks: vec![5, 6],
                primary: Stats::of(&[0.2, 0.4]),
                upscale: Stats::of(&[0.1]),
                reference: Stats::of(&[1.0]),
            }],
            screenshots: vec!["primary-0.png".into()],
            budgets: summarize("embedded", &budgets(), &modes(0.5)),
        };
        let json = serde_json::to_value(&report).unwrap();
        let keys = |v: &serde_json::Value| -> Vec<String> {
            v.as_object().unwrap().keys().cloned().collect()
        };
        assert_eq!(
            keys(&json),
            [
                "altitudes",
                "budgets",
                "build",
                "config",
                "machine",
                "modes",
                "schema_version",
                "screenshots"
            ]
        );
        assert_eq!(json["schema_version"], 3);
        assert_eq!(json["build"]["commit"], serde_json::Value::Null);
        assert_eq!(json["machine"]["timers_supported"], true);
        let pass = &json["modes"]["primary"]["passes"][0];
        // Stats are flattened into the entry; a figure that is missing is null, not absent.
        assert_eq!(
            keys(pass),
            [
                "budget_ms",
                "max_ms",
                "mean_ms",
                "name",
                "p95_ms",
                "samples",
                "status"
            ]
        );
        assert_eq!(pass["status"], "within");
        assert_eq!(
            keys(&json["modes"]["primary"]),
            [
                "frame_time",
                "frames",
                "gpu_total",
                "lod_at_keyframes",
                "passes",
                "scene_update_ms",
                "spp",
                "warmup_frames"
            ]
        );
        assert_eq!(
            keys(&json["modes"]["primary"]["frame_time"]),
            ["max_ms", "mean_ms", "p95_ms", "samples"]
        );
    }
}
