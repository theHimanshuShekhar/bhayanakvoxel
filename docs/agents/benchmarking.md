# Benchmarking

`client bench` is the performance loop of the project (decision in #13). It flies a fixed camera
path over the test Planet, times every GPU pass, and writes a JSON report and screenshots. It
exists so that "is this PR within the frame budget?" has a number for an answer, measured on the
dev's RTX 4090.

## The `needs-benchmark` label

- **Agents** add `needs-benchmark` to every PR that can change GPU timings: shaders, passes, the
  tracer, the scene's GPU layout, or what is uploaded to the GPU and how much of it. They say in
  the PR body which passes are expected to change.
- **CPU-only PRs do not get the label**, even when they are performance-sensitive (planet
  generation, brick packing, the sim hot path): the bench measures GPU passes, so it cannot see
  them. Their throughput is guarded by CI targets and tests instead, and they merge on green CI.
- **The dev** downloads the PR's `client-windows` artifact, runs the bench on the RTX 4090
  (below), and pastes `report.json` into a PR comment, together with a screenshot or two.
- **A pass over its budget blocks the merge** (#13). The dev removes the label once the report is
  in and nothing is over. Budgets are in `bins/client/bench-budgets.toml`; a PR that deliberately
  changes one says so and why.
- CI runs the bench too, but only as a smoke test on lavapipe (software Vulkan): it checks that
  it runs and that the report has the right shape, and its timings mean nothing.

## Running it

From the Windows artifact, which needs no Rust toolchain and no other files:

```
client.exe bench --out bench-out
```

With a checkout, `cargo run --release -p client -- bench --out bench-out` does the same. Build in
release: a debug build's CPU side is slow enough to distort the frame times.

| Option | Meaning |
| --- | --- |
| `--out DIR` | Where `report.json` and the screenshots go. Default `bench-out`. |
| `--budgets FILE` | Use this budgets file instead of the one built into the executable. |
| `--size WxH` | Resolution of the traced (internal) image. Default `1280x720`, the internal resolution behind 1440p output that the budgets are for; primary mode upscales it to twice that in each direction (2560x1440). Do not change it for a report you paste. |
| `--bias F` | LOD bias: output pixels one Voxel may cover, default 1. Every switch distance between levels of detail scales with it (see `CONTEXT.md`, LOD band); 2 does about a quarter of the work for up to a pixel more of error. The report records it. A report with another bias is not comparable with the budgets. |
| `--diag LIST` | Switches for finding out where time goes, separated by commas: `no-follow` (the Planet does not follow the camera), `no-upload` (it does, on the CPU, but the GPU is not told), `no-flush` (writes to the GPU go with the next frame's commands, not at once) and `gap-ms=F` (sleep F ms before each frame, as the CPU work of following would). A report made with any of them is not a measurement. See "Reading a regression" below. |
| `--short` | A tiny run (160x90, a handful of frames, a patch of the Planet): the CI smoke test. |
| `--no-fail` | Exit with status 0 even if a pass is over budget. |

It runs unattended and takes a minute or so (generating the whole Planet is about 22 s of it).
The report is written to `DIR/report.json` and also printed on stdout; progress and a table go
to stderr.

**Exit status:** 0 if every budget is met, 2 if a pass is over its budget (or has a budget but
could not be timed), 1 if the bench could not run (no Vulkan device, an unreadable budgets
file, a bad option).

### What it does

1. Generates the test Planet and uploads it to the GPU, each brick at the level of detail its
   distance from the camera says (the LOD bands, ADR 0002), and prints how much GPU memory they
   take and how many bricks each level has. The bands follow the camera along the path: selecting
   again and packing the bricks that change level, and writing them to the GPU, is done before a
   frame is timed, so it is in no pass and not in the frame's CPU time either.
2. Flies the camera path in **primary mode**, then in **reference mode** (the progressive path
   tracer, 4 samples per frame of the path). Each mode first draws a few warm-up frames, which are
   not measured: the first frames after a mode switch carry the other mode's state (see the notes
   on #28).
   Primary mode traces at `config.width` x `config.height` and upscales to `output_*` (passes
   `primary` and `upscale`); reference mode traces at the traced size, with no upscaling. Primary
   mode reads an image back only for a screenshot (a 1440p image is 15 MB), so that the frame is
   not mostly readback with the GPU idle in between; `frame_time` is then the CPU time of the
   frame as drawn, and a screenshot frame is slower.
3. Writes a screenshot at each keyframe of the path, per mode.
4. Then, after the flight, holds the camera still at four heights above the Planet's mean radius
   (4.8 m, which is the ground, 100 m, 500 m and 2 km), looking at the ground that slopes away
   towards the horizon, and times both modes there for a few dozen frames each (24 primary frames
   and 6 reference frames of 2 samples). That is the report's `altitudes`. It takes a few seconds
   and leaves the flight alone, so the flight stays comparable with the reports of main. `--short`
   has no sweep.

The camera is a function of the frame index only (keyframes, interpolated in a straight line at
constant speed), never of the clock, so every run on every machine shows the GPU the same views.
The path has a version in its name (`planet-flyover-v1`); if the path ever changes, the version
changes, and reports of different versions are not comparable.

It draws **offscreen** (`render::Offscreen`), not in a window. The tracer and the pass timers are
the same as the window's; what is left out is the window's `blit` and `overlay` passes, and
what is gained is that nothing waits for the display and that a frame's GPU times are in when it
returns, so there is no lag to correct for. It asks Vulkan for the high-performance adapter, as
the window does, and the report says which GPU it got: check `machine.device_type` is not `Cpu`
before reading anything into the times.

## Reading the report

```
{
  "schema_version": 3,
  "build":   { "version", "commit" },          // commit: CI builds only, else null; on a PR
                                               // build it is GitHub's merge commit, not the head
  "machine": { "gpu", "device_type", "driver", "driver_info", "backend", "os", "arch",
               "timers_supported" },
  "config":  { "width", "height", "output_width", "output_height", "scene", "path",
               "keyframes", "bias" },          // width, height: traced; output_*: upscaled
  "modes": {
    "primary":   { "frames", "warmup_frames", "spp", "passes": [ENTRY...], "gpu_total": ENTRY,
                   "frame_time": STATS, "scene_update_ms": STATS,
                   "lod_at_keyframes": [ { "frame", "height_m", "bricks": [n...] }... ] },
    "reference": { ... same ... }
  },
  "altitudes": [ { "height_m", "bricks": [n...], "primary": STATS, "upscale": STATS,
                   "reference": STATS }... ],
  "screenshots": [ "primary-frame0000.png", ... ],
  "budgets": { "source", "ok", "failed": ["primary/primary"], "unmatched": [] }
}
```

- `STATS` is `{ "samples", "mean_ms", "p95_ms", "max_ms" }`. The figures are `null` when there are
  no samples. `p95_ms` is the nearest rank. Times come from every sample the GPU timers recorded
  over the measured frames (`Timings::history()`), not from the overlay's rolling window.
- `ENTRY` is `STATS` plus `name`, `budget_ms` (null if the pass has none) and `status`:
  - `within`: the **mean** is at most the budget.
  - `over`: the mean is above it. This blocks the merge.
  - `no-budget`: nothing declares a budget for it.
  - `no-data`: it has a budget but no times, because the GPU has no timestamp queries
    (`machine.timers_supported` is `false`). It fails the run, since nothing was checked.
- `gpu_total` is the GPU time of one profiler frame, from the start of its first pass to the end
  of its last. In primary mode that is one frame of the path. In reference mode every sample is a
  profiler frame of its own, so `passes[]` and `gpu_total` are **per sample** there, with
  `frames * spp` samples, and `gpu_total.reference` budgets one sample.
- `frame_time` is CPU time to draw one frame of the path, and to read the image back where it is
  read (reference mode, and primary mode's screenshot frames). For reference mode a frame is all
  its samples. It is context, not a budget.
- `scene_update_ms` is the CPU time per frame that the Planet takes to follow the camera: the
  bands select the bricks again, those that changed level are packed, and the changes are written
  to the GPU, flushed at once. It is done before the frame is timed, so it is in no pass and not
  in `frame_time`, but a window pays it. Its mean is CPU time, not GPU time, and a CPU that is
  slow to follow leaves the GPU idle between frames (see below).
- `lod_at_keyframes` has, for each keyframe of the path, the frame, how high the camera is above
  the Planet's mean radius in metres, and how many bricks are drawn at each level of detail
  (level 0 first). Empty for `--short`.
- `altitudes` is the sweep: for each height, the bricks drawn at each level and the `primary` and
  `upscale` passes of primary mode and the `reference` pass of reference mode (one sample), with
  no budget held against them. It is how the cost of the LOD bands is read at the ground, 100 m,
  500 m and 2 km.
- `budgets.unmatched` lists budgets for passes that did not run (`passes.shadows` before there is
  a shadows pass): they are not an error.

The mean is what is held to the budget; look at `p95_ms` and `max_ms` for stutter.

### Reading a regression

A pass whose work did not change but whose time did (the `upscale` pass does not read the Planet)
points at how the frames are paced, not at the pass. The bench draws one frame at a time and
waits for it, so the GPU idles for as long as the CPU takes to follow the camera
(`scene_update_ms`) before the next frame, and a GPU that has idled may time a pass that is
limited by memory (`upscale` is) differently from one that has not. To tell the Planet's GPU work
from that gap, run the bench with `--diag` and compare `upscale` and `reference` against a plain
run (the camera path is the same):

| Run | If `upscale` falls back to main's | It is |
| --- | --- | --- |
| `--diag no-follow` | yes | something in following the camera: the next three say which |
| `--diag no-follow,gap-ms=N` (N about `scene_update_ms`) | no, it stays high | the idle gap: nothing the Planet sends to the GPU |
| `--diag no-upload` | no, it stays high | the CPU's time, not the writes |
| `--diag no-flush` | yes | the writes being sent at once, ahead of the frame |

## Budgets

`bins/client/bench-budgets.toml` is compiled into the executable, so the Windows artifact needs no
file beside it. To try other numbers, copy it and pass `--budgets copy.toml`. The report says
which budgets it used (`budgets.source`).

```toml
[passes]            # milliseconds of GPU time per frame, by the name of the pass
primary = 1.0

[gpu_total]         # GPU total of one profiler frame, by mode: primary, or one reference sample
primary = 8.6
```

Anything else in the file, a pass or mode with a zero, negative or non-numeric budget, is an
error: a typo must not silently turn a budget off.

The numbers are the per-stage frame budget of the rendering decision (#10), on the RTX 4090 at 1440p
output. Only `primary` (Primary visibility, 1.0 ms), `upscale` (the TAAU pass, 0.5 ms) and the total
(8.6 ms) are live today, as those are the passes that exist. The later stages (sun shadows 0.4, GI
1.5, reflections 0.7, glass 0.7, volumetrics 1.0, denoise 1.0, atmosphere 0.3, edits 1.0, post 0.5)
are in the file as comments: **whoever adds a pass uncomments its budget and names it the same** (a
pass is named by the string given to `Profiler::compute_pass`). `reference` has no budget in #10;
its 3.0 ms (a sample, pass and total) catches a gross regression, set from the first real run on the
4090, which measured a mean of 1.10 ms.

## Tests

The statistics, budget evaluation and exit status, camera path, argument parsing and report shape
are unit tests in `bins/client/src/bench/`. `bins/client/tests/bench.rs` runs the real
`client bench --short` and checks the report's schema, the screenshots and the exit codes: it
runs in `cargo test` (and so in CI) on lavapipe. `--short` uses a patch of the Planet, the one the
golden image test renders, because the whole Planet takes about 22 s to generate in release.
