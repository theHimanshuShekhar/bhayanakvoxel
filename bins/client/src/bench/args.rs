//! The command line of `client bench`.

use std::path::PathBuf;

pub const USAGE: &str = "\
Usage: client bench [OPTIONS]

Flies a fixed camera path over the Planet in primary and then reference mode, offscreen on the
GPU, and writes report.json and screenshots to the output folder. Exits with status 2 if a pass
is over its budget.

Options:
  --out DIR        Output folder, created if need be [default: bench-out]
  --budgets FILE   Budgets (TOML) to use instead of the ones built into the executable
  --size WxH       Resolution of the traced image [default: 1280x720]; the upscaled image
                   is twice as big in each direction
  --short          A tiny run on a patch of the Planet at low resolution: for a smoke test,
                   not for timings
  --bias F         LOD bias: output pixels one Voxel may cover [default: 1]. Every switch
                   distance between levels of detail scales with it; 2 does a quarter of the
                   work for up to a pixel more of error
  --diag LIST      Switches for finding out where time goes, separated by commas; a report made
                   with any of them is not a measurement. no-follow: the Planet does not follow
                   the camera. no-upload: it does, on the CPU, but the GPU is not told.
                   no-flush: writes to the GPU go with the next frame's commands. gap-ms=F:
                   sleep F milliseconds before each frame, as the CPU work of following would
  --no-fail        Exit with status 0 even if a pass is over its budget
  -h, --help       Print this";

/// What `--diag` switches on. Everything off is a measurement.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Diag {
    pub no_follow: bool,
    pub no_upload: bool,
    pub no_flush: bool,
    pub gap_ms: f64,
}

#[derive(Debug, PartialEq)]
pub struct Args {
    pub out: PathBuf,
    pub budgets: Option<PathBuf>,
    pub short: bool,
    pub no_fail: bool,
    pub size: Option<(u32, u32)>,
    pub bias: f64,
    pub diag: Diag,
}

#[derive(Debug, PartialEq)]
pub enum Parsed {
    Help,
    Run(Args),
}

/// Parses what follows `bench` on the command line.
pub fn parse(args: impl IntoIterator<Item = String>) -> Result<Parsed, String> {
    let mut parsed = Args {
        out: "bench-out".into(),
        budgets: None,
        short: false,
        no_fail: false,
        size: None,
        bias: 1.0,
        diag: Diag::default(),
    };
    let mut args = args.into_iter();
    while let Some(arg) = args.next() {
        let mut value = |option: &str| args.next().ok_or(format!("{option} needs a value"));
        match arg.as_str() {
            "-h" | "--help" => return Ok(Parsed::Help),
            "--short" => parsed.short = true,
            "--no-fail" => parsed.no_fail = true,
            "--out" => parsed.out = value("--out")?.into(),
            "--budgets" => parsed.budgets = Some(value("--budgets")?.into()),
            "--size" => parsed.size = Some(size(&value("--size")?)?),
            "--bias" => parsed.bias = bias(&value("--bias")?)?,
            "--diag" => parsed.diag = diag(&value("--diag")?)?,
            _ => return Err(format!("unexpected argument `{arg}`")),
        }
    }
    Ok(Parsed::Run(parsed))
}

/// `--diag`'s comma-separated switches.
fn diag(text: &str) -> Result<Diag, String> {
    let mut diag = Diag::default();
    for item in text.split(',') {
        match item.split_once('=') {
            None if item == "no-follow" => diag.no_follow = true,
            None if item == "no-upload" => diag.no_upload = true,
            None if item == "no-flush" => diag.no_flush = true,
            Some(("gap-ms", ms)) => {
                diag.gap_ms = ms
                    .parse::<f64>()
                    .ok()
                    .filter(|m| m.is_finite() && *m >= 0.0)
                    .ok_or_else(|| format!("`{ms}` is not a number of milliseconds"))?;
            }
            _ => return Err(format!("`{item}` is not a --diag switch")),
        }
    }
    Ok(diag)
}

/// A LOD bias: a positive, finite number.
fn bias(text: &str) -> Result<f64, String> {
    text.parse::<f64>()
        .ok()
        .filter(|b| b.is_finite() && *b > 0.0)
        .ok_or_else(|| format!("`{text}` is not a LOD bias like 1 or 2.5"))
}

/// `WIDTHxHEIGHT`, both at least 1.
fn size(text: &str) -> Result<(u32, u32), String> {
    let wrong = || format!("`{text}` is not a size like 1280x720");
    let (width, height) = text.split_once('x').ok_or_else(wrong)?;
    let dimension = |d: &str| d.parse::<u32>().ok().filter(|&d| d > 0);
    dimension(width).zip(dimension(height)).ok_or_else(wrong)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_all(args: &[&str]) -> Result<Parsed, String> {
        parse(args.iter().map(|a| a.to_string()))
    }

    fn run(args: &[&str]) -> Args {
        match parse_all(args) {
            Ok(Parsed::Run(args)) => args,
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn with_no_options_the_defaults_apply() {
        assert_eq!(
            run(&[]),
            Args {
                out: "bench-out".into(),
                budgets: None,
                short: false,
                no_fail: false,
                size: None,
                bias: 1.0,
                diag: Diag::default()
            }
        );
    }

    #[test]
    fn every_option_is_read() {
        let args = run(&[
            "--short",
            "--out",
            "o",
            "--no-fail",
            "--budgets",
            "b.toml",
            "--size",
            "640x360",
            "--bias",
            "2",
        ]);
        assert_eq!(args.out, PathBuf::from("o"));
        assert_eq!(args.budgets, Some(PathBuf::from("b.toml")));
        assert!(args.short && args.no_fail);
        assert_eq!(args.size, Some((640, 360)));
        assert_eq!(args.bias, 2.0);
    }

    #[test]
    fn diag_switches_are_read_and_a_wrong_one_is_an_error() {
        let args = run(&["--diag", "no-follow,gap-ms=2.5,no-flush"]);
        assert_eq!(
            args.diag,
            Diag {
                no_follow: true,
                no_flush: true,
                gap_ms: 2.5,
                ..Diag::default()
            }
        );
        assert!(run(&["--diag", "no-upload"]).diag.no_upload);
        for bad in ["fast", "gap-ms=x", "gap-ms=-1", "no-follow,"] {
            assert!(parse_all(&["--diag", bad]).is_err(), "{bad}");
        }
    }

    #[test]
    fn help_is_not_a_run() {
        assert_eq!(parse_all(&["--no-fail", "--help"]), Ok(Parsed::Help));
        assert_eq!(parse_all(&["-h"]), Ok(Parsed::Help));
    }

    #[test]
    fn mistakes_are_errors_that_say_what_was_wrong() {
        for bad in [
            &["--nope"][..],
            &["--out"],
            &["--budgets"],
            &["--size", "640"],
            &["--size", "0x360"],
            &["--size", "axb"],
            &["--bias"],
            &["--bias", "0"],
            &["--bias", "-1"],
            &["--bias", "fast"],
            &["--bias", "inf"],
            &["stray"],
        ] {
            assert!(parse_all(bad).is_err(), "{bad:?}");
        }
        assert!(parse_all(&["--nope"]).unwrap_err().contains("--nope"));
    }
}
