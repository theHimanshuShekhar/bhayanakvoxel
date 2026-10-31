//! Mean, 95th percentile and maximum of the times that the GPU timers recorded.

use serde::Serialize;

/// Summary of a series of times in milliseconds. With no samples, every figure is `null`.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct Stats {
    pub samples: usize,
    pub mean_ms: Option<f64>,
    pub p95_ms: Option<f64>,
    pub max_ms: Option<f64>,
}

impl Stats {
    /// The p95 is the nearest rank: the smallest time that at least 95% of the samples do not
    /// exceed.
    pub fn of(times: &[f64]) -> Self {
        if times.is_empty() {
            return Self {
                samples: 0,
                mean_ms: None,
                p95_ms: None,
                max_ms: None,
            };
        }
        let mut sorted = times.to_vec();
        sorted.sort_by(f64::total_cmp);
        let rank = (sorted.len() * 95).div_ceil(100);
        Self {
            samples: sorted.len(),
            mean_ms: Some(sorted.iter().sum::<f64>() / sorted.len() as f64),
            p95_ms: Some(sorted[rank - 1]),
            max_ms: sorted.last().copied(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_samples_have_no_figures() {
        let stats = Stats::of(&[]);
        assert_eq!(
            stats,
            Stats {
                samples: 0,
                mean_ms: None,
                p95_ms: None,
                max_ms: None
            }
        );
    }

    #[test]
    fn a_single_sample_is_its_own_mean_p95_and_max() {
        let stats = Stats::of(&[2.5]);
        assert_eq!(stats.samples, 1);
        assert_eq!(
            (stats.mean_ms, stats.p95_ms, stats.max_ms),
            (Some(2.5), Some(2.5), Some(2.5))
        );
    }

    #[test]
    fn mean_p95_and_max_of_a_known_series() {
        // 1..=100 ms in reverse order, which must not matter. The nearest-rank p95 of 100
        // samples is the 95th.
        let times: Vec<f64> = (1..=100).rev().map(f64::from).collect();
        let stats = Stats::of(&times);
        assert_eq!(stats.samples, 100);
        assert_eq!(stats.mean_ms, Some(50.5));
        assert_eq!(stats.p95_ms, Some(95.0));
        assert_eq!(stats.max_ms, Some(100.0));
    }

    #[test]
    fn p95_rounds_the_rank_up_so_a_slow_tail_shows() {
        // 10 samples: ceil(0.95 * 10) is the 10th, the slowest.
        let mut times = vec![1.0; 9];
        times.push(9.0);
        assert_eq!(Stats::of(&times).p95_ms, Some(9.0));
        // 20 samples: the 19th, so one outlier in 20 stays out of p95 but is in the max.
        let mut times = vec![1.0; 19];
        times.push(50.0);
        let stats = Stats::of(&times);
        assert_eq!((stats.p95_ms, stats.max_ms), (Some(1.0), Some(50.0)));
    }
}
