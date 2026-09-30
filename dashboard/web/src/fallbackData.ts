// Pre-computed held-out evaluation benchmarks across all 7 scenarios & 6 schedulers
export interface BenchRow {
  scenario: string;
  scheduler: string;
  n: number;
  [k: string]: string | number | null | undefined;
}

export const FALLBACK_BENCHMARK: BenchRow[] = [
  {
    "scenario": "S1_sparse",
    "scheduler": "sweep",
    "n": 5,
    "pd": 0.23948,
    "pd_std": 0.05242,
    "pd_weighted": 0.20779,
    "pd_weighted_std": 0.04559,
    "intercept_rate": 0.115,
    "intercept_rate_std": 0.01369,
    "ttfi_censored_mean": 7.92138,
    "ttfi_censored_mean_std": 2.2801,
    "intercept_time_error_ms": null,
    "intercept_time_error_ms_std": null,
    "pfa": null,
    "pfa_std": null,
    "correct_predictions": null,
    "correct_predictions_std": null
  },
  {
    "scenario": "S1_sparse",
    "scheduler": "random",
    "n": 5,
    "pd": 0.19675,
    "pd_std": 0.05107,
    "pd_weighted": 0.19961,
    "pd_weighted_std": 0.04517,
    "intercept_rate": 0.1,
    "intercept_rate_std": 0.01768,
    "ttfi_censored_mean": 15.02358,
    "ttfi_censored_mean_std": 8.01735,
    "intercept_time_error_ms": null,
    "intercept_time_error_ms_std": null,
    "pfa": null,
    "pfa_std": null,
    "correct_predictions": null,
    "correct_predictions_std": null
  },
  {
    "scenario": "S1_sparse",
    "scheduler": "round_robin",
    "n": 5,
    "pd": 0.2373,
    "pd_std": 0.0303,
    "pd_weighted": 0.21272,
    "pd_weighted_std": 0.01579,
    "intercept_rate": 0.105,
    "intercept_rate_std": 0.01118,
    "ttfi_censored_mean": 12.60091,
    "ttfi_censored_mean_std": 4.76747,
    "intercept_time_error_ms": null,
    "intercept_time_error_ms_std": null,
    "pfa": null,
    "pfa_std": null,
    "correct_predictions": null,
    "correct_predictions_std": null
  },
  {
    "scenario": "S1_sparse",
    "scheduler": "bandit",
    "n": 5,
    "pd": 0.22255,
    "pd_std": 0.0337,
    "pd_weighted": 0.20416,
    "pd_weighted_std": 0.04932,
    "intercept_rate": 0.11,
    "intercept_rate_std": 0.01369,
    "ttfi_censored_mean": 10.72225,
    "ttfi_censored_mean_std": 3.52332,
    "intercept_time_error_ms": null,
    "intercept_time_error_ms_std": null,
    "pfa": null,
    "pfa_std": null,
    "correct_predictions": null,
    "correct_predictions_std": null
  },
  {
    "scenario": "S1_sparse",
    "scheduler": "smart",
    "n": 5,
    "pd": 0.67284,
    "pd_std": 0.1868,
    "pd_weighted": 0.67838,
    "pd_weighted_std": 0.13255,
    "intercept_rate": 0.125,
    "intercept_rate_std": 0.0,
    "ttfi_censored_mean": 9.36575,
    "ttfi_censored_mean_std": 5.51274,
    "intercept_time_error_ms": 7.90201,
    "intercept_time_error_ms_std": 1.61644,
    "pfa": 0.08692,
    "pfa_std": 0.107,
    "correct_predictions": 0.91308,
    "correct_predictions_std": 0.107
  },
  {
    "scenario": "S1_sparse",
    "scheduler": "d3qn",
    "n": 5,
    "pd": 0.75734,
    "pd_std": 0.10611,
    "pd_weighted": 0.7824,
    "pd_weighted_std": 0.1121,
    "intercept_rate": 0.115,
    "intercept_rate_std": 0.01369,
    "ttfi_censored_mean": 8.73409,
    "ttfi_censored_mean_std": 4.47604,
    "intercept_time_error_ms": 8.07213,
    "intercept_time_error_ms_std": 2.89646,
    "pfa": 0.1003,
    "pfa_std": 0.11816,
    "correct_predictions": 0.8997,
    "correct_predictions_std": 0.11816
  },
  {
    "scenario": "S2_dense",
    "scheduler": "sweep",
    "n": 5,
    "pd": 0.13077,
    "pd_std": 0.00883,
    "pd_weighted": 0.10802,
    "pd_weighted_std": 0.01178,
    "intercept_rate": 0.85,
    "intercept_rate_std": 0.04677,
    "ttfi_censored_mean": 8.23734,
    "ttfi_censored_mean_std": 1.48939,
    "intercept_time_error_ms": null,
    "intercept_time_error_ms_std": null,
    "pfa": null,
    "pfa_std": null,
    "correct_predictions": null,
    "correct_predictions_std": null
  },
  {
    "scenario": "S2_dense",
    "scheduler": "random",
    "n": 5,
    "pd": 0.12859,
    "pd_std": 0.0126,
    "pd_weighted": 0.11601,
    "pd_weighted_std": 0.01579,
    "intercept_rate": 0.805,
    "intercept_rate_std": 0.04809,
    "ttfi_censored_mean": 11.78906,
    "ttfi_censored_mean_std": 2.88732,
    "intercept_time_error_ms": null,
    "intercept_time_error_ms_std": null,
    "pfa": null,
    "pfa_std": null,
    "correct_predictions": null,
    "correct_predictions_std": null
  },
  {
    "scenario": "S2_dense",
    "scheduler": "round_robin",
    "n": 5,
    "pd": 0.16413,
    "pd_std": 0.01332,
    "pd_weighted": 0.14181,
    "pd_weighted_std": 0.00944,
    "intercept_rate": 0.82,
    "intercept_rate_std": 0.01118,
    "ttfi_censored_mean": 9.18851,
    "ttfi_censored_mean_std": 1.30113,
    "intercept_time_error_ms": null,
    "intercept_time_error_ms_std": null,
    "pfa": null,
    "pfa_std": null,
    "correct_predictions": null,
    "correct_predictions_std": null
  },
  {
    "scenario": "S2_dense",
    "scheduler": "bandit",
    "n": 5,
    "pd": 0.14093,
    "pd_std": 0.01769,
    "pd_weighted": 0.1205,
    "pd_weighted_std": 0.01557,
    "intercept_rate": 0.825,
    "intercept_rate_std": 0.01768,
    "ttfi_censored_mean": 10.87792,
    "ttfi_censored_mean_std": 1.785,
    "intercept_time_error_ms": null,
    "intercept_time_error_ms_std": null,
    "pfa": null,
    "pfa_std": null,
    "correct_predictions": null,
    "correct_predictions_std": null
  },
  {
    "scenario": "S2_dense",
    "scheduler": "smart",
    "n": 5,
    "pd": 0.48863,
    "pd_std": 0.06956,
    "pd_weighted": 0.47717,
    "pd_weighted_std": 0.07817,
    "intercept_rate": 0.8,
    "intercept_rate_std": 0.03953,
    "ttfi_censored_mean": 7.31042,
    "ttfi_censored_mean_std": 2.87452,
    "intercept_time_error_ms": 6.07878,
    "intercept_time_error_ms_std": 1.10714,
    "pfa": 0.37337,
    "pfa_std": 0.02864,
    "correct_predictions": 0.62663,
    "correct_predictions_std": 0.02864
  },
  {
    "scenario": "S2_dense",
    "scheduler": "d3qn",
    "n": 5,
    "pd": 0.50451,
    "pd_std": 0.09014,
    "pd_weighted": 0.5041,
    "pd_weighted_std": 0.0839,
    "intercept_rate": 0.72,
    "intercept_rate_std": 0.11646,
    "ttfi_censored_mean": 9.66357,
    "ttfi_censored_mean_std": 5.16837,
    "intercept_time_error_ms": 6.54615,
    "intercept_time_error_ms_std": 1.13778,
    "pfa": 0.40604,
    "pfa_std": 0.07833,
    "correct_predictions": 0.59396,
    "correct_predictions_std": 0.07833
  },
  {
    "scenario": "S3_mfr",
    "scheduler": "sweep",
    "n": 5,
    "pd": 0.07922,
    "pd_std": 0.01242,
    "pd_weighted": 0.07425,
    "pd_weighted_std": 0.01345,
    "intercept_rate": 0.24,
    "intercept_rate_std": 0.01369,
    "ttfi_censored_mean": 9.0142,
    "ttfi_censored_mean_std": 2.14911,
    "intercept_time_error_ms": null,
    "intercept_time_error_ms_std": null,
    "pfa": null,
    "pfa_std": null,
    "correct_predictions": null,
    "correct_predictions_std": null
  },
  {
    "scenario": "S3_mfr",
    "scheduler": "random",
    "n": 5,
    "pd": 0.09171,
    "pd_std": 0.01891,
    "pd_weighted": 0.08873,
    "pd_weighted_std": 0.01863,
    "intercept_rate": 0.23,
    "intercept_rate_std": 0.01118,
    "ttfi_censored_mean": 9.96802,
    "ttfi_censored_mean_std": 3.51608,
    "intercept_time_error_ms": null,
    "intercept_time_error_ms_std": null,
    "pfa": null,
    "pfa_std": null,
    "correct_predictions": null,
    "correct_predictions_std": null
  },
  {
    "scenario": "S3_mfr",
    "scheduler": "round_robin",
    "n": 5,
    "pd": 0.10123,
    "pd_std": 0.00284,
    "pd_weighted": 0.09701,
    "pd_weighted_std": 0.00281,
    "intercept_rate": 0.23,
    "intercept_rate_std": 0.02092,
    "ttfi_censored_mean": 10.11377,
    "ttfi_censored_mean_std": 1.67512,
    "intercept_time_error_ms": null,
    "intercept_time_error_ms_std": null,
    "pfa": null,
    "pfa_std": null,
    "correct_predictions": null,
    "correct_predictions_std": null
  },
  {
    "scenario": "S3_mfr",
    "scheduler": "bandit",
    "n": 5,
    "pd": 0.09564,
    "pd_std": 0.00917,
    "pd_weighted": 0.09055,
    "pd_weighted_std": 0.00833,
    "intercept_rate": 0.25,
    "intercept_rate_std": 0.0,
    "ttfi_censored_mean": 8.83222,
    "ttfi_censored_mean_std": 3.59913,
    "intercept_time_error_ms": null,
    "intercept_time_error_ms_std": null,
    "pfa": null,
    "pfa_std": null,
    "correct_predictions": null,
    "correct_predictions_std": null
  },
  {
    "scenario": "S3_mfr",
    "scheduler": "smart",
    "n": 5,
    "pd": 0.58569,
    "pd_std": 0.12286,
    "pd_weighted": 0.60635,
    "pd_weighted_std": 0.12662,
    "intercept_rate": 0.25,
    "intercept_rate_std": 0.0,
    "ttfi_censored_mean": 4.97043,
    "ttfi_censored_mean_std": 2.05371,
    "intercept_time_error_ms": 4.56899,
    "intercept_time_error_ms_std": 0.59728,
    "pfa": 0.52779,
    "pfa_std": 0.02133,
    "correct_predictions": 0.47221,
    "correct_predictions_std": 0.02133
  },
  {
    "scenario": "S3_mfr",
    "scheduler": "d3qn",
    "n": 5,
    "pd": 0.62312,
    "pd_std": 0.0976,
    "pd_weighted": 0.64895,
    "pd_weighted_std": 0.09926,
    "intercept_rate": 0.235,
    "intercept_rate_std": 0.02236,
    "ttfi_censored_mean": 5.85325,
    "ttfi_censored_mean_std": 3.18109,
    "intercept_time_error_ms": 4.68714,
    "intercept_time_error_ms_std": 0.37595,
    "pfa": 0.60146,
    "pfa_std": 0.05831,
    "correct_predictions": 0.39854,
    "correct_predictions_std": 0.05831
  },
  {
    "scenario": "S4_agile_lpi",
    "scheduler": "sweep",
    "n": 5,
    "pd": 0.22379,
    "pd_std": 0.04235,
    "pd_weighted": 0.22668,
    "pd_weighted_std": 0.04446,
    "intercept_rate": 0.225,
    "intercept_rate_std": 0.025,
    "ttfi_censored_mean": 11.52385,
    "ttfi_censored_mean_std": 4.06095,
    "intercept_time_error_ms": null,
    "intercept_time_error_ms_std": null,
    "pfa": null,
    "pfa_std": null,
    "correct_predictions": null,
    "correct_predictions_std": null
  },
  {
    "scenario": "S4_agile_lpi",
    "scheduler": "random",
    "n": 5,
    "pd": 0.23074,
    "pd_std": 0.0461,
    "pd_weighted": 0.24327,
    "pd_weighted_std": 0.05473,
    "intercept_rate": 0.22,
    "intercept_rate_std": 0.02092,
    "ttfi_censored_mean": 12.37261,
    "ttfi_censored_mean_std": 4.52362,
    "intercept_time_error_ms": null,
    "intercept_time_error_ms_std": null,
    "pfa": null,
    "pfa_std": null,
    "correct_predictions": null,
    "correct_predictions_std": null
  },
  {
    "scenario": "S4_agile_lpi",
    "scheduler": "round_robin",
    "n": 5,
    "pd": 0.25803,
    "pd_std": 0.08352,
    "pd_weighted": 0.26705,
    "pd_weighted_std": 0.08626,
    "intercept_rate": 0.2,
    "intercept_rate_std": 0.03953,
    "ttfi_censored_mean": 11.36493,
    "ttfi_censored_mean_std": 4.35765,
    "intercept_time_error_ms": null,
    "intercept_time_error_ms_std": null,
    "pfa": null,
    "pfa_std": null,
    "correct_predictions": null,
    "correct_predictions_std": null
  },
  {
    "scenario": "S4_agile_lpi",
    "scheduler": "bandit",
    "n": 5,
    "pd": 0.25513,
    "pd_std": 0.05452,
    "pd_weighted": 0.25593,
    "pd_weighted_std": 0.05753,
    "intercept_rate": 0.215,
    "intercept_rate_std": 0.04183,
    "ttfi_censored_mean": 10.0163,
    "ttfi_censored_mean_std": 5.20743,
    "intercept_time_error_ms": null,
    "intercept_time_error_ms_std": null,
    "pfa": null,
    "pfa_std": null,
    "correct_predictions": null,
    "correct_predictions_std": null
  },
  {
    "scenario": "S4_agile_lpi",
    "scheduler": "smart",
    "n": 5,
    "pd": 0.83677,
    "pd_std": 0.12821,
    "pd_weighted": 0.82568,
    "pd_weighted_std": 0.15513,
    "intercept_rate": 0.215,
    "intercept_rate_std": 0.0285,
    "ttfi_censored_mean": 5.45343,
    "ttfi_censored_mean_std": 4.10658,
    "intercept_time_error_ms": 7.78534,
    "intercept_time_error_ms_std": 1.42914,
    "pfa": 0.08562,
    "pfa_std": 0.07317,
    "correct_predictions": 0.91438,
    "correct_predictions_std": 0.07317
  },
  {
    "scenario": "S4_agile_lpi",
    "scheduler": "d3qn",
    "n": 5,
    "pd": 0.81362,
    "pd_std": 0.13278,
    "pd_weighted": 0.82461,
    "pd_weighted_std": 0.11615,
    "intercept_rate": 0.225,
    "intercept_rate_std": 0.01768,
    "ttfi_censored_mean": 6.52062,
    "ttfi_censored_mean_std": 4.46399,
    "intercept_time_error_ms": 7.15406,
    "intercept_time_error_ms_std": 0.84087,
    "pfa": 0.08378,
    "pfa_std": 0.0623,
    "correct_predictions": 0.91622,
    "correct_predictions_std": 0.0623
  },
  {
    "scenario": "S5_popup",
    "scheduler": "sweep",
    "n": 5,
    "pd": 0.09885,
    "pd_std": 0.00872,
    "pd_weighted": 0.07871,
    "pd_weighted_std": 0.00797,
    "intercept_rate": 0.31,
    "intercept_rate_std": 0.01369,
    "ttfi_censored_mean": 8.59471,
    "ttfi_censored_mean_std": 3.69088,
    "intercept_time_error_ms": null,
    "intercept_time_error_ms_std": null,
    "pfa": null,
    "pfa_std": null,
    "correct_predictions": null,
    "correct_predictions_std": null
  },
  {
    "scenario": "S5_popup",
    "scheduler": "random",
    "n": 5,
    "pd": 0.09082,
    "pd_std": 0.01303,
    "pd_weighted": 0.08079,
    "pd_weighted_std": 0.01175,
    "intercept_rate": 0.305,
    "intercept_rate_std": 0.02739,
    "ttfi_censored_mean": 9.82534,
    "ttfi_censored_mean_std": 4.17939,
    "intercept_time_error_ms": null,
    "intercept_time_error_ms_std": null,
    "pfa": null,
    "pfa_std": null,
    "correct_predictions": null,
    "correct_predictions_std": null
  },
  {
    "scenario": "S5_popup",
    "scheduler": "round_robin",
    "n": 5,
    "pd": 0.11941,
    "pd_std": 0.02581,
    "pd_weighted": 0.10088,
    "pd_weighted_std": 0.02527,
    "intercept_rate": 0.305,
    "intercept_rate_std": 0.01118,
    "ttfi_censored_mean": 5.89554,
    "ttfi_censored_mean_std": 2.7852,
    "intercept_time_error_ms": null,
    "intercept_time_error_ms_std": null,
    "pfa": null,
    "pfa_std": null,
    "correct_predictions": null,
    "correct_predictions_std": null
  },
  {
    "scenario": "S5_popup",
    "scheduler": "bandit",
    "n": 5,
    "pd": 0.09996,
    "pd_std": 0.0146,
    "pd_weighted": 0.08582,
    "pd_weighted_std": 0.00983,
    "intercept_rate": 0.3,
    "intercept_rate_std": 0.01768,
    "ttfi_censored_mean": 8.90816,
    "ttfi_censored_mean_std": 2.7758,
    "intercept_time_error_ms": null,
    "intercept_time_error_ms_std": null,
    "pfa": null,
    "pfa_std": null,
    "correct_predictions": null,
    "correct_predictions_std": null
  },
  {
    "scenario": "S5_popup",
    "scheduler": "smart",
    "n": 5,
    "pd": 0.55172,
    "pd_std": 0.07493,
    "pd_weighted": 0.55021,
    "pd_weighted_std": 0.08629,
    "intercept_rate": 0.31,
    "intercept_rate_std": 0.01369,
    "ttfi_censored_mean": 5.14343,
    "ttfi_censored_mean_std": 1.33529,
    "intercept_time_error_ms": 5.55061,
    "intercept_time_error_ms_std": 1.22223,
    "pfa": 0.38787,
    "pfa_std": 0.0505,
    "correct_predictions": 0.61213,
    "correct_predictions_std": 0.0505
  },
  {
    "scenario": "S5_popup",
    "scheduler": "d3qn",
    "n": 5,
    "pd": 0.52829,
    "pd_std": 0.08476,
    "pd_weighted": 0.52386,
    "pd_weighted_std": 0.10493,
    "intercept_rate": 0.305,
    "intercept_rate_std": 0.02092,
    "ttfi_censored_mean": 5.78023,
    "ttfi_censored_mean_std": 2.39528,
    "intercept_time_error_ms": 4.80365,
    "intercept_time_error_ms_std": 1.43815,
    "pfa": 0.44704,
    "pfa_std": 0.03232,
    "correct_predictions": 0.55296,
    "correct_predictions_std": 0.03232
  },
  {
    "scenario": "S6_lockin",
    "scheduler": "sweep",
    "n": 5,
    "pd": 0.015,
    "pd_std": 0.03354,
    "pd_weighted": 0.01053,
    "pd_weighted_std": 0.02354,
    "intercept_rate": 0.005,
    "intercept_rate_std": 0.01118,
    "ttfi_censored_mean": 37.12159,
    "ttfi_censored_mean_std": 3.09181,
    "intercept_time_error_ms": null,
    "intercept_time_error_ms_std": null,
    "pfa": null,
    "pfa_std": null,
    "correct_predictions": null,
    "correct_predictions_std": null
  },
  {
    "scenario": "S6_lockin",
    "scheduler": "random",
    "n": 5,
    "pd": 0.13168,
    "pd_std": 0.01501,
    "pd_weighted": 0.13632,
    "pd_weighted_std": 0.0129,
    "intercept_rate": 0.13,
    "intercept_rate_std": 0.02092,
    "ttfi_censored_mean": 15.56189,
    "ttfi_censored_mean_std": 6.76813,
    "intercept_time_error_ms": null,
    "intercept_time_error_ms_std": null,
    "pfa": null,
    "pfa_std": null,
    "correct_predictions": null,
    "correct_predictions_std": null
  },
  {
    "scenario": "S6_lockin",
    "scheduler": "round_robin",
    "n": 5,
    "pd": 0.18277,
    "pd_std": 0.01321,
    "pd_weighted": 0.18559,
    "pd_weighted_std": 0.02059,
    "intercept_rate": 0.15,
    "intercept_rate_std": 0.0,
    "ttfi_censored_mean": 8.99225,
    "ttfi_censored_mean_std": 3.51356,
    "intercept_time_error_ms": null,
    "intercept_time_error_ms_std": null,
    "pfa": null,
    "pfa_std": null,
    "correct_predictions": null,
    "correct_predictions_std": null
  },
  {
    "scenario": "S6_lockin",
    "scheduler": "bandit",
    "n": 5,
    "pd": 0.06222,
    "pd_std": 0.11891,
    "pd_weighted": 0.05178,
    "pd_weighted_std": 0.09116,
    "intercept_rate": 0.035,
    "intercept_rate_std": 0.03791,
    "ttfi_censored_mean": 29.5236,
    "ttfi_censored_mean_std": 9.40249,
    "intercept_time_error_ms": null,
    "intercept_time_error_ms_std": null,
    "pfa": null,
    "pfa_std": null,
    "correct_predictions": null,
    "correct_predictions_std": null
  },
  {
    "scenario": "S6_lockin",
    "scheduler": "smart",
    "n": 5,
    "pd": 0.88104,
    "pd_std": 0.06437,
    "pd_weighted": 0.87801,
    "pd_weighted_std": 0.0861,
    "intercept_rate": 0.15,
    "intercept_rate_std": 0.0,
    "ttfi_censored_mean": 4.39952,
    "ttfi_censored_mean_std": 2.9613,
    "intercept_time_error_ms": 10.34364,
    "intercept_time_error_ms_std": 0.65531,
    "pfa": 0.03909,
    "pfa_std": 0.03465,
    "correct_predictions": 0.96091,
    "correct_predictions_std": 0.03465
  },
  {
    "scenario": "S6_lockin",
    "scheduler": "d3qn",
    "n": 5,
    "pd": 0.92278,
    "pd_std": 0.05576,
    "pd_weighted": 0.92453,
    "pd_weighted_std": 0.0561,
    "intercept_rate": 0.15,
    "intercept_rate_std": 0.0,
    "ttfi_censored_mean": 1.7978,
    "ttfi_censored_mean_std": 1.58886,
    "intercept_time_error_ms": 8.85907,
    "intercept_time_error_ms_std": 1.56826,
    "pfa": 0.03145,
    "pfa_std": 0.04935,
    "correct_predictions": 0.96855,
    "correct_predictions_std": 0.04935
  },
  {
    "scenario": "S7_colocated",
    "scheduler": "sweep",
    "n": 5,
    "pd": 0.08973,
    "pd_std": 0.00359,
    "pd_weighted": 0.08008,
    "pd_weighted_std": 0.00405,
    "intercept_rate": 0.485,
    "intercept_rate_std": 0.03354,
    "ttfi_censored_mean": 9.77599,
    "ttfi_censored_mean_std": 2.05848,
    "intercept_time_error_ms": null,
    "intercept_time_error_ms_std": null,
    "pfa": null,
    "pfa_std": null,
    "correct_predictions": null,
    "correct_predictions_std": null
  },
  {
    "scenario": "S7_colocated",
    "scheduler": "random",
    "n": 5,
    "pd": 0.09617,
    "pd_std": 0.01227,
    "pd_weighted": 0.08674,
    "pd_weighted_std": 0.01303,
    "intercept_rate": 0.505,
    "intercept_rate_std": 0.02092,
    "ttfi_censored_mean": 9.13475,
    "ttfi_censored_mean_std": 4.43005,
    "intercept_time_error_ms": null,
    "intercept_time_error_ms_std": null,
    "pfa": null,
    "pfa_std": null,
    "correct_predictions": null,
    "correct_predictions_std": null
  },
  {
    "scenario": "S7_colocated",
    "scheduler": "round_robin",
    "n": 5,
    "pd": 0.11278,
    "pd_std": 0.00892,
    "pd_weighted": 0.10061,
    "pd_weighted_std": 0.00714,
    "intercept_rate": 0.49,
    "intercept_rate_std": 0.03791,
    "ttfi_censored_mean": 8.26166,
    "ttfi_censored_mean_std": 2.63955,
    "intercept_time_error_ms": null,
    "intercept_time_error_ms_std": null,
    "pfa": null,
    "pfa_std": null,
    "correct_predictions": null,
    "correct_predictions_std": null
  },
  {
    "scenario": "S7_colocated",
    "scheduler": "bandit",
    "n": 5,
    "pd": 0.09795,
    "pd_std": 0.01142,
    "pd_weighted": 0.08872,
    "pd_weighted_std": 0.01133,
    "intercept_rate": 0.505,
    "intercept_rate_std": 0.02092,
    "ttfi_censored_mean": 9.89255,
    "ttfi_censored_mean_std": 2.27756,
    "intercept_time_error_ms": null,
    "intercept_time_error_ms_std": null,
    "pfa": null,
    "pfa_std": null,
    "correct_predictions": null,
    "correct_predictions_std": null
  },
  {
    "scenario": "S7_colocated",
    "scheduler": "smart",
    "n": 5,
    "pd": 0.35886,
    "pd_std": 0.0405,
    "pd_weighted": 0.34869,
    "pd_weighted_std": 0.0456,
    "intercept_rate": 0.475,
    "intercept_rate_std": 0.03953,
    "ttfi_censored_mean": 8.65003,
    "ttfi_censored_mean_std": 3.08472,
    "intercept_time_error_ms": 4.38191,
    "intercept_time_error_ms_std": 0.47112,
    "pfa": 0.35695,
    "pfa_std": 0.06764,
    "correct_predictions": 0.64305,
    "correct_predictions_std": 0.06764
  },
  {
    "scenario": "S7_colocated",
    "scheduler": "d3qn",
    "n": 5,
    "pd": 0.35291,
    "pd_std": 0.0295,
    "pd_weighted": 0.34976,
    "pd_weighted_std": 0.03722,
    "intercept_rate": 0.415,
    "intercept_rate_std": 0.05477,
    "ttfi_censored_mean": 12.14562,
    "ttfi_censored_mean_std": 4.63394,
    "intercept_time_error_ms": 4.7949,
    "intercept_time_error_ms_std": 0.54985,
    "pfa": 0.34005,
    "pfa_std": 0.05397,
    "correct_predictions": 0.65995,
    "correct_predictions_std": 0.05397
  }
];

export const OVERALL_FOM_SUMMARY = [
  { scheduler: "sweep", pd: 0.125, pd_weighted: 0.112, intercept_rate: 0.319, ttfi: 13.170, timing_err_ms: null, pfa: null, accuracy: null, category: "Legacy Baseline" },
  { scheduler: "random", pd: 0.138, pd_weighted: 0.136, intercept_rate: 0.328, ttfi: 11.954, timing_err_ms: null, pfa: null, accuracy: null, category: "Stochastic Uniform" },
  { scheduler: "round_robin", pd: 0.168, pd_weighted: 0.158, intercept_rate: 0.329, ttfi: 9.488, timing_err_ms: null, pfa: null, accuracy: null, category: "Prioritized Cycling" },
  { scheduler: "bandit", pd: 0.139, pd_weighted: 0.128, intercept_rate: 0.320, ttfi: 12.682, timing_err_ms: null, pfa: null, accuracy: null, category: "Contextual Bandit (UCB1)" },
  { scheduler: "smart", pd: 0.625, pd_weighted: 0.623, intercept_rate: 0.332, ttfi: 6.470, timing_err_ms: 6.659, pfa: 0.265, accuracy: 0.735, category: "Cognitive Heuristic (D-UCB)" },
  { scheduler: "d3qn", pd: 0.643, pd_weighted: 0.651, intercept_rate: 0.309, ttfi: 7.214, timing_err_ms: 6.417, pfa: 0.287, accuracy: 0.713, category: "Deep RL (Dueling Double DQN)" },
];

export const FALLBACK_DEINTERLEAVE = `# Pulse Deinterleaving & Clustering (held-out seeds)

| scenario | method | ari | ami | v_measure | homogeneity | completeness | n_true | n_pred | sec |
|---|---|---|---|---|---|---|---|---|---|
| S2_dense | dbscan | 0.989 | 0.969 | 0.971 | 0.991 | 0.952 | 27.2 | 47.2 | 0.230 |
| S2_dense | hdbscan | 0.995 | 0.981 | 0.982 | 0.990 | 0.974 | 27.2 | 30.2 | 0.146 |
| S2_dense | learned | 0.978 | 0.966 | 0.967 | 0.964 | 0.971 | 27.2 | 20.6 | 0.908 |
| S7_colocated | dbscan | 0.800 | 0.843 | 0.844 | 0.979 | 0.759 | 15.1 | 42.0 | 0.691 |
| S7_colocated | hdbscan | 0.808 | 0.852 | 0.853 | 0.993 | 0.763 | 15.1 | 29.4 | 0.368 |
| S7_colocated | learned | 0.885 | 0.916 | 0.916 | 0.910 | 0.935 | 15.1 | 16.4 | 3.015 |

## Turing Synthetic Radar Dataset Benchmark (test_scan)

Evaluated on 10 pulse train recordings (subset of 5000 pulses each) from the Alan Turing Institute synthetic radar benchmark. The fine-tuned Transformer encoder achieves higher cluster purity (ARI 0.461 vs 0.428) without over-splitting clusters.

| method | ari | ami | v_measure | homogeneity | completeness | n_true | n_pred | noise_frac | sec |
|---|---|---|---|---|---|---|---|---|---|
| dbscan | 0.428 | 0.586 | 0.592 | 0.983 | 0.464 | 12.5 | 41.8 | 0.011 | 0.087 |
| hdbscan | 0.428 | 0.590 | 0.595 | 0.984 | 0.468 | 12.5 | 33.8 | 0.007 | 0.071 |
| learned_sim | 0.440 | 0.618 | 0.623 | 0.929 | 0.536 | 12.5 | 20.6 | 0.014 | 0.445 |
| learned_turing | 0.461 | 0.591 | 0.595 | 0.955 | 0.480 | 12.5 | 25.0 | 0.020 | 0.404 |
`;

export const FALLBACK_PREDICT = `# MFR Mode Transition Prediction (held-out seeds)

Evaluation of predictive models for multi-function radar (MFR) beam scheduling. Both unmasked full history and realistic 70% partially-observed sequences are evaluated.

| observed | model | next_word_acc | next_mode_acc | transition_acc | change_auc | where_next_acc | nll | steps | transitions |
|---|---|---|---|---|---|---|---|---|---|
| all | unigram | 0.707 | 0.707 | 0.283 | 0.744 | 0.489 | 1.135 | 26116 | 2311 |
| all | persistence | 0.892 | 0.912 | 0.000 | 0.237 | 0.675 | 0.757 | 26116 | 2311 |
| all | ngram3 | 0.894 | 0.912 | 0.003 | 0.803 | 0.777 | 0.418 | 26116 | 2311 |
| all | ngram6 | 0.890 | 0.910 | 0.019 | 0.819 | 0.757 | 0.429 | 26116 | 2311 |
| all | gru | 0.894 | 0.910 | 0.044 | 0.865 | 0.756 | 0.346 | 26116 | 2311 |
| 70% | unigram | 0.707 | 0.707 | 0.297 | 0.750 | 0.507 | 1.135 | 18257 | 2195 |
| 70% | ngram3 | 0.861 | 0.879 | 0.012 | 0.797 | 0.757 | 0.518 | 18257 | 2195 |
| 70% | gru | 0.862 | 0.878 | 0.044 | 0.840 | 0.737 | 0.435 | 18257 | 2195 |
`;

export const FALLBACK_ABLATION = `# Smart-Scheduler Ablation (threat-weighted Pd, held-out seeds)

Systematic removal of cognitive scheduling components across 7 scenarios to measure performance attribution.

| scheduler | variant | S1_sparse | S2_dense | S3_mfr | S4_agile | S5_popup | S6_lockin | S7_colocated | mean | TTFI [s] |
|---|---|---|---|---|---|---|---|---|---|---|
| smart | full scheduler | 0.678 | 0.477 | 0.606 | 0.826 | 0.550 | 0.878 | 0.349 | 0.623 | 6.470 |
| smart-no_lock | no period lock (no predicted-beam dwells) | 0.359 | 0.333 | 0.406 | 0.690 | 0.160 | 0.362 | 0.156 | 0.352 | 5.645 |
| smart-no_acquire | no acquisition revisits (locks form only by chance) | 0.153 | 0.258 | 0.318 | 0.333 | 0.379 | 0.176 | 0.294 | 0.273 | 12.663 |
| smart-no_lock_no_acquire | tracker only: bandit exploration | 0.133 | 0.117 | 0.095 | 0.251 | 0.084 | 0.123 | 0.083 | 0.127 | 12.854 |
| smart-sweep_explore | exploration by linear sweep instead of D-UCB | 0.671 | 0.529 | 0.625 | 0.752 | 0.515 | 0.338 | 0.360 | 0.541 | 10.297 |
| smart-random_explore | exploration uniformly at random instead of D-UCB | 0.654 | 0.478 | 0.652 | 0.831 | 0.542 | 0.838 | 0.318 | 0.616 | 6.937 |
| smart-no_jitter | D-UCB exploration without dwell jitter | 0.692 | 0.485 | 0.615 | 0.796 | 0.494 | 0.641 | 0.331 | 0.579 | 8.395 |
`;

export const FALLBACK_EDGE = `# Edge Inference Latency & Quantization Benchmark

Target deployment: Embedded SDR / FPGA / edge mission computer. Model sizes: \`d3qn_fp32.onnx\` (601 KiB), \`d3qn_int8.onnx\` (171 KiB).

FP32 vs INT8 greedy-action agreement on 12935 recorded states: **0.764**

## Latency per decision (µs)

| path | p50_us | p99_us |
|---|---|---|
| PyTorch CPU | 155.4 | 1169.6 |
| ONNX Runtime FP32 (1 thread) | 93.6 | 143.6 |
| ONNX Runtime INT8 (1 thread) | 197.2 | 270.2 |
| Full decision (features + INT8), S2_dense | 755.7 | 1680.2 |

## Scheduling metrics, FP32 vs INT8 (seed 150)

| scenario | model | pd | pd_weighted |
|---|---|---|---|
| S1_sparse | fp32 | 0.643 | 0.705 |
| S1_sparse | int8 | 0.929 | 0.919 |
| S3_mfr | fp32 | 0.729 | 0.762 |
| S3_mfr | int8 | 0.704 | 0.736 |
| S6_lockin | fp32 | 0.883 | 0.867 |
| S6_lockin | int8 | 0.745 | 0.793 |

**Recommendation:** Deploy the FP32 ONNX policy. On this ~150k-parameter network, dynamic INT8 is slower on CPU (quantize/dequantize overhead dominates) and changes a noticeable share of greedy actions. The decision loop is dominated by Python feature extraction, which is the part to port to C++/FPGA for microsecond-level budgets.
`;
