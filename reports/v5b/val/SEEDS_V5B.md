# V5b seed repeats: arm A (control) vs arm B, 3 seeds each

Matched negative-image FP rate (IoU>=0.4, `scripts/v4/compare_matched_fp.py --fp-targets 0.01,0.02,0.05,0.10,0.20`), inputs `reports/v5b/val/eval_<sys>_<set>.raw.parquet`. New systems S1_A_s1/S1_A_s2/S2_B_s1/S2_B_s2 come from val job `v5bval3` (`cluster/v5b/run_val_eval.sh`, 2026-10-02 23:35Z - 00:01Z); the others are the v5bval1/v5bval2 parquets reused unchanged. Tables: `reports/v5b/val/matched_fp_<set>_seeds.json`.

## Runs

| system | arm | seed (args.yaml) | epochs | checkpoint |
|---|---|---|---|---|
| S1_A | A (pyro-sdis only, 60 ep) | 0 | 60/60 | `models/trained/expV5bA_ctrl_s3/seg05/best.pt` |
| S1_A_s1 | A (pyro-sdis only, 60 ep) | 1 | 60/60 | `models/v4/runs/expV5bA_ctrl_s1/seg05/best.pt` |
| S1_A_s2 | A (pyro-sdis only, 60 ep) | 2 | 60/60 | `models/v4/runs/expV5bA_ctrl_s2/seg05/best.pt` |
| S2_B | B (CORPUS_B_v1, 43 ep) | 0 | 43/43 | `models/v4/runs/expV5bB_s0/seg05/best.pt` |
| S2_B_s1 | B (CORPUS_B_v1, 43 ep) | 1 | 43/43 | `models/v4/runs/expV5bB_s1/seg04/best.pt` |
| S2_B_s2 | B (CORPUS_B_v1, 43 ep) | 2 | 43/43 | `models/v4/runs/expV5bB_s2/seg04/best.pt` |

Seed verification: `seed: 1` / `seed: 2` in every segment args.yaml of expV5bA_ctrl_s{1,2} and expV5bB_s{1,2} (the earlier run_yolo_arm.sh seed bug is fixed; run_v4_arm.sh passes --seed through). All four orchestrators ended with status.json phase=done, train_rc=0, results.csv rows = epochs. B_s1/B_s2 provenance: the first orchestrators died at seg 4 (2026-10-02 17:34Z, expired Midway credentials during a stale-job resubmit); the pods kept training on the scratch run_root, and the relaunch (`--start-seg 4`, logs/v5b/armB_s{1,2}_resume.log) resumed from the local last.pt (`resumed_from: local`) and finished epoch 43 in one segment. Loss/metric curves are continuous (results.csv epochs 1..43).

## pyro-sdis VAL — 7335 images, 1158 negatives

| system | R@1% | R@2% | R@5% | R@10% | R@20% | tiny@5% | tiny@10% | tiny@20% | P@5% | P@10% | P@20% |
|---|---|---|---|---|---|---|---|---|---|---|---|
| S0_expE | 0.059 | 0.092 | 0.190 | 0.309 | 0.479 | 0.097 | 0.201 | 0.364 | 0.932 | 0.915 | 0.888 |
| S0b_expB | 0.058 | 0.098 | 0.204 | 0.329 | 0.485 | 0.104 | 0.203 | 0.359 | 0.940 | 0.925 | 0.887 |
| S1_A | 0.056 | 0.108 | 0.202 | 0.302 | 0.499 | 0.109 | 0.191 | 0.374 | 0.938 | 0.921 | 0.892 |
| S1_A_s1 | 0.043 | 0.077 | 0.184 | 0.287 | 0.458 | 0.089 | 0.167 | 0.331 | 0.938 | 0.916 | 0.887 |
| S1_A_s2 | 0.050 | 0.080 | 0.204 | 0.313 | 0.493 | 0.100 | 0.179 | 0.338 | 0.940 | 0.924 | 0.891 |
| S2_B | 0.056 | 0.103 | 0.183 | 0.302 | 0.457 | 0.075 | 0.174 | 0.335 | 0.941 | 0.916 | 0.881 |
| S2_B_s1 | 0.037 | 0.082 | 0.180 | 0.292 | 0.449 | 0.085 | 0.173 | 0.325 | 0.933 | 0.908 | 0.871 |
| S2_B_s2 | 0.042 | 0.093 | 0.196 | 0.328 | 0.465 | 0.089 | 0.204 | 0.329 | 0.941 | 0.916 | 0.882 |

## boreal heinola VAL — 1123 images, 217 negatives

| system | R@1% | R@2% | R@5% | R@10% | R@20% | tiny@5% | tiny@10% | tiny@20% | P@5% | P@10% | P@20% |
|---|---|---|---|---|---|---|---|---|---|---|---|
| S0_expE | 0.553 | 0.698 | 0.751 | 0.779 | 0.810 | 0.000 | 0.000 | 0.000 | 0.743 | 0.711 | 0.618 |
| S0b_expB | 0.663 | 0.709 | 0.762 | 0.771 | 0.782 | 0.000 | 0.000 | 0.000 | 0.864 | 0.848 | 0.813 |
| S1_A | 0.562 | 0.743 | 0.800 | 0.814 | 0.837 | 0.000 | 0.000 | 0.000 | 0.717 | 0.684 | 0.639 |
| S1_A_s1 | 0.582 | 0.598 | 0.626 | 0.673 | 0.724 | 0.000 | 0.000 | 0.000 | 0.801 | 0.760 | 0.709 |
| S1_A_s2 | 0.682 | 0.717 | 0.748 | 0.757 | 0.783 | 0.000 | 0.000 | 0.000 | 0.781 | 0.760 | 0.711 |
| S2_B | 0.852 | 0.901 | 0.955 | n/a | n/a | 0.000 | n/a | n/a | 0.583 | n/a | n/a |
| S2_B_s1 | 0.828 | 0.917 | 0.937 | n/a | n/a | 0.000 | n/a | n/a | 0.668 | n/a | n/a |
| S2_B_s2 | 0.850 | 0.863 | 0.925 | 0.930 | 0.939 | 0.000 | 0.000 | 0.000 | 0.711 | 0.677 | 0.618 |

## FireMan VAL (s5) — 1706 images, 231 negatives

| system | R@1% | R@2% | R@5% | R@10% | R@20% | tiny@5% | tiny@10% | tiny@20% | P@5% | P@10% | P@20% |
|---|---|---|---|---|---|---|---|---|---|---|---|
| S0_expE | 0.963 | 0.971 | 0.981 | 0.993 | n/a | n/a | n/a | n/a | 0.489 | 0.439 | n/a |
| S0b_expB | 0.994 | 0.996 | 0.997 | 1.000 | 1.000 | n/a | n/a | n/a | 0.334 | 0.315 | 0.295 |
| S1_A | 0.971 | 0.984 | 0.997 | 1.000 | n/a | n/a | n/a | n/a | 0.450 | 0.405 | n/a |
| S1_A_s1 | 0.960 | 0.965 | 0.972 | 0.983 | 1.000 | n/a | n/a | n/a | 0.510 | 0.475 | 0.389 |
| S1_A_s2 | 0.923 | 0.929 | 0.960 | 0.977 | 0.996 | n/a | n/a | n/a | 0.638 | 0.582 | 0.495 |
| S2_B | 0.992 | 0.993 | 0.995 | 0.997 | 0.997 | n/a | n/a | n/a | 0.538 | 0.521 | 0.485 |
| S2_B_s1 | 0.983 | 0.988 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| S2_B_s2 | 0.968 | 0.980 | 0.993 | 0.997 | n/a | n/a | n/a | n/a | 0.636 | 0.576 | n/a |

## Per-arm 3-seed mean ± std (sample std, n=3)

| set | metric | A (seeds 0,1,2) | B (seeds 0,1,2) | B − A paired mean | B − A range [min, max] | per-pair (s0, s1, s2) |
|---|---|---|---|---|---|---|
| sdisval | R@5% | 0.1967 ± 0.0109 | 0.1865 ± 0.0083 | -0.0102 | [-0.0190, -0.0038] | -0.0190, -0.0038, -0.0077 |
| sdisval | R@10% | 0.3007 ± 0.0128 | 0.3077 ± 0.0185 | +0.0069 | [+0.0001, +0.0156] | +0.0001, +0.0051, +0.0156 |
| sdisval | tiny@10% | 0.1794 ± 0.0121 | 0.1839 ± 0.0173 | +0.0045 | [-0.0169, +0.0244] | -0.0169, +0.0061, +0.0244 |
| borealval | R@5% | 0.7247 ± 0.0889 | 0.9389 ± 0.0150 | +0.2143 | [+0.1552, +0.3102] | +0.1552, +0.3102, +0.1774 |
| borealval | R@10% | 0.7481 ± 0.0709 | n/a | n/a | n/a | n/a, n/a, +0.1732 |
| borealval | tiny@10% | 0.0000 ± 0.0000 | n/a | n/a | n/a | n/a, n/a, +0.0000 |
| firemanval | R@5% | 0.9762 ± 0.0192 | n/a | n/a | n/a | -0.0027, n/a, +0.0335 |
| firemanval | R@10% | 0.9866 ± 0.0120 | n/a | n/a | n/a | -0.0034, n/a, +0.0204 |
| firemanval | tiny@10% | n/a | n/a | n/a | n/a | n/a, n/a, n/a |

Per-seed values (A s0/s1/s2 ; B s0/s1/s2):

- sdisval R@5%: A 0.2023, 0.1841, 0.2036 ; B 0.1833, 0.1803, 0.1959
- sdisval R@10%: A 0.3024, 0.2872, 0.3126 ; B 0.3025, 0.2923, 0.3282
- sdisval tiny@10%: A 0.1914, 0.1673, 0.1795 ; B 0.1745, 0.1734, 0.2039
- borealval R@5%: A 0.7997, 0.6265, 0.7478 ; B 0.9549, 0.9367, 0.9252
- borealval R@10%: A 0.8144, 0.6733, 0.7567 ; B n/a, n/a, 0.9299
- borealval tiny@10%: A 0.0000, 0.0000, 0.0000 ; B n/a, n/a, 0.0000
- firemanval R@5%: A 0.9972, 0.9716, 0.9597 ; B 0.9945, n/a, 0.9932
- firemanval R@10%: A 1.0000, 0.9828, 0.9769 ; B 0.9966, n/a, 0.9973
- firemanval tiny@10%: A n/a, n/a, n/a ; B n/a, n/a, n/a

## Median-seed pick (pre-registered rule: per arm, the seed whose pyro-sdis VAL R@10% is the median of the three)

- **Arm A: S1_A (seed 0)**, sdis R@10% = 0.3024 (others: S1_A_s1=0.2872, S1_A_s2=0.3126) → `models/trained/expV5bA_ctrl_s3/seg05/best.pt`
- **Arm B: S2_B (seed 0)**, sdis R@10% = 0.3025 (others: S2_B_s1=0.2923, S2_B_s2=0.3282) → `models/v4/runs/expV5bB_s0/seg05/best.pt`

### Why some cells are n/a

n/a = the target neg-image FP rate lies above the model's maximum achievable neg-FP on that split (the conf grid bottoms out at 0.01), i.e. the model is too conservative to be pushed to that FP level, not an evaluation failure. Max neg-FP at conf 0.01 and the recall reached there:

- borealval S2_B: max neg-FP 0.051 (recall 0.9553 at conf 0.01)  <- below 10% target
- borealval S2_B_s1: max neg-FP 0.055 (recall 0.9411 at conf 0.01)  <- below 10% target
- firemanval S1_A: max neg-FP 0.117 (recall 1.0000 at conf 0.01)
- firemanval S2_B_s1: max neg-FP 0.043 (recall 0.9892 at conf 0.01)  <- below 10% target  <- below 5% target
- firemanval S2_B_s2: max neg-FP 0.121 (recall 0.9986 at conf 0.01)


Notes: boreal has 217 negatives, so FP 10% rests on ~22 images and some systems never reach 10%/20% neg-FP within the conf grid (n/a → mean/diff reported as n/a). FireMan has no tiny-bucket GT (tiny@10% n/a). Std with n=3 is only indicative; the paired range is the honest spread of the A-vs-B contrast. S0_expE/S0b_expB are the v4 baselines (single seed) for reference only.
