# V5b validation selection (FINAL)

Matched negative-image FP rate (IoU>=0.4, scripts/v4/compare_matched_fp.py, fp-targets 1/2/5/10/20%). Inputs: reports/v5b/val/eval_<sys>_<set>.raw.parquet from cluster/v5b/run_val_eval.sh jobs v5bval1 (A,B,C,D) and v5bval2 (D2,D3); baselines S0_expE/S0b_expB from v5bval0. Source jsons: reports/v5b/val/matched_fp_<set>.json. R=box recall, tiny=tiny-GT recall, P=precision; n/a = FP level outside the model's achievable range on that split.

## pyro-sdis VAL — 7335 images, 1158 negatives

| system | R@1% | R@2% | R@5% | R@10% | R@20% | tiny@1% | tiny@2% | tiny@5% | tiny@10% | tiny@20% | P@1% | P@2% | P@5% | P@10% | P@20% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S0_expE | 0.059 | 0.092 | 0.190 | 0.309 | 0.479 | 0.013 | 0.026 | 0.097 | 0.201 | 0.364 | 0.961 | 0.946 | 0.932 | 0.915 | 0.888 |
| S0b_expB | 0.058 | 0.098 | 0.204 | 0.329 | 0.485 | 0.011 | 0.030 | 0.104 | 0.203 | 0.359 | 0.961 | 0.954 | 0.940 | 0.925 | 0.887 |
| S1_A | 0.056 | 0.108 | 0.202 | 0.302 | 0.499 | 0.012 | 0.041 | 0.109 | 0.191 | 0.374 | 0.954 | 0.950 | 0.938 | 0.921 | 0.892 |
| S2_B | 0.056 | 0.103 | 0.183 | 0.302 | 0.457 | 0.011 | 0.027 | 0.075 | 0.174 | 0.335 | 0.965 | 0.957 | 0.941 | 0.916 | 0.881 |
| S3_C | 0.049 | 0.089 | 0.183 | 0.276 | 0.433 | 0.009 | 0.027 | 0.087 | 0.172 | 0.313 | 0.957 | 0.952 | 0.934 | 0.913 | 0.880 |
| S4_D | 0.043 | 0.082 | 0.170 | 0.287 | 0.454 | 0.008 | 0.023 | 0.084 | 0.175 | 0.336 | 0.959 | 0.949 | 0.932 | 0.919 | 0.884 |
| S4b_D2 | 0.044 | 0.083 | 0.158 | 0.274 | 0.444 | 0.011 | 0.033 | 0.074 | 0.167 | 0.325 | 0.956 | 0.946 | 0.927 | 0.911 | 0.881 |
| S4c_D3 | 0.055 | 0.074 | 0.164 | 0.239 | 0.419 | 0.006 | 0.013 | 0.052 | 0.113 | 0.276 | 0.962 | 0.948 | 0.934 | 0.909 | 0.879 |

Conf thresholds at 5/10/20% neg-image FP: S0_expE 0.72/0.65/0.54; S0b_expB 0.72/0.65/0.53; S1_A 0.73/0.67/0.53; S2_B 0.74/0.67/0.56; S3_C 0.75/0.69/0.59; S4_D 0.76/0.69/0.58; S4b_D2 0.77/0.70/0.57; S4c_D3 0.72/0.68/0.56

## boreal heinola VAL — 1123 images, 217 negatives

| system | R@1% | R@2% | R@5% | R@10% | R@20% | tiny@1% | tiny@2% | tiny@5% | tiny@10% | tiny@20% | P@1% | P@2% | P@5% | P@10% | P@20% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S0_expE | 0.553 | 0.698 | 0.751 | 0.779 | 0.810 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.901 | 0.814 | 0.743 | 0.711 | 0.618 |
| S0b_expB | 0.663 | 0.709 | 0.762 | 0.771 | 0.782 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.925 | 0.915 | 0.864 | 0.848 | 0.813 |
| S1_A | 0.562 | 0.743 | 0.800 | 0.814 | 0.837 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.901 | 0.786 | 0.717 | 0.684 | 0.639 |
| S2_B | 0.852 | 0.901 | 0.955 | n/a | n/a | 0.000 | 0.000 | 0.000 | n/a | n/a | 0.866 | 0.784 | 0.583 | n/a | n/a |
| S3_C | 0.861 | 0.891 | 0.946 | 0.948 | n/a | 0.000 | 0.000 | 0.000 | 0.000 | n/a | 0.910 | 0.859 | 0.626 | 0.595 | n/a |
| S4_D | 0.822 | 0.868 | n/a | n/a | n/a | 0.000 | 0.000 | n/a | n/a | n/a | 0.909 | 0.859 | n/a | n/a | n/a |
| S4b_D2 | 0.881 | 0.943 | n/a | n/a | n/a | 0.000 | 0.000 | n/a | n/a | n/a | 0.944 | 0.809 | n/a | n/a | n/a |
| S4c_D3 | 0.821 | 0.865 | n/a | n/a | n/a | 0.000 | 0.000 | n/a | n/a | n/a | 0.928 | 0.895 | n/a | n/a | n/a |

Conf thresholds at 5/10/20% neg-image FP: S0_expE 0.26/0.22/0.15; S0b_expB 0.35/0.33/0.31; S1_A 0.26/0.23/0.18; S2_B 0.02/n/a/n/a; S3_C 0.03/0.02/n/a; S4_D n/a/n/a/n/a; S4b_D2 n/a/n/a/n/a; S4c_D3 n/a/n/a/n/a

## FireMan VAL (s5) — 1706 images, 231 negatives

| system | R@1% | R@2% | R@5% | R@10% | R@20% | tiny@1% | tiny@2% | tiny@5% | tiny@10% | tiny@20% | P@1% | P@2% | P@5% | P@10% | P@20% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S0_expE | 0.963 | 0.971 | 0.981 | 0.993 | n/a | n/a | n/a | n/a | n/a | n/a | 0.523 | 0.503 | 0.489 | 0.439 | n/a |
| S0b_expB | 0.994 | 0.996 | 0.997 | 1.000 | 1.000 | n/a | n/a | n/a | n/a | n/a | 0.360 | 0.346 | 0.334 | 0.315 | 0.295 |
| S1_A | 0.971 | 0.984 | 0.997 | 1.000 | n/a | n/a | n/a | n/a | n/a | n/a | 0.489 | 0.474 | 0.450 | 0.405 | n/a |
| S2_B | 0.992 | 0.993 | 0.995 | 0.997 | 0.997 | n/a | n/a | n/a | n/a | n/a | 0.546 | 0.544 | 0.538 | 0.521 | 0.485 |
| S3_C | 0.994 | 0.995 | 0.998 | 0.998 | 1.000 | n/a | n/a | n/a | n/a | n/a | 0.588 | 0.585 | 0.563 | 0.549 | 0.524 |
| S4_D | 0.992 | 0.995 | 0.996 | 0.998 | n/a | n/a | n/a | n/a | n/a | n/a | 0.606 | 0.593 | 0.583 | 0.558 | n/a |
| S4b_D2 | 0.970 | 0.988 | 0.994 | 0.997 | n/a | n/a | n/a | n/a | n/a | n/a | 0.630 | 0.600 | 0.572 | 0.548 | n/a |
| S4c_D3 | 0.990 | 0.992 | 0.993 | 0.994 | 0.999 | n/a | n/a | n/a | n/a | n/a | 0.621 | 0.611 | 0.606 | 0.587 | 0.502 |

Conf thresholds at 5/10/20% neg-image FP: S0_expE 0.18/0.07/n/a; S0b_expB 0.14/0.06/0.02; S1_A 0.09/0.03/n/a; S2_B 0.13/0.08/0.03; S3_C 0.07/0.04/0.02; S4_D 0.08/0.04/n/a; S4b_D2 0.05/0.02/n/a; S4c_D3 0.26/0.20/0.04

## Selection (pre-registered rule, reports/v5b/PREREGISTRATION_V5B.md)

score = mean(sdis-VAL R@10%, FireMan-VAL R@10%); ties within 0.5pt -> simpler arm (B<C<D<D2<D3).

| arm | sdis R@10% | FireMan R@10% | mean |
|---|---|---|---|
| S2_B | 0.3025 | 0.9966 | 0.6496 |
| S3_C | 0.2761 | 0.9983 | 0.6372 |
| S4_D | 0.2875 | 0.9982 | 0.6428 |
| S4b_D2 | 0.2741 | 0.9974 | 0.6358 |
| S4c_D3 | 0.2394 | 0.9939 | 0.6167 |

Highest mean: S2_B (0.6496). Within 0.5pt: ['S2_B'] -> selected **S2_B** (B).

## Recommended seed repeats (seed 1; repeat with s2 for the second seed) — NOT launched

EP from data/v5b/manifests/CORPUS_B_v1.json budget.epochs=43; A control EP from budget.arm_A.epochs=60. Note: expV5bA_ctrl_s3 ran at seed 0 (run_yolo_arm.sh cannot pass --seed; args.yaml seed: 0), so the matched-control repeat uses run_v4_arm.sh --seed 1 with the pod-local expB yaml (data/yolo/yamls/pod/expB.yaml; no --data-tar needed, data lives at /mnt/local/smoke_data).

```
export SMOKE_REGISTRY_DIR=cluster/v5b; nohup bash cluster/v4/run_v4_arm.sh expV5bB_s1 v5bbs1 --data-yaml-file data/v5b/yolo/yamls/pod/B_v1.yaml --data-tar v5b_b=data/v5b/stage/B_v1.tar --pod-data-root /scratch/scratch/tianrgua-sandbox/smoke/v5bdata --epochs 43 --base-weights yolo11m.pt --seed 1 > logs/v5b/armB_s1.log 2>&1 &
export SMOKE_REGISTRY_DIR=cluster/v5b; nohup bash cluster/v4/run_v4_arm.sh expV5bA_ctrl_s1 v5bactrls1 --data-yaml-file data/yolo/yamls/pod/expB.yaml --epochs 60 --base-weights yolo11m.pt --seed 1 > logs/v5b/armA_s1.log 2>&1 &
```
