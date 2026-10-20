_generated 2026-09-15T03:55Z by scripts/v4/make_report_tables.py from saved artifacts — do not edit by hand_

### T28. FINAL TEST roster: systems, checkpoints, selection basis, fixed thresholds, provenance

Once-guards: YOLO `.EVALUATED_ONCE` = 2026-09-15T03:32:30Z; DINO `.DINO_EVALUATED_ONCE` = DINO final test eval started 2026-09-15T02:31:44Z by tianrgua on ip-172-19-234-75.amazon.com | runs: v4B6_lora_Hplus v4B1_frozen_L v4B4_lora_L | thresholds: reports/v4/test/dino_test_thresholds.json | completed 2026-09-15T02:42:52Z failed=0. The DINO half was launched at 02:31Z by the pre-restart coordinator session with exactly the pre-registered systems/thresholds and was NOT re-run (STATUS.md 02:40Z, 03:00Z). D_curr is a POST-HOC addition: it was not a pre-registered candidate and its 3-seed median became available only after the test job launched; the shipped checkpoint coincidentally equals the median-seed pick (STATUS.md 03:00Z). Thresholds were fixed on VAL before any test prediction (reports/v4/test/dino_test_thresholds.json; yolo_test_at_preregistered_thresholds.json `thresholds`).

| id | system | checkpoint | selection basis (val) | fixed thresholds (conf) | role |
|---|---|---|---|---|---|
| S0 | expE (iteration-2 YOLO11m candidate) | `models/trained/expE_mix_y11m_1280_v2/seg05/best.pt` (iteration-2 expE YOLO11m) | deployment baseline; boreal/forest-val thresholds from reports/v4/eval_expE_{borealval,forestval}.json (v4evalr7) | sdis@5/10/20%: 0.721 / 0.6546 / 0.5354; boreal@1/5%: 0.4983 / 0.2615 | deployment baseline (pre-registered) |
| S0b | expB (deployed default, YOLO11s@1280) | `models/expB_best.pt` (byte-identical to models/trained/expB_mix_y11s_1280/seg03/best.pt; RECONCILIATION_V4.md) | deployed default; boreal/forest-val thresholds from reports/v4/eval_expB_{borealval,forestval}.json (v4evalr7) | sdis@5/10/20%: 0.7231 / 0.6503 / 0.5323; boreal@1/5%: 0.4983 / 0.3477 | deployment baseline (pre-registered) |
| S1 | A_ctrl_E, median seed | `v4a_s2` -> models/trained/expV4A_ctrl_s2/seg05/best.pt | median-seed rule: sdis-val R@10% 0.304 (n_seeds=3) | sdis@5/10/20%: 0.7283 / 0.6727 / 0.5392; boreal@1/5%: 0.4883 / 0.2558 | existing-data control (pre-registered) |
| S2 | E_ET, median seed | `v4e_s1` -> models/v4/runs/expV4E_et_s1/seg05/best.pt | median-seed rule: sdis-val R@10% 0.2913 (n_seeds=3); S2 rule -> E_ET (higher 3-seed mean boreal R@1%; tie <1pt -> E_ET (diff=-0.0041)) | sdis@5/10/20%: 0.7154 / 0.6603 / 0.5534; boreal@1/5%: 0.0483 / n/a (val FP floor) | best v4 data arm by S2 rule (pre-registered) |
| S2c | C_EGT, median seed | `v4c_s1` -> models/v4/runs/expV4C_egt_s1/seg05/best.pt | median-seed rule: sdis-val R@10% 0.262 (n_seeds=3) | sdis@5/10/20%: 0.7344 / 0.6768 / 0.5767; boreal@1/5%: 0.2983 / 0.0705 | matched control = the G-value comparison (pre-registered) |
| S3 | B_Gonly, seed 0 | `v4b` (seed 0) | only seed trained | sdis@5/10/20%: 0.4231 / 0.338 / 0.1885; boreal@1/5%: 0.4583 / 0.3558 | completeness (pre-registered) |
| S3 | D_curr, median seed | `v4dada_s1` -> models/v4/runs/expV4Dada_s1/seg03/best.pt | median-seed rule: sdis-val R@10% 0.318 (n_seeds=3) | sdis@5/10/20%: 0.7292 / 0.653 / 0.5258; boreal@1/5%: 0.1683 / 0.1303 | POST-HOC addition (not a pre-registered candidate) |
| S4 | v4B6_lora_Hplus (DINOv3 LoRA-H+) | `/scratch/scratch/tianrgua-sandbox/smoke/runs_v3/v4B6_lora_Hplus/best.pt` (--lora 16) | pre-registered pick (rule: best sdis-val matched-FP) | sdis@5/10/20%: 0.9939 / 0.9901 / 0.9801; boreal@1/5%: 0.8883 / 0.5115 | DINO rule pick: best sdis-val R@5% (pre-registered) |
| S4c | v4B1_frozen_L (DINOv3 frozen-L) | `/scratch/scratch/tianrgua-sandbox/smoke/runs_v3/v4B1_frozen_L/best.pt` (plain) | control (frozen backbone) | sdis@5/10/20%: 0.9677 / 0.9407 / 0.868; boreal@1/5%: 0.4983 / 0.0558 | DINO control (pre-registered) |
| S4p | v4B4_lora_L (DINOv3 LoRA-L) | `/scratch/scratch/tianrgua-sandbox/smoke/runs_v3/v4B4_lora_L/best.pt` (--lora 16) | post-hoc efficiency candidate (labelled post-hoc) | sdis@5/10/20%: 0.9934 / 0.9881 / 0.973; boreal@1/5%: 0.8683 / 0.1458 | POST-HOC efficiency candidate (labelled) |

Sources:
- `reports/v4/test/yolo_test_at_preregistered_thresholds.json`
- `reports/v4/test/dino_test_thresholds.json`
- `reports/v4/seed_summary.json (selection)`
- `reports/v4/replay/replay_{v4a_s2,v4e_s1,v4c_s1,v4dada_s1}_figlib39.json (weights)`
- `reports/v4/test/.EVALUATED_ONCE, .DINO_EVALUATED_ONCE`
- `STATUS.md 02:40Z/03:00Z/03:55Z`
