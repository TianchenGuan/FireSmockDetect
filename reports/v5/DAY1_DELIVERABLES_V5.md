# V5 Day-1 deliverables — index (2026-10-02, prep phase; 0.31 node-h used of the 1-node cap)

| asked for | delivered | status |
|---|---|---|
| frozen protocol, data use, splits | reports/v5/PROTOCOL_V5.md (+A1 licence amendment) | frozen 00:40Z |
| calibration results, fixed-camera | reports/v5/CALIBRATION_FIXED_V5.md, reports/v5/calib/ | done — thresholds alone cannot cut episodes ≥50% at non-inferior recall (29–31 eps/h for every system; −50% costs 26–29pt recall) |
| calibration results, aerial | reports/v5/CALIBRATION_AERIAL_V5.md, data/v5/calib_aerial/ | done — one burn / 7.3 min; expB needs conf 0.107 and alerts 11/14 clips; T-arms/DINO 4–7/14; 3 clips fire for every system → human check |
| bypass (shadow) demo | reports/v5/SHADOW_DEMO.md, src/serve/shadow.py, scripts/v5/test_shadow_service.py | done — 61 records, all historical_replay; response invariance test PASS; :8000 untouched |
| data inventory by purpose | data/v5/manifests/DATA_INVENTORY_V5.{csv,md} | done — 63 rows incl. V5 acquisitions; permission-pending + quarantine listed |
| candidate annotation packages | data/v5/candidates/sdisval_day1/, data/v5/review/{heinola,bambi,newfootage_day1}/, data/v5/mmodalfire/AUX_LABEL_PACKAGE/ | done — all agent-provisional; consolidated in reports/v5/REVIEW_QUEUE_V5.md |
| author email drafts | reports/v5/EMAIL_DRAFTS_V5.md | drafted, NOT sent — needs sender details + your approval |
| acquisition leads | reports/v5/ACQUISITION_LEADS_V5.md; downloads data/v5/raw/{bambi_raw 13 flights 1.15 h, monimet_mc124, commons_usgov_burn_videos, zenodo_forest_smoke_video} | 46 GB licensed (CC-BY / public domain), all calibration/val candidates, none verified, no model trained on any |
| onset audit | reports/v5/AUDIT_ONSET_V5.md, data/v5/audit/ | done — 0 true onsets in Boreal train clips; sdis flicker = label omission |
| MmodalFire review | reports/v5/MMODALFIRE_REVIEW_V5.md | done — mist = real look-alike; 3/4 onsets flame-first |
| tools | scripts/v5/{negative_review,onset_annotate,export_candidates}.py; reports/v5/TOOLS_V5.md | done, proven on real data |
| new-footage candidate FP | reports/v5/NEWFOOTAGE_CANDIDATE_FP_V5.md | done — MONIMET "no-smoke" set contains real smoke |

## Blockers (specific)
1. **Human review** is the gate for everything: 0 h of verified negatives, 0 reviewed onsets. Queue: reports/v5/REVIEW_QUEUE_V5.md (≈1,800 frames / 91 sheets / 20 sequences / 7 interval proposals / 300 indoor labels).
2. **Aerial negatives**: 1.15 h candidate (BAMBI, 10 usable sites) vs §6 bar 2 h — ~double the download volume needed (daytime flights; 17 Zenodo parts are unpublished). Boreal raw flights require the FGI email.
3. **Fixed-camera continuous negatives**: 0 h (MONIMET is 30-min sampled). Leads: RxCADRE (>20 GB, needs go-ahead), HPWREN/ALERTCalifornia archives (CC BY-NC-ND; contact UCSD OIC), Fintraffic weathercams (new collector).
4. **Onset truth**: 0 of 10 events. Only free source with ignition times = RxCADRE 2012 (needs go-ahead for ~60 GB). FLAME 3 needs an IEEE login.
5. **Licences**: AuSmoke/MultiNatSmoke pending (email); FIgLib is CC BY-NC-ND (eval-only, no derivative release).
6. The "early temporal improvement" conclusion is DEFERRED until onset acceptance is met (PROTOCOL §4); negative-sample training proceeds when §6 negatives are verified.
