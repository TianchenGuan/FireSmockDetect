"""Append V5-acquired candidate rows (BAMBI, MONIMET) to DATA_INVENTORY_V5.{csv,md}.
Run AFTER make_data_inventory_v5.py (which only knows v1-v4 manifests).
Sources: data/v5/manifests/bambi_manifest.parquet (+bambi_CARD.md),
data/v5/manifests/newfootage_day1_ingest.csv (MONIMET row)."""
import csv, pandas as pd
from pathlib import Path
CSV = Path("data/v5/manifests/DATA_INVENTORY_V5.csv"); MD = Path("data/v5/manifests/DATA_INVENTORY_V5.md")
rows = list(csv.DictReader(open(CSV))); cols = list(rows[0].keys())
def blank(): return {c: "" for c in cols}
b = pd.read_parquet("data/v5/manifests/bambi_manifest.parquet")
dark = b[b.flight_id.astype(str).str.contains("367")] if "flight_id" in b else b.iloc[0:0]
usable = b[~b.index.isin(dark.index)]
r = blank(); r.update(source="bambi_raw (V5 acquisition)", unit_group="13 wildlife-survey UAV flights, Austria 2023-24 (12 new + flight_98)",
  n_units=str(b.flight_id.nunique()), n_images_or_frames=str(len(b)),
  hours_if_video=f"{len(usable)/3600:.2f} h reviewable @1 fps (flight_367 pre-dawn black excluded: {len(dark)} fr); 11 physical sites / 10 usable",
  positives="0 (expected)", verified_negatives="0 (human review pending)", negative_evidence_type="none (dataset has no smoke by design; UNVERIFIED)",
  onset_truth="no", scenario="aerial", purpose_v5="calibration/val candidate (never training)", licence="CC-BY-4.0 (Zenodo records re-verified per part)",
  permission_status="confirmed", notes="nadir-dominant, mixed habitat (farmyard take-offs, enclosures, forest canopy for 98/58/61); flight_269 tractor exhaust flag; review pkg data/v5/review/bambi/ (179 sheets); see bambi_CARD.md")
rows.append(r)
ing = list(csv.DictReader(open("data/v5/manifests/newfootage_day1_ingest.csv")))
for i in ing:
    if not i["source"].startswith("monimet"): continue
    r = blank(); r.update(source="monimet_mc124 (V5 acquisition)", unit_group=i["unit"], n_units=i["n_subunits"], n_images_or_frames=i["n_frames"],
      hours_if_video="30-min SAMPLED cadence; frame-FP source, not episodes (PROTOCOL §3 note)", positives="≥2 frames with REAL smoke found (2020-04-20/21 field burning) — agent-provisional",
      verified_negatives="0 (human review pending)", negative_evidence_type="none (phenocam archive; UNVERIFIED)", onset_truth="no", scenario="fixed-camera",
      purpose_v5="calibration/val candidate (never training)", licence=i["licence"], permission_status="confirmed",
      notes="1 camera-year, ONE unit; candidate FP: expB 5.4% / B4 0.8% of frames; review pkg data/v5/review/newfootage_day1/")
    rows.append(r)
with open(CSV, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows(rows)
MD.write_text(MD.read_text() + f"""

---
## V5 acquisitions (appended by scripts/v5/inventory_v5_extras.py)
| source | units | frames | candidate hours | purpose | licence | verified |
|---|---|---|---|---|---|---|
| BAMBI raw UAV (Austria) | {b.flight_id.nunique()} flights / 11 sites | {len(b)} | {len(usable)/3600:.2f} h reviewable @1 fps | aerial calibration/val candidate | CC-BY-4.0 | 0 h (pending human review) |
| MONIMET MC124 phenocam (Finland) | 1 camera-year / 183 camera-days | 6375 | 30-min sampled (frame-FP only) | fixed-camera calibration/val candidate | CC BY 4.0 | 0 (pending; contains ≥2 real-smoke frames) |

Acceptance (PROTOCOL §6) after V5 Day-1: aerial candidate footage 1.15 h of the 2 h bar, 0 h verified; fixed-camera continuous footage still 0 h (MONIMET is sampled); onset events human-reviewed 0 of 10.
FIgLib licence = CC BY-NC-ND 4.0 (Amendment A1) is now in the generator.
""")
print(f"appended {len(rows)-len(list(csv.DictReader(open(CSV))))+2} rows; total {len(rows)}")
