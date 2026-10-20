# data/v5/review — human no-smoke verification workspace

Built by `scripts/v5/negative_review.py sheets` (see reports/v5/TOOLS_V5.md).

* `<unit>/sheet_NNN.jpg` — contact sheets (4x6 thumbnails, unit / frame idx / relative time burned in).
* `<unit>/frames_index.csv` — idx -> sheet/row/col, time, full-res path (or video + frame number).
* `units.csv` — per-unit metadata incl. PROTOCOL_V5 §3 valid observed seconds (gap rule applied).
* `verdicts.csv` — THE human verdict file. Fill `verdict` with one of `no_smoke | smoke | uncertain`,
  `first_smoke_frame_or_blank` with the first frame idx where smoke is clearly visible (if any),
  `reviewer`, `notes`, `reviewed_utc`. Blank rows are "unreviewed". Only `no_smoke` counts as verified.
* `verdicts.EXAMPLE_ONLY.csv` — synthetic example to show the summarise output; NOT a review.
* `screening.agent_provisional.csv` — agent (LLM) contact-sheet screening to PRIORITISE human review.
  It is not a verdict and is never read by `summarise`. Three heinola "Has Smoke=false" clips
  (13, 53, 62) show visible smoke; 103 is uncertain. A human must confirm or reject these.
* `_inputs/` — the manifests that were rendered.
* Units `boreal_subsetA_empty_heinola_DJI_00{24,28,48}` are the 217 Subset-A `Empty-Images` stills (v4/v5 aerial negFP
  denominator) grouped per flight; agent screening flags `heinola_DJI_0028_frame14` (clear plume) and frame39-41 (faint).
