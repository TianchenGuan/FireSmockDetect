#!/usr/bin/env python3
"""Build data/v5/manifests/bambi_manifest.parquet, bambi_CARD.md and the negative_review input CSV
from the extracted BAMBI flights under data/v5/raw/bambi_raw/flight_*/ (needs frames_1fps/index.csv + _extract_log.json).
No model inference. Usage: bambi_build_manifest.py [--review-csv data/v5/review/_inputs/bambi_negatives.csv]"""
import argparse, csv, datetime as dt, glob, json, os, re
from pathlib import Path
import pandas as pd

ROOT = Path("/local/home/tianrgua/cvproject")
RAW = ROOT / "data/v5/raw/bambi_raw"
MAN = ROOT / "data/v5/manifests"
LICENCE = "CC-BY-4.0"
LIC_VERBATIM = ("docs/dataset-versions.md (github.com/bambi-eco/Dataset): \"Every version of this dataset is **CC-BY-4.0** except "
                "`environment-nc`, which is **CC-BY-NC-4.0**.\" Zenodo record licence id on every record used: cc-by-4.0, access_right: open.")
W = {1: "sunny", 2: "cloudy", 4: "rainy", 8: "snowy", 16: "windy", 32: "foggy"}
REGION = {"NOe": "Lower Austria", "OOe": "Upper Austria", "Ktn": "Carinthia", "Stm": "Styria", "Sbg": "Salzburg", "T": "Tyrol"}


def wx(v):
    return "+".join(n for b, n in W.items() if v and v & b) or "not recorded"


def season(d):
    m = int(d[5:7]); return "winter" if m in (12, 1, 2) else "spring" if m in (3, 4, 5) else "summer" if m in (6, 7, 8) else "autumn"


def region_of(site, lat, lon):
    code = re.sub(r"^\d{4}_\d{2}_\d{2}_", "", site).split("_")[0]
    if code in REGION: return REGION[code]
    if "Gaenserndorf" in site: return "Lower Austria"
    if "Haag" in site and lat: return "Lower Austria (Tierpark Haag area)" if lon > 14.3 else "Upper Austria (Haag am Hausruck area)"
    return "Austria (state not encoded)"


HABITAT = {  # agent contact-sheet screening 2026-10-02 (first + mid sheet per flight); NOT a verdict, human must confirm
    "flight_7": "Tierpark Haag wildlife park: launch pad, then ~1.5 min of SKY (gimbal +45 deg), then oblique views of deciduous woodland, enclosures, pond, buildings; two segments 16 min apart",
    "flight_58": "alpine spruce forest with forest road and a parked car; nadir; clean forest",
    "flight_61": "snow-covered alpine spruce forest with tracks; nadir; clean forest; two segments",
    "flight_136": "wild-boar enclosure next to a farm: farm buildings, red car, paved road, bare ground, leafless trees, snow patches; overcast/grey; forest-adjacent",
    "flight_144": "BAMBI folder says 'Gaenserndorf' + Matrice 30T, but GPS = Grafenstein (Carinthia, same site as flight_149) and file naming is Mavic-style: snow-covered farm buildings, fields, snow-laden trees; frame 0 is a horizontal view",
    "flight_149": "wild-boar enclosure (Thon): snow-covered buildings, fences, feeding racks, animals, frosted trees; sunny",
    "flight_157": "farmyard with buildings and cars at take-off, then meadows, pond and frosted deciduous/conifer trees; forest-adjacent",
    "flight_252": "houses, cars, woodpile at take-off, then larch/spruce forest and a large meadow with a power pole; 14 min, mixed",
    "flight_255": "farmland-forest mosaic: fields, dirt road, hedgerow trees, meadows (autumn colours); forest-adjacent",
    "flight_269": "OBLIQUE (-35..-68 deg): tall-grass meadow and forest edge, then an active farmyard (red tractor mowing, silos, machinery) - tractor exhaust possible, check",
    "flight_283": "settlement edge in spruce forest: houses with red roofs, road, later a gravel stream bed with deciduous trees; nadir",
    "flight_367": "PRE-DAWN (06:37 local, late Oct): all 444 frames essentially black (mean luma 6.7/255) - not reviewable in RGB, exclude",
    "flight_98": "dense deciduous/mixed forest canopy (Wienerwald); nadir; clean forest",
}


SITE_KEY = {  # physical site from SRT GPS (Nominatim) + BAMBI sub-site; same key = same location
    "flight_7": "Haag_Tierpark_NOe", "flight_58": "See_Tirol", "flight_61": "See_Tirol", "flight_98": "Purkersdorf_Wienerwald_NOe",
    "flight_136": "StVeit_Mairist_Ktn", "flight_144": "Grafenstein_Thon_Ktn", "flight_149": "Grafenstein_Thon_Ktn", "flight_157": "Klagenfurt_Bichl_Ktn",
    "flight_252": "Villach_Oberwollanig_Ktn", "flight_255": "Riegersburg_Lembach_Stm", "flight_269": "Klagenfurt_Mageregg_Ktn",
    "flight_283": "StPankraz_OOe", "flight_367": "Ansfelden_OOe",
}


def view_class(p):
    if p != p or p is None: return "unknown"
    return "nadir" if p <= -80 else ("oblique" if p <= -10 else "horizontal_or_up")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--review-csv", default=str(ROOT / "data/v5/review/_inputs/bambi_negatives.csv")); a = ap.parse_args()
    sel = {r["flight_id"]: r for r in json.load(open(RAW / "_selection_v5.json"))}
    meta = {f"flight_{r['key']}": r for r in json.load(open(RAW / "_github/flight_metadata_table.json"))}
    avail = {f"flight_{r['flight']}": r for r in json.load(open(RAW / "_zenodo/available_flights.json"))}
    geo = json.load(open(RAW / "_geocode_nominatim.json")) if (RAW / "_geocode_nominatim.json").exists() else {}
    luma = pd.read_csv(RAW / "_frame_luma.csv").set_index("abs_path")["mean_luma"] if (RAW / "_frame_luma.csv").exists() else None
    frames = []; flights = []
    for fdir in sorted(RAW.glob("flight_*"), key=lambda p: int(p.name.split("_")[1])):
        idx = fdir / "frames_1fps/index.csv"; lg = fdir / "_extract_log.json"
        if not (idx.exists() and lg.exists()): continue
        fid = fdir.name; log = json.load(open(lg)); av = avail.get(fid, {}); md = meta.get(fid, {})
        dl = None
        if (fdir / "_download_log.jsonl").exists():
            dl = [json.loads(l) for l in open(fdir / "_download_log.jsonl")][-1]
        df = pd.read_csv(idx)
        df = df.sort_values(["ts_utc", "segment", "video_frame"]).reset_index(drop=True)
        site = md.get("site") or av.get("site") or ""; date = (md.get("start") or "")[:10]
        lat = pd.to_numeric(df["lat"], errors="coerce").mean() if "lat" in df else None
        lon = pd.to_numeric(df["lon"], errors="coerce").mean() if "lon" in df else None
        rec_url = f"https://doi.org/{av['doi']}" if av.get("doi") else "https://doi.org/10.5281/zenodo.19177615"
        out = pd.DataFrame(dict(
            flight_id=fid, site=site, date=date, frame_idx=range(len(df)), ts=pd.to_datetime(df["ts_utc"], utc=True, format="ISO8601", errors="coerce"),
            abs_path=[str(fdir / "frames_1fps" / f) for f in df["frame_file"]],
            altitude_m=pd.to_numeric(df["rel_alt_m"], errors="coerce"), licence=LICENCE, source_record_url=rec_url,
            verified_no_smoke=False, review_status="pending",
            segment=df["segment"], video_file=df["video_file"], video_frame=df["video_frame"], ts_source=df["ts_source"], ts_local=df["ts_local"],
            lat=pd.to_numeric(df["lat"], errors="coerce"), lon=pd.to_numeric(df["lon"], errors="coerce"),
            abs_alt_m=pd.to_numeric(df["abs_alt_m"], errors="coerce"), gimbal_pitch_deg=pd.to_numeric(df["gimbal_pitch_deg"], errors="coerce"),
            width=df["width"], height=df["height"], region=region_of(site, lat, lon), season=season(date) if date else "",
            drone=md.get("drone", ""), weather_flags=wx(md.get("weather")), bambi_split=md.get("split", ""), species_labelled=md.get("species", ""),
            zenodo_part=av.get("part"), zenodo_recid=av.get("recid"), zip_md5=log.get("zip_md5"), zip_bytes=log.get("zip_bytes"),
            purpose="AERIAL no-smoke CALIBRATION/VAL candidate (never training)", scenario="aerial", cadence_s=1.0))
        g = geo.get(fid, {})
        if g.get("state"):
            out["region"] = g["state"]; out["municipality"] = g.get("municipality", "")
        out["mean_luma"] = out["abs_path"].map(luma) if luma is not None else float("nan")
        out["view_class"] = out["gimbal_pitch_deg"].map(view_class)
        out["quality_flag"] = ["dark" if (l == l and l < 20) else "ok" for l in out["mean_luma"]]
        out["habitat_note_agent"] = HABITAT.get(fid, "")
        out["physical_site"] = SITE_KEY.get(fid, site)
        frames.append(out)
        vids = [v for v in log["videos"] if v["status"] == "ok"]
        flights.append(dict(flight_id=fid, site=site, region=(geo.get(fid, {}).get("state") or region_of(site, lat, lon)), municipality=geo.get(fid, {}).get("municipality", ""), date=date, season=season(date) if date else "", local_start=(md.get("start") or "")[11:16],
                            drone=md.get("drone", ""), weather=wx(md.get("weather")), species=re.sub(r"\s*\([^)]*\)", "", md.get("species") or "") or "-",
                            n_rgb_videos=len(vids), rgb_video_s=log["total_rgb_video_s"], minutes=round(log["total_rgb_video_s"] / 60, 1), n_frames=len(df),
                            zip_gb=round(log["zip_bytes"] / 1e9, 2), zip_md5=log["zip_md5"], md5_status=(dl or {}).get("status", "sample-zip (verified in bambi_raw_sample/CARD.md)"),
                            part=av.get("part"), recid=av.get("recid"), doi=av.get("doi"), lat=round(lat, 4) if lat == lat and lat else None, lon=round(lon, 4) if lon == lon and lon else None,
                            alt_min=round(out["altitude_m"].min(), 1) if out["altitude_m"].notna().any() else None, alt_max=round(out["altitude_m"].max(), 1) if out["altitude_m"].notna().any() else None,
                            pitch=f"{out['gimbal_pitch_deg'].min():.1f}..{out['gimbal_pitch_deg'].max():.1f}" if out["gimbal_pitch_deg"].notna().any() else "n/a",
                            res=f"{int(df['width'].iloc[0])}x{int(df['height'].iloc[0])}", srt=all(v.get("srt_found") for v in vids), ts_src=",".join(sorted(df["ts_source"].unique())),
                            zip_deleted=log.get("zip_deleted"), frames_dir=str(fdir / "frames_1fps"),
                            mean_luma=round(float(out["mean_luma"].mean()), 1) if out["mean_luma"].notna().any() else None,
                            dark_frames=int((out["quality_flag"] == "dark").sum()), nadir_frac=round(float((out["view_class"] == "nadir").mean()), 2),
                            habitat=HABITAT.get(fid, ""), physical_site=SITE_KEY.get(fid, site)))
    if not frames:
        print("no extracted flights found"); return
    M = pd.concat(frames, ignore_index=True)
    MAN.mkdir(parents=True, exist_ok=True)
    M.to_parquet(MAN / "bambi_manifest.parquet", index=False)
    pd.DataFrame(flights).to_csv(MAN / "bambi_flights.csv", index=False)
    tot_min = sum(f["minutes"] for f in flights)
    # ---- review input CSV for scripts/v5/negative_review.py ----
    Path(a.review_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(a.review_csv, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["unit", "path", "scenario", "source", "independent_unit", "license", "cadence_s", "sample_s", "contamination", "notes"])
        for f in flights:
            w.writerow([f"bambi_{f['flight_id']}", f["frames_dir"], "aerial", "bambi", f["physical_site"],
                        f"CC-BY-4.0 (BAMBI raw, Zenodo doi:{f['doi']}; data/v5/manifests/bambi_CARD.md)", 1.0, "", "",
                        f"BAMBI wildlife-survey flight {f['flight_id']} {f['site']} {f['date']} {f['local_start']} local, {f['drone']}, {f['weather']}, nadir {f['pitch']} deg, "
                        f"rel alt {f['alt_min']}-{f['alt_max']} m, {f['minutes']} min RGB @1 fps; no fire by design, smoke-free EXPECTED not verified; AERIAL CALIBRATION/VAL candidate - never training"])
    # ---- CARD ----
    now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
    n_dates = len({f["date"] for f in flights}); n_regions = len({f["region"] for f in flights})
    n_sites = len({re.sub(r"^\d{4}_\d{2}_\d{2}_", "", f["site"]) for f in flights})
    parts = sorted({f["part"] for f in flights if f["part"]})
    L = [f"# bambi_raw — BAMBI Raw Dataset, {len(flights)} flights (CARD, V5, built {now})", "",
         "## Provenance",
         "- Project: BAMBI — Biodiversity Airborne Monitoring Based on Intelligent UAV sampling, University of Applied Sciences Upper Austria (https://www.bambi.eco/; code + per-flight metadata https://github.com/bambi-eco/Dataset, snapshot saved under data/v5/raw/bambi_raw/_github/).",
         "- Dataset version used: `raw` = \"The original, unprocessed videos as they came off the drone, with undistortion parameters, SRT subtitles and Airdata logs\" (one zip per flight, named flight_<id>.zip). Zenodo series \"The BAMBI Raw Dataset: Unprocessed Nadir UAV-Recordings of Forest Wildlife (Part N)\", creators Praschl, Schedl, Maschek, Probst, Stöckl, Böss.",
         f"- Zenodo parts used: {parts} (record JSONs saved under data/v5/raw/bambi_raw/_zenodo/). Flight selection and rationale: data/v5/raw/bambi_raw/_selection_v5.json; per-flight _download_log.jsonl (md5 vs Zenodo checksum), _zip_listing.json, _extract_log.json, meta/ (SRT, Airdata CSV, calibration JSON).",
         "- Availability note (2026-10-02): of the 47 raw parts listed in the GitHub docs, 30 resolve on Zenodo (parts 1-8, 17-25, 35-47; 262 flights); parts 9-16 and 26-34 return HTTP 404 both via api/records and doi.org and are absent from the Zenodo search index -> not published / not reachable; nothing was taken from them.",
         "- flight_98 was downloaded on Day 1 as data/v5/raw/bambi_raw_sample/flight_98.zip (see its CARD); its frames were extracted here from that zip, which is kept there.", "",
         "## Licence",
         f"- {LIC_VERBATIM}",
         "- URL of the licence statement: https://github.com/bambi-eco/Dataset/blob/main/docs/dataset-versions.md ; licence text: https://creativecommons.org/licenses/by/4.0/",
         "- Re-verified per record at download time (field metadata.license.id == 'cc-by-4.0' and access_right == 'open' for all 30 reachable raw records; the 12 records used are listed in the table). No `environment-nc` material was downloaded.",
         "- Attribution: Praschl, Schedl, Maschek, Probst, Stöckl, Böss — The BAMBI Raw Dataset (Zenodo, 2026), University of Applied Sciences Upper Austria, CC BY 4.0.", "",
         "## Flights",
         "| flight | site (BAMBI folder) | state / municipality (Nominatim from SRT GPS) | date | season | local start | drone (BAMBI metadata) | weather | RGB videos | minutes @1fps | frames | zip GB | md5 | part / record | lat, lon | rel alt m | gimbal pitch | nadir frac | mean luma | habitat / content (agent screening, unverified) |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for f in flights:
        L.append(f"| {f['flight_id']} | {f['site']} | {f['region']} / {f['municipality']} | {f['date']} | {f['season']} | {f['local_start']} | {f['drone']} | {f['weather']} | {f['n_rgb_videos']} | {f['minutes']} | {f['n_frames']} | {f['zip_gb']} | {f['md5_status']} | {f['part']} / {f['recid']} | {f['lat']}, {f['lon']} | {f['alt_min']}-{f['alt_max']} | {f['pitch']} | {f['nadir_frac']} | {f['mean_luma']} | {f['habitat']} |")
    usable = [f for f in flights if f["dark_frames"] < 0.5 * f["n_frames"]]
    us_min = sum(f["minutes"] for f in usable)
    L += ["", f"- **Total: {len(flights)} flights, {tot_min:.1f} min = {tot_min/60:.2f} h of RGB footage, {len(M):,} frames at 1 fps (cadence 1.0 s; JPEG q90, native {flights[0]['res']}). Reviewable (excluding the pre-dawn flight_367): {len(usable)} flights, {us_min:.1f} min = {us_min/60:.2f} h, {int((M['quality_flag']=='ok').sum()):,} frames.** Minutes are measured from the decoded RGB videos (cv2), not from the metadata flight window (a zip holds the recorded segments only; e.g. flight_98's 15.7-min window contains one 96-s segment).",
          f"- Sum of zip sizes downloaded: {sum(f['zip_gb'] for f in flights if f['flight_id']!='flight_98'):.2f} GB (zips deleted after extraction; md5 recorded in _download_log.jsonl / _extract_log.json). Frames on disk: {sum(os.path.getsize(p) for p in M['abs_path'])/1e9:.2f} GB.",
          "- Species columns in BAMBI metadata (for context only): " + "; ".join(f"{f['flight_id']}: {f['species']}" for f in flights), "",
          "## Independence",
          f"- {len(flights)} flights on {n_dates} distinct calendar dates in {n_regions} Austrian states; seasons: " + ", ".join(f"{s}={sum(1 for f in flights if f['season']==s)}" for s in ("winter","spring","summer","autumn")) + ". Each flight is one independent unit (PROTOCOL_V5 §3: units are never concatenated; §6 counts independent units). Two drones (DJI Mavic 3 Thermal, DJI Matrice 30T) and both leaf-on and leaf-off/snow conditions are represented.",
          "- Physical sites by SRT GPS (not by folder name): 11 distinct among the 13 flights. Same-site pairs: flight_58 + flight_61 (See, Tyrol; May 2023 vs Jan 2024) and flight_144 + flight_149 (Grafenstein/Thon enclosure, Carinthia; 19 Jan vs 16 Feb 2023 - flight_144's BAMBI folder name 'Gaenserndorf' is contradicted by its GPS). Excluding the unreviewable dark flight_367: 12 flights / 10 physical sites / 12 dates. Use the site count, not the flight count, for a strict §6 independent-unit tally.",
          "- flight_157 (Bichl) and flight_269 (Mageregg) are both in Klagenfurt municipality but 5.5 km apart; flight_136 (St. Veit) and flight_144/149 (Grafenstein) are separate enclosures 12+ km apart.", "",
          "## Semantics / intended V5 use",
          "- Purpose tag at ingest (PROTOCOL_V5 §2, assigned by whole flight): **AERIAL NO-SMOKE CALIBRATION/VAL candidate — never training.** No model has seen these frames. `verified_no_smoke=False`, `review_status=pending` for every row until the human review in data/v5/review/bambi/verdicts.csv is filled; only `no_smoke` verdicts promote a flight.",
          "- Review package: data/v5/review/bambi/ (contact sheets per flight, units.csv with §3 valid observed seconds, verdicts.csv template) built with scripts/v5/negative_review.py from " + a.review_csv.replace(str(ROOT) + "/", "") + ".", "",
          "## Caveats",
          "- Mostly nadir (per-frame `gimbal_pitch_deg` and `view_class` in the manifest): " + f"{sum(1 for f in flights if f['nadir_frac'] >= 0.95)} flights are >= 95% nadir; below that: " + ", ".join(f"{f['flight_id']} {int(f['nadir_frac']*100)}%" for f in flights if f['nadir_frac'] < 0.95) + ". flight_269 is entirely oblique (-35..-68 deg), flight_255 has oblique passes, flight_7 mixes nadir, oblique and ~1.5 min of upward sky shots (+45 deg) - filter on `view_class` for a nadir-only set. Boreal UAV footage spans 0 to -90 deg.",
          "- Habitat is NOT uniformly forest: take-off areas are farmyards/settlements and several surveys cover game enclosures, meadows and fields (see per-flight 'habitat / content' column from the agent screening of the first and a middle contact sheet). Clean forest-canopy flights: 98 (Wienerwald), 58 and 61 (alpine spruce, Tyrol); forest-dominant with settlement edge: 283, 252; farmland/enclosure-dominant: 136, 144, 149, 157, 255, 269; wildlife park: 7. Temperate Austria, not boreal Finland.",
          "- flight_367 (Ansfelden, 06:37 local on 2024-10-26) is pre-dawn: every frame has mean luminance < 20/255 (`quality_flag=dark`). It is kept in the manifest for completeness but cannot be reviewed for smoke in RGB and should not count toward candidate hours; a daytime Ansfelden flight (e.g. flight_394/395, 2.3-2.6 GB) would be the replacement if the download budget is extended.",
          "- flight_269 contains an active farmyard with a tractor mowing at dawn (exhaust plumes possible); flight_136 and flight_144 are grey/overcast with possible ground fog - reviewer may return `uncertain` for these.",
          "- Wildlife-survey flights (red deer, wild boar, roe deer, chamois/ibex at Haag — a wildlife-park setting), many at dawn or in winter with snow; several flights have fog or cloud in the BAMBI weather flags — legitimate no-smoke conditions but visually smoke-like, so the reviewer may return `uncertain`.",
          "- Smoke-free status is EXPECTED (no fire by design) but NOT verified: detector silence is not evidence; a human must watch the contact sheets / full-res frames (chimney smoke, mist, dust, breath/exhaust of people and vehicles are possible).",
          "- RGB from the Mavic 3T wide camera (24 mm eq.) / Matrice 30T wide camera; 1-fps JPEG q90 re-encoding of H.265 video. Timestamps: per-frame DJI SRT (local Europe/Vienna, converted to UTC in file names and `ts`); where an SRT was missing the segment start time from the DJI file name plus frame/fps is used (`ts_source`).",
          "- Only 30 of 47 raw Zenodo parts are reachable; the selection is drawn from those. Zip archives were deleted after extraction to save disk (md5 recorded); thermal streams were not extracted.",
          "- A flight zip holds only the recorded segments, so footage per flight (1.4-14.4 min) is far shorter than the BAMBI metadata flight window (8-31 min); ~0.69 GB of zip per minute of 4K RGB+thermal video. Reaching the §6 2 h aerial bar needs roughly 2x this download volume of daytime flights.",
          "- BAMBI metadata errors observed: flight_144's folder/drone fields disagree with its GPS and file naming (see table). Treat folder-derived site names as hints; the manifest's `region`/`municipality` come from Nominatim reverse geocoding of the SRT GPS (data/v5/raw/bambi_raw/_geocode_nominatim.json).",
          ]
    (MAN / "bambi_CARD.md").write_text("\n".join(L) + "\n")
    print(f"manifest rows {len(M):,} flights {len(flights)} total_min {tot_min:.1f}; wrote {MAN/'bambi_manifest.parquet'}, bambi_flights.csv, bambi_CARD.md, {a.review_csv}")


if __name__ == "__main__":
    main()
