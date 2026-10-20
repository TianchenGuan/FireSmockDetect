"""Render the markdown tables of reports/v4/EARLY_WARNING_V4.md straight from the
eval JSON, so no number in the report is retyped by hand.

Usage: python scripts/v4/trackd_report_tables.py reports/v4/trackd_v4_eval.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ORDER_HINT = ["s1_expE", "s1_v4a_ctrlE", "s1_v4B1_dinoL", "s2_persist_3of5"]


def sysorder(keys):
    ks = [k for k in ORDER_HINT if k in keys]
    ks += sorted(k for k in keys if k not in ORDER_HINT)
    return ks


def table_a(rep):
    A = rep["A_sdis_val_frame_level"]
    pc = rep.get("param_counts", {})
    print("### A. pyro-sdis val, frame level, matched FPR on val negatives")
    print("Frames %d (pos %d / neg %d; clean neg %d); tiny-GT pos frames %d\n"
          % (A["n_frames"], A["n_pos"], A["n_neg"], A["n_clean_neg"],
             A["n_tiny_pos_frames"]))
    for pool in ["all_neg", "clean_neg"]:
        print("\n**pool = %s**\n" % pool)
        print("| system | params | PR-AUC | ROC-AUC | R@FPR0.5% | R@FPR1% | "
              "R@FPR2% | R@FPR5% | R@FPR10% | tinyR@FPR5% |")
        print("|---|---|---|---|---|---|---|---|---|---|")
        for n in sysorder(A["systems"]):
            e = A["systems"][n]
            m = e["matched_fpr"]
            p = pc.get(n, {}).get("n_params")
            print("| %s | %s | %.4f | %.4f | %.4f | %.4f | %.4f | %.4f | %.4f | %.4f |"
                  % (n, "{:,}".format(p) if p else "0 (rule)",
                     e["pr_auc"], e["roc_auc"],
                     m["%s@0.005" % pool]["recall_pos_frames"],
                     m["%s@0.01" % pool]["recall_pos_frames"],
                     m["%s@0.02" % pool]["recall_pos_frames"],
                     m["%s@0.05" % pool]["recall_pos_frames"],
                     m["%s@0.1" % pool]["recall_pos_frames"],
                     m["%s@0.05" % pool]["recall_tiny_frames"]))
    print("\n**episode detection + onset latency (thresholds matched on FRAME "
          "FPR, all_neg pool)**\n")
    print("| system | episodes | det@FPR1% | lat@FPR1% | det@FPR5% | lat@FPR5% |")
    print("|---|---|---|---|---|---|")
    for n in sysorder(A["systems"]):
        m = A["systems"][n]["matched_fpr"]
        a, b = m["all_neg@0.01"], m["all_neg@0.05"]
        print("| %s | %d | %.4f | %s | %.4f | %s |"
              % (n, a["episodes"], a["episode_detection_rate"],
                 a["median_onset_latency_frames"], b["episode_detection_rate"],
                 b["median_onset_latency_frames"]))
    print("\n**deployed discrete operating point**\n")
    print("| point | FPR all_neg | FPR clean_neg | recall pos frames | recall tiny |")
    print("|---|---|---|---|---|")
    for k, v in A["deployed_points"].items():
        print("| %s | %.4f | %.4f | %.4f | %.4f |"
              % (k, v["fpr_all_neg"], v["fpr_clean_neg"],
                 v["recall_pos_frames"], v["recall_tiny_frames"]))


def table_b(rep):
    B = rep.get("B_v4_val_replay")
    if not B:
        return
    print("\n### B. v4 val replay, per source (thresholds from section A)\n")
    for g, v in B["sources"].items():
        wc = ("%.3f h wall clock" % v["wall_clock_h"] if v.get("wall_clock_h")
              else "wall clock UNKNOWN (%s)" % v.get("wall_clock_note"))
        print("\n**%s** - %d sequences / %d frames, %s, "
              "truth_level=%s, frame_truth=%s, units=%d"
              % (g, v["sequences"], v["frames"], wc,
                 v["truth_level"], v["frame_truth"], len(v["parent_units"])))
        if v["notes"]:
            for n in v["notes"]:
                print("  - CAVEAT: %s" % n)
        print()
        has_rec = any("recall_pos_frames" in r for s in v["systems"].values()
                      for r in s.values())
        hdr = "| system | thr@FPR1% | clip alert | frame alert | episodes | ep/h"
        hdr += " | recall pos frames |" if has_rec else " |"
        print(hdr)
        print("|---|---|---|---|---|---|" + ("---|" if has_rec else ""))
        for n in sysorder(v["systems"]):
            r = v["systems"][n].get("matched@fpr0.01")
            if not r:
                continue
            line = ("| %s | %.4f | %.4f (%d/%d) | %.4f | %d | %s"
                    % (n, r["threshold"], r["clip_alert_rate"],
                       r["clips_alerted"], r["clips"], r["frame_alert_rate"],
                       r["episodes_cooldown_merged"],
                       ("%.2f %s" % (r["episodes_per_hour"],
                                     r.get("episodes_per_hour_ci95", "")))
                       if r.get("episodes_per_hour") is not None else "-"))
            line += (" | %s |" % r.get("recall_pos_frames")) if has_rec else " |"
            print(line)
        d = v["deployed_confirmed_or_provisional"]
        print("| DEPLOYED (prov OR conf) | - | %.4f | %.4f | %d | %s | - |"
              % (d["clip_alert_rate"], d["frame_alert_rate"],
                 d["episodes_cooldown_merged"],
                 ("%.2f %s" % (d["episodes_per_hour"],
                               d.get("episodes_per_hour_ci95", "")))
                 if d.get("episodes_per_hour") is not None else "-"))


def table_c(rep):
    C = rep["C_false_alert_episodes_per_hour"]
    print("\n### C. False-alert EPISODES per hour on continuous negative footage")
    print("Episode = maximal run of alert-on frames; runs separated by < %ss "
          "(deployed cooldown) count as one. Provisional AND confirmed both "
          "count. CI = exact Poisson 95%%.\n" % C["cooldown_s"])
    for pool, v in C["pools"].items():
        flag = " **[CONTAMINATED - see caveat]**" if v["contaminated"] else ""
        print("\n**%s**%s - %d seqs / %d frames / %.3f h"
              % (pool, flag, v["sequences"], v["frames"], v["wall_clock_h"]))
        print("  %s\n" % v["note"])
        print("| system | thr@FPR1% | ep/h @FPR1% | CI95 | ep/h @FPR5% | CI95 |")
        print("|---|---|---|---|---|---|")
        for n in sysorder(v["systems"]):
            a = v["systems"][n].get("matched@fpr0.01")
            b = v["systems"][n].get("matched@fpr0.05")
            if not a:
                continue
            print("| %s | %.4f | %s | %s | %s | %s |"
                  % (n, a["threshold"], a["episodes_per_hour"],
                     a.get("episodes_per_hour_ci95"), b["episodes_per_hour"],
                     b.get("episodes_per_hour_ci95")))
        d = v["deployed_confirmed_or_provisional"]
        print("| DEPLOYED (prov OR conf) | 0.30 / 0.15 | %s | %s | - | - |"
              % (d["episodes_per_hour"], d.get("episodes_per_hour_ci95")))


def table_d(rep):
    allD = rep.get("D_yolo_plus_dino_sweep")
    if not allD:
        return
    print("\n### D. YOLO every frame + independent DINO sweep every Nth frame")
    for tag, D in allD.items():
        print("\n#### DINO source: %s (%s)" % (tag, rep["D_dino_sources"][tag]["path"]))
        _table_d_one(D)


def _table_d_one(D):
    L = D["latency_ms_per_frame"]
    print("Measured batch-1 L40S fp32: YOLO11m expE %s ms, frozen DINOv3-L B1 "
          "%s ms -> one DINO pass costs %sx one YOLO pass. N for <=20%% added "
          "compute = **%d**.\n" % (L["yolo11m_expE_L40S"], L["dinov3L_B1_L40S"],
                                   L["cost_ratio"], D["n_for_20pct_added_cost"]))
    sc = D["sequence_coverage"]
    print("Val sequences: %d, median length %.1f frames. Fraction that get ZERO "
          "DINO pass: %s\n" % (sc["n_sequences"], sc["median_len_frames"],
                               json.dumps(sc["frac_sequences_with_zero_dino_pass"])))
    for key in sorted(D["results"]):
        r = D["results"][key]
        y = r["yolo_alone"]
        print("\n**%s** (thr_yolo %.4f, thr_dino %.4f). YOLO alone: episode "
              "detection %.4f of %d episodes, median latency %s frames, %d FP "
              "frames of %d negatives.\n"
              % (key, r["thr_yolo"], r["thr_dino"],
                 y["episode_detection_rate"], y["episodes"],
                 y["median_latency_frames"], y["false_alert_frames"],
                 y["n_negative_frames"]))
        print("| N | added compute | combo ep-det | added FP frames | "
              "YOLO alone at SAME FP | combo - yolo @matched FP | beats? | "
              "median lat delta |")
        print("|---|---|---|---|---|---|---|---|")
        for n, v in r["sweep"].items():
            ye = v["yolo_alone_at_same_fp"]
            print("| %s | %.1f%% | %.4f | %+.1f | %.4f (FP %d) | %+.4f | %s | %s |"
                  % (n, 100 * v["added_compute_frac"],
                     v["episode_detection_rate"],
                     v["added_false_alert_frames_per_phase"],
                     ye["episode_detection_rate"], ye["false_alert_frames"],
                     v["combo_minus_yolo_at_same_fp"],
                     "YES" if v["combo_beats_yolo_at_matched_fp"] else "no",
                     v["median_latency_delta_frames_vs_yolo_alone"]))


def main():
    rep = json.loads(Path(sys.argv[1]).read_text())
    table_a(rep)
    table_b(rep)
    table_c(rep)
    table_d(rep)


if __name__ == "__main__":
    main()
