"""Emit the per-system markdown tables + review workload counts for reports/v5/NEWFOOTAGE_CANDIDATE_FP_V5.md
from data/v5/newfootage/metrics_<set>_summary.json, frame_scores_<set>.csv and data/v5/review/newfootage_day1/.
Exact Poisson 95 % intervals on episode counts (scipy.stats.chi2), as in CALIBRATION_AERIAL_V5 §2. Prints markdown only."""
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from scipy.stats import chi2

ROOT = Path("/local/home/tianrgua/cvproject")
NF = ROOT / "data/v5/newfootage"
REV = ROOT / "data/v5/review/newfootage_day1"
CAVEAT = "candidate FP on UNVERIFIED footage: detector output is not evidence of smoke or of its absence"


def poisson_ci(k, hours):
    lo = 0.0 if k == 0 else chi2.ppf(0.025, 2 * k) / 2
    hi = chi2.ppf(0.975, 2 * k + 2) / 2
    return lo / hours, hi / hours


def table(set_name, scenario):
    S = json.load(open(NF / f"metrics_{set_name}_summary.json"))
    rows = []
    rows.append(f"| system | operating point (conf_single/conf_seq) | alert frames / scored | frame alert rate | units with >=1 alert | episodes | valid h ({S['cadence']}) | episodes per valid h [exact 95 %] | frac. time in alert |")
    rows.append("|---|---|---|---|---|---|---|---|---|")
    for s, d in S["systems"].items():
        if d.get("status") == "NOT SCORED":
            rows.append(f"| {s} | - | NOT SCORED | | | | | | |"); continue
        lo, hi = poisson_ci(d["episodes"], d["valid_h"])
        rows.append(f"| {d['label']} | {d['conf_single']}/{d['conf_seq']} | {d['alert_frames']}/{d['scored']} | "
                    f"{100 * d['frame_alert_rate']:.2f} % | {d['units_with_alert']}/{d['units']} | {d['episodes']} | {d['valid_h']:.3f} | "
                    f"**{d['episodes_per_valid_h']:.3f}** [{lo:.3f}, {hi:.3f}] | {d['frac_time_in_alert']:.4f} |")
    return "\n".join(rows)


def union_counts(set_name):
    fr = list(csv.DictReader(open(NF / f"frame_scores_{set_name}.csv")))
    systems = [c[6:] for c in fr[0] if c.startswith("score_")]
    any_a = [r for r in fr if int(r["n_systems_alert"]) > 0]
    all_a = [r for r in fr if int(r["n_systems_alert"]) == len(systems)]
    units_any = len({r["unit"] for r in any_a})
    by_n = Counter(int(r["n_systems_alert"]) for r in fr)
    pair = Counter()
    for r in any_a:
        pair[tuple(s for s in systems if r[f"alert_{s}"] == "1")] += 1
    month = defaultdict(lambda: [0, 0])
    for r in fr:
        m = (r["ts_as_written"] or "t")[:7]
        month[m][1] += 1; month[m][0] += int(int(r["n_systems_alert"]) > 0)
    return dict(n=len(fr), systems=systems, any=len(any_a), all=len(all_a), units_any=units_any, by_n=dict(sorted(by_n.items())),
                combos=pair.most_common(8), month={k: f"{v[0]}/{v[1]}" for k, v in sorted(month.items())})


def review_counts(set_name):
    mp = REV / f"_inputs_manifest_{set_name}.csv"
    vp = REV / f"frame_verdicts_TEMPLATE_{set_name}.csv"
    if not vp.exists():
        return None
    v = list(csv.DictReader(open(vp)))
    man = list(csv.DictReader(open(mp)))
    sheets = 0
    for m in man:
        d = REV / m["unit"]
        sheets += len(list(d.glob("sheet_*.jpg"))) if d.exists() else 0
    why = Counter(r["why"] for r in v)
    return dict(frames=len(v), review_units=len(man), sheets=sheets, why=dict(why),
                camera_days=len({r["camera_day_or_unit"] for r in v}))


if __name__ == "__main__":
    for set_name, scen in (("monimet_mc124", "fixed"), ("bambi_flight98", "aerial")):
        print(f"\n### {set_name} ({scen}) — {CAVEAT}\n")
        print(table(set_name, scen))
        u = union_counts(set_name)
        print(f"\nunion: frames alerting for >=1 system {u['any']}/{u['n']} (all systems: {u['all']}); units with any alert {u['units_any']}; "
              f"frames by #systems alerting {u['by_n']}; top combos {u['combos']}")
        print(f"by month (any-system alerts / frames): {u['month']}")
        print("review package:", review_counts(set_name))
