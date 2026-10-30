# harness/compare.py - HPA-only vs HPA+AEGIS (and, in v2, vs static over-provisioning).
#   python harness/compare.py --scheme v1      # the v1 A/B: data/ec2 xb-* (HPA only) and xa-* (HPA+AEGIS)   -> results/ab_v1/
#   python harness/compare.py --scheme v2      # the v2 test: data/ec2v2 yb-*, ya-*, ys-* (HPA floor fixed at 4)   -> results/ab_v2/
# Protocol: docs/protocol_ec2.md (v1) and docs/protocol_v2.md (v2). Mann-Whitney U (two-sided), Cliff's delta, Fisher exact for
# breached runs, plus the protocol's PREDICTIVE test: did AEGIS first raise the floor after the surge began and >= 30 s before
# the HPA-only runs breached?
import argparse
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import metrics

BASE, AEGIS, STATIC = "baseline", "aegis", "static4"
LABEL = {BASE: "HPA only", AEGIS: "HPA+AEGIS", STATIC: "static 4 pods"}
COLORS = {BASE: "#718096", AEGIS: "#2b6cb0", STATIC: "#dd6b20"}
SCHEMES = {"v1": {"data": "data/ec2", "runlog": "harness/runs_ec2.csv", "calm": 180, "out": "results/ab_v1",
                  "prefix": {"xb-": BASE, "xa-": AEGIS}},
           "v2": {"data": "data/ec2v2", "runlog": "harness/runs_v2.csv", "calm": 240, "out": "results/ab_v2",
                  "prefix": {"yb-": BASE, "ya-": AEGIS, "ys-": STATIC}}}
TESTS = [("slo_violation_pct", "SLO violation (% of time)"), ("peak_p95_s", "Peak p95 latency (s)"),
         ("recovery_s", "Recovery time (s), breached runs only"), ("replica_minutes", "Replica-minutes (cost)")]
LEAD_S = 30          # protocol v2: a boost only counts as predictive if it starts >= 30 s before the HPA-only breach


def cliffs_delta(a, b):
    """(#a>b - #a<b) / (n_a * n_b). Negative = values in a tend to be LOWER than in b."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    return (sum((x > b).sum() for x in a) - sum((x < b).sum() for x in a)) / (len(a) * len(b))


def mann_whitney(a, b):
    try:
        r = stats.mannwhitneyu(a, b, alternative="two-sided", method="auto")
        return float(r.statistic), float(r.pvalue)
    except ValueError:                                   # identical groups
        return float("nan"), 1.0


def trials_from_names(data_dir, prefix_map, runlog=None):
    rows = []
    for f in sorted(os.listdir(data_dir)):
        for pre, cond in prefix_map.items():
            if f.startswith(pre) and f.endswith(".csv"):
                rows.append({"trial_id": f[:-4], "pair": int(f[len(pre):-4]), "condition": cond})
    t = pd.DataFrame(rows, columns=["trial_id", "pair", "condition"])
    if runlog and os.path.exists(runlog) and len(t):
        t["dropped_iterations"] = t.trial_id.map(pd.read_csv(runlog).set_index("run_id")["dropped_iterations"])
    return t


def trial_metrics(trials, data_dir):
    rows, missing = [], []
    for _, t in trials.iterrows():
        p = os.path.join(data_dir, f"{t.trial_id}.csv")
        if not os.path.exists(p):
            missing.append(t.trial_id); continue
        d = metrics.load_run(p)
        m = metrics.run_metrics(d)
        hm = d["hpa_min"].astype(float)
        raised = d["t_sec"][hm > hm.iloc[0]]
        m["first_boost_s"] = float(raised.iloc[0]) if len(raised) else np.nan            # first time the HPA floor went above its start value
        m["boosts"] = int(((hm > hm.shift(1)) & hm.shift(1).notna()).sum())
        m["condition"], m["pair"], m["trial_id"] = t.condition, t.pair, t.trial_id
        m["dropped_iterations"] = t.get("dropped_iterations", np.nan)
        rows.append(m)
    return pd.DataFrame(rows), missing


def describe(x):
    x = pd.Series(x).dropna()
    if not len(x):
        return {"n": 0, "median": np.nan, "q1": np.nan, "q3": np.nan, "mean": np.nan, "sd": np.nan}
    return {"n": len(x), "median": x.median(), "q1": x.quantile(.25), "q3": x.quantile(.75), "mean": x.mean(),
            "sd": x.std(ddof=1) if len(x) > 1 else np.nan}


def compare(tm, ca, cb):
    """metrics of condition ca compared with cb (Mann-Whitney, Cliff's delta); tests need >= 3 runs per group"""
    out = []
    for col, label in TESTS:
        a, b = tm[tm.condition == ca][col].dropna(), tm[tm.condition == cb][col].dropna()
        row = {"comparison": f"{LABEL[ca]} vs {LABEL[cb]}", "metric": label}
        for name, s in ((cb, b), (ca, a)):
            for k, v in describe(s).items():
                row[f"{name}_{k}"] = v
        ok = len(a) >= 3 and len(b) >= 3
        row["U"], row["p_value"] = mann_whitney(a, b) if ok else (np.nan, np.nan)
        row["cliffs_delta"] = cliffs_delta(a, b) if ok else np.nan
        out.append(row)
    return pd.DataFrame(out)


def timing_check(tm, calm):
    """Protocol v2 'predictive' test for every AEGIS run: floor first raised AFTER the surge began (t >= calm) and >= 30 s before the
    HPA-only runs breached (their median first-breach time)."""
    a, b = tm[tm.condition == AEGIS].copy(), tm[tm.condition == BASE]
    breach_t = b.first_breach_s.median() if b.first_breach_s.notna().any() else np.nan
    a["surge_start_s"] = calm
    a["hpa_only_first_breach_s"] = breach_t
    a["boost_before_surge"] = a.first_boost_s < calm
    a["lead_before_hpa_breach_s"] = breach_t - a.first_boost_s
    a["predictive"] = (a.first_boost_s >= calm) & (a.lead_before_hpa_breach_s >= LEAD_S)
    return a[["trial_id", "first_boost_s", "surge_start_s", "boost_before_surge", "hpa_only_first_breach_s", "lead_before_hpa_breach_s", "predictive"]]


def fig_boxes(tm, conds, title, out):
    fig, axs = plt.subplots(1, 4, figsize=(11, 3.4)); rng = np.random.default_rng(0)
    for ax, (col, label) in zip(axs, TESTS):
        for i, c in enumerate(conds, 1):
            v = tm[tm.condition == c][col].dropna().values
            if len(v):
                ax.boxplot([v], positions=[i], widths=.5, showfliers=False, medianprops={"color": "black"})
                ax.scatter(rng.normal(i, .05, len(v)), v, s=22, color=COLORS[c], zorder=3, alpha=.85)
        ax.set_xticks(range(1, len(conds) + 1)); ax.set_xticklabels([f"{LABEL[c]}\nn={int(tm[tm.condition == c][col].notna().sum())}" for c in conds], fontsize=7)
        ax.set_ylabel(label, fontsize=8); ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(title, fontsize=10); fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig)


def write_markdown(tm, conds, tables, timing, missing, calm, path):
    n = {c: int((tm.condition == c).sum()) for c in conds}
    f = lambda x, d=2: "-" if pd.isna(x) else f"{x:.{d}f}"
    L = ["# " + " vs ".join(f"{LABEL[c]} (n={n[c]})" for c in conds), ""]
    if min(n.values()) < 5:
        pmin = 2 / math.comb(n[conds[0]] + n[conds[1]], n[conds[0]])
        L += [f"**Preliminary: only {min(n.values())} runs in the smallest group.** With {n[conds[0]]} and {n[conds[1]]} runs the smallest attainable "
              f"two-sided Mann-Whitney p-value is {pmin:.3f}; treat the results as descriptive.", ""]
    if missing:
        L += [f"Trials without data file (skipped): {', '.join(missing)}", ""]
    L += ["Breached runs: " + ", ".join(f"{LABEL[c]} {int(tm[tm.condition == c].breached.sum())}/{n[c]}" for c in conds), ""]
    if BASE in conds and AEGIS in conds and n[BASE] and n[AEGIS]:
        a, b = int(tm[tm.condition == AEGIS].breached.sum()), int(tm[tm.condition == BASE].breached.sum())
        L += [f"Fisher exact test, breached runs, {LABEL[AEGIS]} vs {LABEL[BASE]}: p = {stats.fisher_exact([[a, n[AEGIS] - a], [b, n[BASE] - b]])[1]:.3f}", ""]
    L += ["| Comparison | Metric | A: median [Q1-Q3] (mean ± SD) | B: median [Q1-Q3] (mean ± SD) | U | p (two-sided) | Cliff's delta (A vs B) |", "|---|---|---|---|---|---|---|"]
    for t in tables:
        ca, cb = t.attrs["a"], t.attrs["b"]
        for _, r in t.iterrows():
            cell = lambda p: f"{f(r[p + '_median'])} [{f(r[p + '_q1'])}-{f(r[p + '_q3'])}] ({f(r[p + '_mean'])} ± {f(r[p + '_sd'])}), n={int(r[p + '_n'])}"
            L.append(f"| {r.comparison} | {r.metric} | {cell(ca)} | {cell(cb)} | {f(r.U, 1)} | {f(r.p_value, 3)} | {f(r.cliffs_delta)} |")
    if timing is not None and len(timing):
        L += ["", f"## Was AEGIS predictive? (surge starts at t = {calm} s; a boost counts only if it starts at or after the surge start and at least {LEAD_S} s before the HPA-only runs breach)", "",
              "| Run | First floor raise (s) | Before the surge? | HPA-only first breach (s) | Lead before that breach (s) | Predictive |", "|---|---|---|---|---|---|"]
        for _, r in timing.iterrows():
            L.append(f"| {r.trial_id} | {f(r.first_boost_s, 0)} | {'yes' if r.boost_before_surge else 'no'} | {f(r.hpa_only_first_breach_s, 0)} | {f(r.lead_before_hpa_breach_s, 0)} | {'yes' if r.predictive else 'no'} |")
        L += ["", f"**Predictive in {int(timing.predictive.sum())} of {len(timing)} AEGIS runs; floor raised before the surge began in {int(timing.boost_before_surge.sum())} of {len(timing)}.**"]
    aeg = tm[tm.condition == AEGIS]
    if len(aeg):
        L += ["", f"Controller activity: {int(aeg.boosts.sum())} floor raises in total ({aeg.boosts.mean():.1f} per AEGIS run)."]
    L += ["", "Notes: Cliff's delta below zero means the values of A tend to be lower than those of B (lower is better for violation, p95, recovery and cost). No correction for "
          "testing several metrics; the primary metric is SLO violation. Replica-minutes must always be read together with the latency metrics."]
    open(path, "w").write("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme", choices=sorted(SCHEMES), default="v2")
    ap.add_argument("--data"); ap.add_argument("--runlog"); ap.add_argument("--out")
    ap.add_argument("--calm", type=int, help="seconds of calm traffic before the surge starts")
    a = ap.parse_args()
    s = SCHEMES[a.scheme]
    data, runlog, out, calm = a.data or s["data"], a.runlog or s["runlog"], a.out or s["out"], a.calm or s["calm"]
    if not os.path.isdir(data):
        raise SystemExit(f"no data folder {data} yet: the {a.scheme} runs have not been exported (see docs/protocol_{'v2' if a.scheme == 'v2' else 'ec2'}.md)")
    os.makedirs(out, exist_ok=True)
    tm, missing = trial_metrics(trials_from_names(data, s["prefix"], runlog), data)
    if not len(tm):
        raise SystemExit(f"no runs named {list(s['prefix'])} found in {data}")
    conds = [c for c in s["prefix"].values() if (tm.condition == c).any()]
    pairs = [(AEGIS, BASE), (STATIC, BASE), (AEGIS, STATIC)]
    tables = []
    for ca, cb in pairs:
        if ca in conds and cb in conds:
            t = compare(tm, ca, cb); t.attrs.update(a=ca, b=cb); tables.append(t)
    timing = timing_check(tm, calm) if AEGIS in conds and BASE in conds else None
    tm.round(3).to_csv(f"{out}/trials_per_run.csv", index=False)
    if tables:
        pd.concat(tables).round(4).to_csv(f"{out}/comparison_table.csv", index=False)
    if timing is not None:
        timing.round(1).to_csv(f"{out}/boost_timing.csv", index=False)
    write_markdown(tm, conds, tables, timing, missing, calm, f"{out}/comparison.md")
    fig_boxes(tm, conds, f"{' vs '.join(LABEL[c] for c in conds)} ({a.scheme}; each dot = one run)", f"{out}/fig_comparison_boxplots.png")
    print(open(f"{out}/comparison.md").read())


if __name__ == "__main__":
    main()
