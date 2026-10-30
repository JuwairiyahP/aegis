# harness/prescaling_pairs.py - what does having capacity ready EARLY buy? (NOT an AEGIS result)
# A's "fl-*" runs raise the HPA floor by hand before the load starts. Each one is paired with a plain-HPA training run
# that has the same peak and ramp. Pairs differ in recording time, so this is descriptive evidence of the idea only.
#   python harness/prescaling_pairs.py            -> results/prescaling_pairs.csv + fig_prescaling_pairs.png
import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/pilot_laptop")
    ap.add_argument("--out", default="results/pilot_laptop")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    runs = {os.path.basename(f)[:-4]: metrics.load_run(f) for f in sorted(glob.glob(f"{a.data}/*.csv"))}
    m = pd.DataFrame([metrics.run_metrics(d) for d in runs.values()]).set_index("run_id")
    meta = pd.concat([d.iloc[[0]] for d in runs.values()]).set_index("run_id")[["peak", "ramp", "hold"]]
    m = m.join(meta)
    m["floor"] = [int(d["hpa_min"].iloc[:60].max()) for d in runs.values()]          # floor during the first 5 minutes
    rows = []
    for fid in [r for r in m.index if r.startswith("fl-")]:
        f = m.loc[fid]
        base = m[(m.index.str.startswith("tr-")) & (m.peak == f.peak) & (m.ramp == f.ramp) & (m.floor == 1)]
        if base.empty:
            continue
        b = base.iloc[0]
        rows.append({"profile": f"peak {int(f.peak)}, ramp {int(f.ramp)} s", "floor_pods": int(f.floor), "hpa_run": base.index[0], "floor_run": fid,
                     "hpa_violation_pct": b.slo_violation_pct, "floor_violation_pct": f.slo_violation_pct,
                     "hpa_peak_p95": b.peak_p95_s, "floor_peak_p95": f.peak_p95_s,
                     "hpa_replica_min": b.replica_minutes, "floor_replica_min": f.replica_minutes})
    t = pd.DataFrame(rows)
    t.round(2).to_csv(f"{a.out}/prescaling_pairs.csv", index=False)
    print(t.round(2).to_string(index=False))

    fig, axs = plt.subplots(1, 3, figsize=(10, 3.4))
    for ax, (hc, fc, lab, log) in zip(axs, [("hpa_violation_pct", "floor_violation_pct", "SLO violation (% of time)", False),
                                           ("hpa_peak_p95", "floor_peak_p95", "peak p95 latency (s, log)", True),
                                           ("hpa_replica_min", "floor_replica_min", "replica-minutes (cost)", False)]):
        for _, r in t.iterrows():
            ax.plot([0, 1], [r[hc], r[fc]], color="#a0aec0", lw=1, zorder=1)
        ax.scatter([0] * len(t), t[hc], color="#718096", s=30, zorder=2); ax.scatter([1] * len(t), t[fc], color="#2b6cb0", s=30, zorder=2)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["plain HPA", "floor raised\nbefore load"]); ax.set_ylabel(lab, fontsize=8)
        if log:
            ax.set_yscale("log"); ax.axhline(metrics.SLO, color="#c53030", ls="--", lw=1)
        ax.set_xlim(-.3, 1.3); ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Manual pre-scaling vs plain HPA on matched profiles (6 pairs; each line = one pair). NOT an AEGIS result.", fontsize=9)
    fig.tight_layout(); fig.savefig(f"{a.out}/fig_prescaling_pairs.png", dpi=200); plt.close(fig)


if __name__ == "__main__":
    main()
