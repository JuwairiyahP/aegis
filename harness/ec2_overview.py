# harness/ec2_overview.py - what the EC2 data show so far (protocol: docs/protocol_ec2.md).
#   python harness/ec2_overview.py            -> results/ec2/
# Makes: capacity-probe figure (p95 vs offered rate), the EC2 load-profile figure and a table of the training runs.
import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import metrics

BLUE, ORANGE, RED, GREY = "#2b6cb0", "#ed8936", "#c53030", "#718096"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
BASE, CALM = 4, 180          # k6 base rate on EC2 (req/s) and the calm phase before the ramp (s)


def rising(p):
    """rows up to the highest offered rate: afterwards the 30 s p95 window still holds the overload while the load has stopped"""
    p = p[p.rps.notna()]
    return p.loc[:p.rps.idxmax()]


def fig_probe(p, out):
    p = rising(p); p = p[p.p95.notna() & (p.rps > 0.5)]
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.5), gridspec_kw={"width_ratios": [1.15, 1]})
    d = pd.read_csv(PROBE_CSV)
    ax = axs[0]; ax.plot(d.t_sec / 60, d.rps, color=BLUE); ax.set_xlabel("time (minutes)"); ax.set_ylabel("offered load (req/s)", color=BLUE)
    ax2 = ax.twinx(); ax2.plot(d.t_sec / 60, pd.to_numeric(d.p95, errors="coerce"), color="black", lw=1.2); ax2.set_yscale("log")
    ax2.axhline(metrics.SLO, color=RED, ls="--", lw=1); ax2.text(0.15, metrics.SLO * 1.12, "SLO 0.25 s", color=RED, fontsize=7); ax2.set_ylabel("p95 latency (s, log)"); ax2.spines["top"].set_visible(False)
    ax.set_title("Staircase probe of one pod", fontsize=9)
    ax = axs[1]
    ax.scatter(p.rps, p.p95, s=14, color=BLUE, alpha=.8); ax.set_yscale("log"); ax.axhline(metrics.SLO, color=RED, ls="--", lw=1)
    ax.set_xlabel("offered load (req/s, 30 s rate)"); ax.set_ylabel("p95 latency (s, log)")
    ax.set_title("p95 stays low until the CPU limit (rising phase only)", fontsize=9)
    fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig)


def stages(peak, ramp, hold, calm=CALM, down=60, tail=120, base=BASE, floor=None):
    t, pts = 0, [(0, base)]
    for dur, tgt in ((calm, base), (ramp, peak), (hold, peak), (down, base), (tail, base)):
        t += dur; pts.append((t, tgt))
    return pts


def fig_profiles(out):
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    train = [(p, r, 120) for p in (30, 38, 46, 54) for r in (30, 60, 120)] + [(34, 180, 120), (50, 180, 120), (42, 60, 120)]
    for i, (p, r, h) in enumerate(train):
        xs, ys = zip(*stages(p, r, h)); ax.plot([x / 60 for x in xs], ys, color=BLUE, alpha=.25, lw=1, label="training profiles" if i == 0 else None)
    xs, ys = zip(*stages(42, 45, 120)); ax.plot([x / 60 for x in xs], ys, color=ORANGE, lw=2.5, label="held-out profile (peak 42, ramp 45 s, hold 120 s)")
    ax.axhline(24, color=GREY, ls=":", lw=1); ax.text(0.1, 24.8, "one pod: p95 ~0.10 s at 24 req/s", fontsize=7, color=GREY)
    ax.axhline(27, color=RED, ls="--", lw=1); ax.text(0.1, 27.8, "one pod: breach at ~26-28 req/s", fontsize=7, color=RED)
    ax.set_xlabel("time since start of run (minutes)"); ax.set_ylabel("offered load (req/s)"); ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig)


def planned_requests(peak, ramp, hold, calm=CALM, down=60, tail=120, base=BASE):
    """requests k6 should send for one run (area under the load profile)"""
    return base * calm + (base + peak) / 2 * ramp + peak * hold + (base + peak) / 2 * down + base * tail


def main():
    global PROBE_CSV
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/ec2")
    ap.add_argument("--probe", default="data/probe_ec2/probe-ec2.csv")
    ap.add_argument("--runlog", default="harness/runs_ec2.csv")
    ap.add_argument("--out", default="results/ec2")
    a = ap.parse_args()
    PROBE_CSV = a.probe
    os.makedirs(a.out, exist_ok=True)
    probe = metrics.load_run(a.probe)
    fig_probe(probe, f"{a.out}/fig_ec2_probe.png")
    fig_profiles(f"{a.out}/fig_ec2_profiles.png")
    up = rising(probe); ok = up[up.p95 <= metrics.SLO].rps.max(); first_bad = up[up.p95 > metrics.SLO].rps.min()
    pd.DataFrame([{"within_slo_up_to_rps": round(ok, 1), "first_breach_rps": round(first_bad, 1)}]).to_csv(f"{a.out}/ec2_probe_summary.csv", index=False)
    print(f"probe: p95 within the SLO up to {ok:.1f} req/s; first breach seen at {first_bad:.1f} req/s (30 s rate)")

    log = pd.read_csv(a.runlog).set_index("run_id")
    rows = []
    for f in sorted(glob.glob(f"{a.data}/*.csv")):
        d = metrics.load_run(f); m = metrics.run_metrics(d); rid = m["run_id"]
        hm = d["hpa_min"].astype(float)
        g = metrics.run_metrics(d, fill_gaps=True)
        p = pd.to_numeric(d["p95"], errors="coerce")
        na = p.isna().values.copy(); na[0] = False                      # the first sample (before any traffic) is not a gap
        run = best = 0
        for v in na:
            run = run + 1 if v else 0; best = max(best, run)
        fl = (p.ffill(limit=6).fillna(0) > metrics.SLO).values          # the predictor's feature code: carry p95 forward for up to 30 s
        m.update({"longest_gap_samples": best, "n_onsets_ffill30": int(sum(fl[i] and (i == 0 or not fl[i - 1]) for i in range(len(fl))))})
        m.update({"slo_violation_pct_gapfilled": g["slo_violation_pct"], "recovery_s_gapfilled": g["recovery_s"], "gap_samples": int((d.p95.isna() & (d.t_sec > 0)).sum()), "n_onsets_gapfilled": g["n_breach_onsets"]})
        m.update({"scenario": d.scenario.iloc[0], "peak": d.peak.iloc[0], "ramp": d.ramp.iloc[0], "hold": d.hold.iloc[0],
                  "floor": int(hm.iloc[:36].max()), "dropped_iterations": int(log.loc[rid, "dropped_iterations"]) if rid in log.index else np.nan,
                  "telemetry_gap_rows": int(d.rps.isna().sum())})
        m["planned_requests"] = round(planned_requests(float(d.peak.iloc[0]), float(d.ramp.iloc[0]), float(d.hold.iloc[0])))
        m["dropped_pct"] = 100 * m["dropped_iterations"] / m["planned_requests"]
        rows.append(m)
    t = pd.DataFrame(rows)
    t.round(3).to_csv(f"{a.out}/ec2_runs_summary.csv", index=False)
    show = t[["run_id", "peak", "ramp", "floor", "breached", "n_breach_onsets", "slo_violation_pct", "peak_p95_s", "max_replicas", "replica_minutes", "recovery_s", "dropped_iterations", "dropped_pct"]]
    print(show.round(2).to_string(index=False))
    tr = t[t.run_id.str.startswith(("et-", "ef-"))]
    print(f"\nEC2 training runs: {len(tr)}; breached: {int(tr.breached.sum())}; breach onsets: {int(tr.n_breach_onsets.sum())}; "
          f"runs with dropped k6 iterations: {int((tr.dropped_iterations > 0).sum())}; max pods seen: {int(tr.max_replicas.max())}")


if __name__ == "__main__":
    main()
