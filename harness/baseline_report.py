# harness/baseline_report.py - HPA-only baseline results for the report (eval runs ev-*.csv).
#   python harness/baseline_report.py                      # reads data/runs/, writes results/
#   python harness/baseline_report.py --data data/runs --out results --example ev-001 --example ev-002
import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import metrics

CALM_S = 180          # every run starts with 3 minutes of calm traffic; breaches before the ramp are not load-driven
BLUE, ORANGE, RED, GREY = "#2b6cb0", "#ed8936", "#c53030", "#718096"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def box_with_points(ax, values, label, color=BLUE, log=False):
    v = pd.Series(values).dropna().values
    ax.boxplot([v], widths=0.5, showfliers=False, medianprops={"color": "black"})
    ax.scatter(np.random.default_rng(0).normal(1, 0.05, len(v)), v, s=22, color=color, zorder=3, alpha=.85)
    ax.set_xticks([1]); ax.set_xticklabels([f"{label}\nn={len(v)}"])
    if log:
        ax.set_yscale("log")


def fig_boxplots(table, out):
    fig, axs = plt.subplots(1, 4, figsize=(10, 3.2))
    box_with_points(axs[0], table.slo_violation_pct, "all runs"); axs[0].set_ylabel("SLO violation (% of time)")
    box_with_points(axs[1], table.peak_p95_s, "all runs", log=True); axs[1].set_ylabel("peak p95 latency (s, log scale)")
    axs[1].axhline(metrics.SLO, color=RED, ls="--", lw=1); axs[1].text(1.3, metrics.SLO * 1.08, "SLO 0.25 s", color=RED, fontsize=7)
    box_with_points(axs[2], table.recovery_s, "breached runs only", color=ORANGE); axs[2].set_ylabel("recovery time (s)")
    box_with_points(axs[3], table.replica_minutes, "all runs"); axs[3].set_ylabel("replica-minutes (cost)")
    fig.suptitle("HPA-only baseline runs (each dot = one run)", fontsize=10)
    fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig)


def fig_timeseries(d, out):
    m = metrics.run_metrics(d)
    t = d.t_sec / 60
    fig, axs = plt.subplots(3, 1, figsize=(7.5, 6), sharex=True, gridspec_kw={"height_ratios": [1, 1.4, 1]})
    axs[0].plot(t, d.rps, color=BLUE); axs[0].set_ylabel("offered load\n(requests/s)")
    axs[1].plot(t, d.p95, color="black", lw=1.4); axs[1].axhline(metrics.SLO, color=RED, ls="--", lw=1)
    axs[1].fill_between(t, 0, d.p95.max() * 1.05, where=metrics.breach_flags(d), color=RED, alpha=.15, label="SLO breach")
    axs[1].set_ylabel("p95 latency (s)"); axs[1].legend(loc="upper left", fontsize=7)
    axs[1].text(t.iloc[-1], metrics.SLO, " SLO 0.25 s", color=RED, va="bottom", ha="right", fontsize=7)
    axs[2].step(t, d.replicas, where="post", color=ORANGE, lw=2, label="ready pods")
    axs[2].step(t, d.hpa_min, where="post", color=GREY, lw=1, ls=":", label="HPA minReplicas")
    axs[2].set_ylabel("pods"); axs[2].set_yticks(range(0, int(d.replicas.max()) + 2)); axs[2].legend(loc="upper left", fontsize=7)
    axs[2].set_xlabel("time since start of run (minutes)")
    note = f"{d.run_id.iloc[0]}: peak p95 {m['peak_p95_s']:.2f} s, SLO violation {m['slo_violation_pct']:.1f}%, max {int(m['max_replicas'])} pods"
    fig.suptitle(note, fontsize=10); fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig)


def stages(peak, ramp, hold, bursts=1, calm=180, down=60, gap=150, tail=120, base=2):
    pts, t = [(0, base)], 0
    def go(dur, target):
        nonlocal t
        t += dur; pts.append((t, target))
    go(calm, base)
    for b in range(bursts):
        if b: go(gap, base)
        go(ramp, peak); go(hold, peak); go(down, base)
    go(tail, base)
    return pts


def fig_load_profiles(out, loadtest="loadtest"):
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    for name in ("profiles.csv", "profiles_edge.csv"):
        p = os.path.join(loadtest, name)
        if os.path.exists(p):
            for _, r in pd.read_csv(p).iterrows():
                xs, ys = zip(*stages(r.peak, r.ramp, r.hold, r.bursts))
                ax.plot([x / 60 for x in xs], ys, color=BLUE, alpha=.25, lw=1,
                        label="training profiles (18 runs)" if (name == "profiles.csv" and r.run_id.endswith("001")) else None)
    p = os.path.join(loadtest, "eval.csv")
    if os.path.exists(p):
        r = pd.read_csv(p).iloc[0]
        xs, ys = zip(*stages(r.peak, r.ramp, r.hold, r.bursts))
        ax.plot([x / 60 for x in xs], ys, color=ORANGE, lw=2.5, label=f"held-out evaluation profile (peak {int(r.peak)}, ramp {int(r.ramp)} s, hold {int(r.hold)} s)")
    ax.axhline(6, color=GREY, ls=":", lw=1); ax.text(0.1, 6.2, "one pod: comfortable limit ~6 req/s", fontsize=7, color=GREY)
    ax.axhline(8, color=RED, ls="--", lw=1); ax.text(0.1, 8.2, "one pod: latency breaks at ~8 req/s", fontsize=7, color=RED)
    ax.set_xlabel("time since start of run (minutes)"); ax.set_ylabel("offered load (requests/s)")
    ax.legend(fontsize=7, loc="upper left"); fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/pilot_laptop")
    ap.add_argument("--out", default="results/pilot_laptop")
    ap.add_argument("--prefix", default="ev-", help="run ids of the HPA-only runs to summarise (ev- = laptop pilot, xb- = EC2 held-out)")
    ap.add_argument("--train-prefix", default="tr-", help="run ids used for the pooled HPA reaction time (tr- laptop, et- EC2)")
    ap.add_argument("--runlog", default="harness/runs_pilot_laptop.csv")
    ap.add_argument("--calm", type=int, default=180)
    ap.add_argument("--no-profiles", action="store_true", help="skip the laptop load-profile figure")
    ap.add_argument("--example", action="append", default=None, help="run ids for the time-series figures (default ev-001)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    global CALM_S
    CALM_S = a.calm

    ev = {os.path.basename(f)[:-4]: metrics.load_run(f) for f in sorted(glob.glob(f"{a.data}/{a.prefix}*.csv"))}
    if not ev:
        raise SystemExit(f"no runs named {a.prefix}* in {a.data}")
    table = pd.DataFrame([metrics.run_metrics(d) for d in ev.values()])
    if os.path.exists(a.runlog):
        log = pd.read_csv(a.runlog).set_index("run_id")
        table["dropped_iterations"] = table.run_id.map(log["dropped_iterations"])
    gaps = {k: int(d.rps.isna().sum()) for k, d in ev.items()}
    table["telemetry_gap_rows"] = table.run_id.map(gaps)
    table["data_quality_flag"] = (table.get("dropped_iterations", 0).fillna(0) > 0) | (table.telemetry_gap_rows > 0)
    table.round(3).to_csv(f"{a.out}/baseline_per_run.csv", index=False)

    rows = []
    for col, label in [("slo_violation_pct", "SLO violation (% of time)"), ("peak_p95_s", "Peak p95 latency (s)"),
                       ("recovery_s", "Recovery time (s), breached runs only"), ("replica_minutes", "Replica-minutes"),
                       ("max_replicas", "Max ready pods")]:
        s = metrics.summarize(table[col]); s["metric"] = label; rows.append(s)
    events = []
    for grp, pattern in (("eval", f"{a.data}/{a.prefix}*.csv"), ("training", f"{a.data}/{a.train_prefix}*.csv")):
        for f in sorted(glob.glob(pattern)):
            d = metrics.load_run(f); v = metrics.breach_flags(d)
            onsets = [t for t in d["t_sec"][v & ~v.shift(1, fill_value=False)] if d["hpa_min"][d["t_sec"] == t].iloc[0] <= 1 and t >= CALM_S]   # skip breaches before the ramp (start-up artefacts)
            for t in onsets:
                base = d["replicas"][d["t_sec"] == t].iloc[0]
                later = d[(d["t_sec"] >= t) & (d["replicas"] > base)]
                events.append({"group": grp, "run_id": d["run_id"].iloc[0], "breach_start_s": t,
                               "reaction_s": later["t_sec"].iloc[0] - t if len(later) else float("nan")})
    ev_df = pd.DataFrame(events); ev_df.to_csv(f"{a.out}/baseline_reaction_events.csv", index=False)
    for label, sub in (("HPA reaction (s), evaluation runs", ev_df[ev_df.group == "eval"]),
                       ("HPA reaction (s), all plain-HPA runs (evaluation + training)", ev_df)):
        s = metrics.summarize(sub.reaction_s); s["metric"] = label; s["events"] = len(sub); rows.append(s)
    summ = pd.DataFrame(rows)[["metric", "n", "mean", "sd", "median", "q1", "q3", "min", "max"]]
    summ.round(2).to_csv(f"{a.out}/baseline_summary.csv", index=False)

    clean = table[~table.data_quality_flag]
    rows2 = []
    for col, label in [("slo_violation_pct", "SLO violation (% of time)"), ("peak_p95_s", "Peak p95 latency (s)"),
                       ("recovery_s", "Recovery time (s), breached runs only"), ("replica_minutes", "Replica-minutes")]:
        s2 = metrics.summarize(clean[col]); s2["metric"] = label; rows2.append(s2)
    pd.DataFrame(rows2)[["metric", "n", "mean", "sd", "median", "q1", "q3", "min", "max"]].round(2).to_csv(
        f"{a.out}/baseline_summary_excluding_flagged.csv", index=False)
    print(f"runs flagged for data quality (dropped k6 iterations or telemetry gap): {list(table[table.data_quality_flag].run_id)}")
    if not a.no_profiles:
        fig_load_profiles(f"{a.out}/fig_load_profiles.png")
    fig_boxplots(table, f"{a.out}/fig_baseline_boxplots.png")
    for rid in (a.example or [sorted(ev)[0]]):
        if rid in ev:
            fig_timeseries(ev[rid], f"{a.out}/fig_timeseries_{rid}.png")

    print(table.round(2).to_string(index=False)); print(); print(summ.round(2).to_string(index=False))
    print(f"\nbreached runs: {int(table.breached.sum())} of {len(table)}")


if __name__ == "__main__":
    main()
