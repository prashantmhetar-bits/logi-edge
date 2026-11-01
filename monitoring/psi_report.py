#!/usr/bin/env python3
import argparse
import csv
import json
import os
import sys
from datetime import datetime

ALERT, RECOVERED, ALERT_DEADLINE_S = 0.25, 0.10, 300
SURFACE, INK, INK_MUTED, RULE = "#fcfcfb", "#0b0b0b", "#52514e", "#d9d8d3"
LINE = "#2a78d6"
LABEL_BOX = {
    "facecolor": SURFACE,
    "edgecolor": RULE,
    "boxstyle": "round,pad=0.3",
    "linewidth": 0.6,
}
PHASE_FILL = {"clean": "#f1f1ee", "drift": "#e6e5df", "recovery": "#f1f1ee"}


def ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def load(log_path, events_path):
    with open(log_path, newline="") as f:
        rows = [
            (ts(r["utc"]), float(r["psi"]), int(r["window_n"]))
            for r in csv.DictReader(f)
        ]
    with open(events_path, newline="") as f:
        events = {r["event"]: ts(r["utc"]) for r in csv.DictReader(f)}
    return rows, events


def analyse(rows, events):
    t_inj, t_rec = events["inject_combined"], events["restore_clean"]
    clean = [p for t, p, _ in rows if t < t_inj]
    drift = [(t, p) for t, p, _ in rows if t_inj <= t < t_rec]
    recovery = [(t, p) for t, p, _ in rows if t >= t_rec]
    first_alert = next(((t, p) for t, p in drift if p > ALERT), None)
    first_ok = next(((t, p) for t, p in recovery if p < RECOVERED), None)
    r = {
        "thresholds": {
            "alert": ALERT,
            "recovered": RECOVERED,
            "alert_deadline_s": ALERT_DEADLINE_S,
        },
        "before": {
            "psi_last": clean[-1] if clean else None,
            "psi_max": max(clean) if clean else None,
            "reports": len(clean),
        },
        "during": {
            "psi_max": max((p for _, p in drift), default=None),
            "reports": len(drift),
            "first_alert_psi": first_alert[1] if first_alert else None,
            "seconds_to_alert": round(first_alert[0] - t_inj) if first_alert else None,
        },
        "after": {
            "psi_final": recovery[-1][1] if recovery else None,
            "reports": len(recovery),
            "first_below_recovery_psi": first_ok[1] if first_ok else None,
            "seconds_to_recovery": round(first_ok[0] - t_rec) if first_ok else None,
        },
    }
    r["passed_alert_within_5_min"] = bool(
        first_alert and first_alert[0] - t_inj <= ALERT_DEADLINE_S
    )
    r["passed_recovery"] = bool(first_ok)
    return r


def draw(rows, events, r, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t0 = events.get("monitor_start", rows[0][0])
    mins = [(t - t0) / 60 for t, _, _ in rows]
    psi = [p for _, p, _ in rows]
    t_inj, t_rec = (events["inject_combined"] - t0) / 60, (
        events["restore_clean"] - t0
    ) / 60
    t_end = max(mins[-1], (events.get("end", rows[-1][0]) - t0) / 60)

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.edgecolor": RULE,
            "axes.labelcolor": INK_MUTED,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
        }
    )
    fig, ax = plt.subplots(figsize=(11, 4.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    top = max(max(psi) * 1.3, 0.5)
    for a, b, name, label, row in [
        (0, t_inj, "clean", "Clean (--anomaly none)", 0),
        (t_inj, t_rec, "drift", "Drift injected (--anomaly combined)", 1),
        (t_rec, t_end, "recovery", "Clean data restored", 0),
    ]:
        ax.axvspan(a, b, color=PHASE_FILL[name], zorder=0, linewidth=0)
        ax.text(
            a + t_end * 0.006,
            top * (0.97 - 0.07 * row),
            label,
            ha="left",
            va="top",
            fontsize=8.5,
            color=INK_MUTED,
        )
    for y, label in [
        (ALERT, f"Alert threshold PSI > {ALERT}"),
        (RECOVERED, f"Recovered PSI < {RECOVERED}"),
    ]:
        ax.axhline(y, color=INK_MUTED, linewidth=1, linestyle=":", zorder=1)
        ax.text(
            t_end,
            y,
            f" {label}",
            va="center",
            ha="left",
            fontsize=8,
            color=INK_MUTED,
            clip_on=False,
        )
    ax.plot(mins, psi, color=LINE, linewidth=2, marker="o", markersize=4, zorder=3)

    d, a = r["during"], r["after"]
    if d["seconds_to_alert"] is not None:
        x = t_inj + d["seconds_to_alert"] / 60
        ax.annotate(
            f"[LOGIBRIDGE DRIFT ALERT] PSI={d['first_alert_psi']:.3f}\n"
            f"{d['seconds_to_alert']} s after injection",
            (x, d["first_alert_psi"]),
            xytext=(12, -6),
            textcoords="offset points",
            fontsize=8,
            color=INK,
            va="top",
            bbox=LABEL_BOX,
        )
    if a["seconds_to_recovery"] is not None:
        x = t_rec + a["seconds_to_recovery"] / 60
        ax.annotate(
            f"PSI={a['first_below_recovery_psi']:.3f} < {RECOVERED}\n"
            f"{a['seconds_to_recovery'] / 60:.1f} min after restore",
            (x, a["first_below_recovery_psi"]),
            xytext=(-10, 40),
            textcoords="offset points",
            ha="right",
            fontsize=8,
            color=INK,
            bbox=LABEL_BOX,
        )

    ax.set_xlim(0, t_end)
    ax.set_ylim(0, top)
    ax.set_xlabel(
        "Minutes since monitoring started (PSI on the last 100 inferences, every 60 s)"
    )
    ax.set_ylabel("PSI vs reference_dist.json")
    ax.set_title(
        "E1 PSI drift monitoring on P(Normal): injection and recovery",
        loc="left",
        color=INK,
        fontsize=11,
        fontweight="bold",
    )
    ax.grid(True, axis="y", color=RULE, linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def markdown(r):
    b, d, a = r["before"], r["during"], r["after"]
    f = lambda v: "—" if v is None else f"{v:.3f}"
    return "\n".join(
        [
            "### E1 PSI drift monitoring: before / during / after injection",
            "",
            "| Phase | PSI | Detail |",
            "|---|---:|---|",
            f"| Before (clean) | {f(b['psi_last'])} | last report; max {f(b['psi_max'])} over {b['reports']} reports |",
            f"| During (`--anomaly combined`) | {f(d['psi_max'])} (max) | first alert PSI {f(d['first_alert_psi'])} "
            f"after **{d['seconds_to_alert']} s** (limit {ALERT_DEADLINE_S} s): "
            f"{'✅' if r['passed_alert_within_5_min'] else '❌'} |",
            f"| After (clean restored) | {f(a['psi_final'])} (final) | below {RECOVERED} after "
            f"**{'—' if a['seconds_to_recovery'] is None else round(a['seconds_to_recovery'] / 60, 1)} min**: "
            f"{'✅' if r['passed_recovery'] else '❌'} |",
        ]
    )


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="E1 PSI drift-injection report")
    ap.add_argument("--log", required=True)
    ap.add_argument("--events", required=True)
    ap.add_argument("--json", required=True)
    ap.add_argument("--png")
    args = ap.parse_args()

    rows, events = load(args.log, args.events)
    if not rows:
        print("::error::no PSI values in the monitor log", file=sys.stderr)
        return 1
    r = analyse(rows, events)
    with open(args.json, "w") as fh:
        json.dump(r, fh, indent=2)
    if args.png:
        draw(rows, events, r, args.png)
    md = markdown(r)
    print(md)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as fh:
            fh.write(md + "\n\n")
    return 0 if r["passed_alert_within_5_min"] and r["passed_recovery"] else 1


if __name__ == "__main__":
    sys.exit(main())
