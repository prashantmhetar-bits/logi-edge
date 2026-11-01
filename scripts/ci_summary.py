#!/usr/bin/env python3
import glob
import json
import os
import re
import sys
from pathlib import Path

FEATURES = [
    "temp mean (°C)",
    "temp std (°C)",
    "temp rate (°C/min)",
    "vib RMS (g)",
    "vib peak (g)",
    "vib kurtosis",
]


def emit(md):
    print(md)
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if path:
        with open(path, "a", encoding="utf-8") as f:
            f.write(md + "\n\n")


def stats(path):
    import numpy as np

    s = np.load(path, allow_pickle=True).item()
    rows = [
        "### C2 normalisation statistics (`training_stats.npy`, loaded at runtime, never recomputed)",
        "",
        "| Feature | Mean | Std |",
        "|---|---:|---:|",
    ]
    rows += [
        f"| {n} | {m:.4f} | {d:.4f} |" for n, m, d in zip(FEATURES, s["mean"], s["std"])
    ]
    emit("\n".join(rows))


def metrics(*paths):
    rows = [
        "### Model quality gates (validation split)",
        "",
        "| Variant | Accuracy | Recall N / W / C | Critical recall | Gates (>88 % acc, >95 % C recall) | Notes |",
        "|---|---:|---|---:|---|---|",
    ]
    for p in paths:
        m = json.loads(Path(p).read_text())
        rec = " / ".join(f"{v * 100:.1f} %" for v in m["recall_per_class"].values())
        notes = []
        if "params" in m:
            notes.append(f"{m['params']} params")
        if "n_calibration" in m:
            notes.append(f"{m['n_calibration']} calibration samples")
        if "fp32_int8_agreement" in m:
            notes.append(f"FP32↔INT8 agreement {m['fp32_int8_agreement'] * 100:.1f} %")
        if "structure" in m:
            st = m["structure"]
            notes.append(
                f"hidden {st['hidden_units_before']}→{st['hidden_units_after']}, "
                f"{st['params_before']}→{st['params_after']} params"
            )
        if "file_size_kb" in m:
            notes.append(f"{m['file_size_kb']} KB")
        rows.append(
            f"| {m['variant']} | {m['val_accuracy'] * 100:.2f} % | {rec} | "
            f"{m['critical_recall'] * 100:.1f} % | {'✅ PASS' if m['passed'] else '❌ FAIL'} | "
            f"{'; '.join(notes)} |"
        )
        cm = m["confusion_matrix"]
        rows_cm = " · ".join(f"{lbl}: {r}" for lbl, r in zip("NWC", cm))
        rows.append(f"| | confusion (rows = true) | {rows_cm} | | | |")
    emit("\n".join(rows))


def benchmark(path):
    d = json.loads(Path(path).read_text())
    env = d["environment"]
    rows = [
        "### F2 five-metric benchmark (this CI runner)",
        "",
        f"{env['platform']} · {env['logical_cpus']} logical CPUs · Python {env['python']} · "
        f"TF {env['tensorflow']} · {d['methodology']['warmup_runs_excluded']} warm-up + "
        f"{d['methodology']['timed_runs']} timed runs · E = {d['methodology']['tdp_w']} W × mean latency. "
        "The report uses the laptop run committed in `optimisation/results/`; this run shows the "
        "method reproduces on other hardware.",
        "",
        "| Variant | Mean latency (ms) | p95 latency (ms) | File size (KB) | Accuracy (%) | Energy (mJ) | Critical recall |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, r in d["results"].items():
        rows.append(
            f"| {name} | {r['mean_latency_ms']:.4f} | {r['p95_latency_ms']:.4f} | {r['file_size_kb']:.2f} | "
            f"{r['val_accuracy'] * 100:.2f} | {r['energy_mj']:.5f} | {r['critical_detected']} |"
        )
    emit("\n".join(rows))


def payloads(root):
    rows = [
        "### D2 integration: one image, three models via `MODEL_PATH`",
        "",
        "| MODEL_PATH | Payloads | Status (model + rule) | Model class | P(Normal) | Temp mean (°C) | Alerts logged |",
        "|---|---:|---|---|---|---|---:|",
    ]
    for f in sorted(glob.glob(os.path.join(root, "it-*", "payloads.jsonl"))):
        ps = [json.loads(line) for line in open(f) if line.strip()]
        if not ps:
            continue
        alerts = os.path.join(os.path.dirname(f), "alerts.jsonl")
        n_alerts = (
            sum(1 for line in open(alerts) if line.strip())
            if os.path.exists(alerts)
            else 0
        )
        rows.append(
            f"| `{ps[0]['model']}` | {len(ps)} | {', '.join(sorted({p['status'] for p in ps}))} | "
            f"{', '.join(sorted({p['model_status'] for p in ps}))} | "
            f"{min(p['probabilities'][0] for p in ps):.3f}–{max(p['probabilities'][0] for p in ps):.3f} | "
            f"{min(p['temp_mean_c'] for p in ps):.2f}–{max(p['temp_mean_c'] for p in ps):.2f} | {n_alerts} |"
        )
    rows.append("")
    rows.append(
        "Topic: `logibridge/trucks/{truck_id}/inference`. The simulator runs `--anomaly combined`, "
        "so every variant must report CRITICAL."
    )
    emit("\n".join(rows))


def ansible(log1, log2):
    def recap(p):
        lines = [line.rstrip() for line in open(p) if re.match(r"^\S+\s+:\s+ok=", line)]
        return lines[-1] if lines else "(no PLAY RECAP)"

    r1, r2 = recap(log1), recap(log2)
    ok = re.search(r"changed=0\s", r2 + " ") and re.search(r"failed=0", r2)
    emit(
        "\n".join(
            [
                "### E2 Ansible deployment: idempotency",
                "",
                "| Run | PLAY RECAP |",
                "|---|---|",
                f"| 1 (deploys) | `{r1}` |",
                f"| 2 (no changes) | `{r2}` |",
                "",
                f"Second run reports **changed=0**: {'✅ idempotent' if ok else '❌ NOT idempotent'}",
            ]
        )
    )


def playbook(path):
    import yaml

    plays = yaml.safe_load(Path(path).read_text())
    tasks = [t for play in plays for t in play.get("tasks", [])]
    rows = [
        "### E2 playbook structure (`logibridge_deploy.yml`)",
        "",
        "| # | Task | Module |",
        "|---:|---|---|",
    ]
    for i, t in enumerate(tasks, 1):
        module = next((k for k in t if "." in k), "?")
        rows.append(f"| {i} | {t.get('name', '')} | `{module}` |")
    ok = len(tasks) == 7
    rows += [
        "",
        f"Task count: **{len(tasks)}** (brief requires exactly 7): {'✅' if ok else '❌'}",
    ]
    emit("\n".join(rows))
    return 0 if ok else 1


def dockerfile(path):
    lines = [line.strip() for line in Path(path).read_text().splitlines()]
    instr = [line for line in lines if line and not line.startswith("#")]

    def idx(pred):
        return next((i for i, line in enumerate(instr) if pred(line)), None)

    from_line = idx(lambda line: line.upper().startswith("FROM "))
    pip_line = idx(
        lambda line: line.upper().startswith("RUN ") and "pip install" in line
    )
    model_line = idx(
        lambda line: line.upper().startswith("COPY ") and ".tflite" in line
    )
    fs_steps = [
        i
        for i, line in enumerate(instr)
        if line.split()[0].upper() in ("RUN", "COPY", "ADD", "WORKDIR")
    ]
    checks = [
        (
            "Base image is `python:3.11-slim`",
            from_line is not None and instr[from_line].split()[1] == "python:3.11-slim",
        ),
        (
            "All `pip install` layers come before the model COPY",
            pip_line is not None
            and model_line is not None
            and all(
                i < model_line for i, line in enumerate(instr) if "pip install" in line
            ),
        ),
        (
            "Model COPY is the last filesystem layer (OTA rebuilds only it)",
            model_line is not None and fs_steps[-1] == model_line,
        ),
        (
            "`MODEL_PATH` is an environment variable (switch variants without rebuild)",
            any(
                line.upper().startswith("ENV") and "MODEL_PATH=" in line
                for line in instr
            ),
        ),
        (
            "Runs as a non-root user",
            any(
                line.upper().startswith("USER ")
                and line.split()[1] not in ("root", "0")
                for line in instr
            ),
        ),
    ]
    rows = ["### D2 Dockerfile compliance", "", "| Requirement | Result |", "|---|---|"]
    rows += [f"| {name} | {'✅' if ok else '❌'} |" for name, ok in checks]
    emit("\n".join(rows))
    return 0 if all(ok for _, ok in checks) else 1


def deliverables():
    items = [
        (
            "monitoring/reference_dist.json",
            "E1 reference distribution (300 clean inferences) - commit it from the live-demos run",
        ),
        ("demo/demo_video_link.txt", "Demo video link (submission requirement)"),
        ("reports/final_report.pdf", "Final report PDF"),
    ]
    rows = ["### Submission deliverables", "", "| File | Status |", "|---|---|"]
    for path, what in items:
        p = Path(path)
        present = p.is_file() and p.stat().st_size > 0
        if present and path.endswith(".txt") and "PASTE" in p.read_text():
            present = False
        rows.append(
            f"| `{path}` | {'✅ present' if present else '⚠️ missing'} — {what} |"
        )
        if not present:
            print(f"::warning file={path}::{what} is missing")
    emit("\n".join(rows))
    return 0


def main(argv):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if len(argv) < 2:
        print(
            "usage: ci_summary.py {stats|metrics|benchmark|payloads|ansible|playbook|dockerfile|deliverables} [args]"
        )
        return 2
    cmd, args = argv[1], argv[2:]
    fn = {
        "stats": stats,
        "metrics": metrics,
        "benchmark": benchmark,
        "payloads": payloads,
        "ansible": ansible,
        "playbook": playbook,
        "dockerfile": dockerfile,
        "deliverables": deliverables,
    }[cmd]
    return fn(*args) or 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
