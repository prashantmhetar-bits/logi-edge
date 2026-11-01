#!/usr/bin/env python3
import argparse
import json
import os
import re
import sys
import urllib.request

PRICE_PER_MB = 0.10
FLEETS = {"pilot": 85, "full": 265}
UPDATES_PER_YEAR = 365 / 42
MANIFEST_TYPES = ", ".join(
    [
        "application/vnd.oci.image.index.v1+json",
        "application/vnd.docker.distribution.manifest.list.v2+json",
        "application/vnd.oci.image.manifest.v1+json",
        "application/vnd.docker.distribution.manifest.v2+json",
    ]
)


def fetch_json(url):
    req = urllib.request.Request(url, headers={"Accept": MANIFEST_TYPES})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def image_manifest(registry, repo, ref, platform=("linux", "amd64")):
    doc = fetch_json(f"{registry}/v2/{repo}/manifests/{ref}")
    if "manifests" in doc:
        for m in doc["manifests"]:
            p = m.get("platform", {})
            if (p.get("os"), p.get("architecture")) == platform:
                return image_manifest(registry, repo, m["digest"], platform)
        raise ValueError(f"{repo}:{ref} has no {platform[0]}/{platform[1]} manifest")
    return doc


def blobs(manifest):
    layers = [(layer["digest"], int(layer["size"])) for layer in manifest["layers"]]
    config = (manifest["config"]["digest"], int(manifest["config"]["size"]))
    return layers, config


def parse_buildlog(text):
    names, cached = {}, set()
    for line in text.splitlines():
        m = re.match(r"^#(\d+) \[(?:[^\]]*\s)?(\d+/\d+)\] (.+)$", line)
        if m:
            names.setdefault(m.group(1), (m.group(2), m.group(3).strip()))
        m = re.match(r"^#(\d+) CACHED\s*$", line)
        if m:
            cached.add(m.group(1))
    steps = [
        (
            step,
            instr,
            (
                "BASE"
                if instr.startswith("FROM ")
                else "CACHED" if vid in cached else "RE-RUN"
            ),
        )
        for vid, (step, instr) in names.items()
    ]
    return sorted(steps, key=lambda s: int(s[0].split("/")[0]))


def mb(n_bytes):
    return n_bytes / 1e6


def build_report(v1_manifest, v2_manifest, v2_steps, update_note):
    v1_layers, v1_cfg = blobs(v1_manifest)
    v2_layers, v2_cfg = blobs(v2_manifest)
    v1_digests = {d for d, _ in v1_layers}
    new_layers = [(d, s) for d, s in v2_layers if d not in v1_digests]
    full_bytes = sum(s for _, s in v2_layers) + v2_cfg[1]
    delta_bytes = sum(s for _, s in new_layers) + (
        v2_cfg[1] if v2_cfg[0] != v1_cfg[0] else 0
    )

    fleets = {}
    for name, trucks in FLEETS.items():
        full_mb, delta_mb = mb(full_bytes) * trucks, mb(delta_bytes) * trucks
        fleets[name] = {
            "trucks": trucks,
            "full_image_mb": full_mb,
            "layer_cache_mb": delta_mb,
            "saved_mb": full_mb - delta_mb,
            "full_image_cost_rs": full_mb * PRICE_PER_MB,
            "layer_cache_cost_rs": delta_mb * PRICE_PER_MB,
            "saved_rs_per_update": (full_mb - delta_mb) * PRICE_PER_MB,
            "saved_rs_per_year": (full_mb - delta_mb) * PRICE_PER_MB * UPDATES_PER_YEAR,
        }
    return {
        "update": update_note,
        "price_rs_per_mb": PRICE_PER_MB,
        "mb_definition": "1 MB = 1,000,000 bytes (cellular billing unit)",
        "updates_per_year": UPDATES_PER_YEAR,
        "v2_build_steps": [
            {"step": s, "instruction": i, "result": r} for s, i, r in v2_steps
        ],
        "layers_total": len(v2_layers),
        "layers_new_in_v2": len(new_layers),
        "new_layer_digests": [d for d, _ in new_layers],
        "full_image_bytes_per_truck": full_bytes,
        "layer_cache_bytes_per_truck": delta_bytes,
        "saving_fraction": 1 - delta_bytes / full_bytes,
        "fleets": fleets,
    }


def markdown(r):
    p, f = r["fleets"]["pilot"], r["fleets"]["full"]
    out = [
        "### D2 OTA layer-cache demo",
        "",
        f"Model-only update: {r['update']}",
        "",
        "**v2 rebuild after changing only the model file**",
        "",
        "| Step | Instruction | Result |",
        "|---|---|---|",
    ]
    for s in r["v2_build_steps"]:
        instr = (
            s["instruction"]
            if len(s["instruction"]) <= 90
            else s["instruction"][:87] + "..."
        )
        badge = {
            "CACHED": "✅ CACHED",
            "RE-RUN": "🔄 RE-RUN",
            "BASE": "base image (already on truck)",
        }[s["result"]]
        out.append(f"| {s['step']} | `{instr}` | {badge} |")
    out += [
        "",
        f"Registry blobs: **{r['layers_new_in_v2']} of {r['layers_total']} layers** are new in v2.",
        "",
        "**Bandwidth per OTA model update** (compressed bytes a truck downloads; "
        f"₹{r['price_rs_per_mb']:.2f}/MB, MB = 10⁶ bytes)",
        "",
        "| | Per truck | 85 trucks (pilot) | 265 trucks (full fleet) |",
        "|---|---:|---:|---:|",
        f"| Full image pull | {mb(r['full_image_bytes_per_truck']):,.2f} MB | {p['full_image_mb']:,.1f} MB = ₹{p['full_image_cost_rs']:,.2f} | {f['full_image_mb']:,.1f} MB = ₹{f['full_image_cost_rs']:,.2f} |",
        f"| Layer-cached update | {mb(r['layer_cache_bytes_per_truck']):,.4f} MB | {p['layer_cache_mb']:,.3f} MB = ₹{p['layer_cache_cost_rs']:,.2f} | {f['layer_cache_mb']:,.3f} MB = ₹{f['layer_cache_cost_rs']:,.2f} |",
        f"| **Saved per update** | **{r['saving_fraction'] * 100:.3f} %** | **{p['saved_mb']:,.1f} MB = ₹{p['saved_rs_per_update']:,.2f}** | **{f['saved_mb']:,.1f} MB = ₹{f['saved_rs_per_update']:,.2f}** |",
        f"| Saved per year ({r['updates_per_year']:.2f} updates) | | ₹{p['saved_rs_per_year']:,.2f} | ₹{f['saved_rs_per_year']:,.2f} |",
        "",
    ]
    return "\n".join(out)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(
        description="D2 OTA layer-cache report and fleet bandwidth saving"
    )
    ap.add_argument("--registry", default="http://localhost:5000")
    ap.add_argument("--repo", default="logi-edge")
    ap.add_argument("--v1-tag", default="ota-v1")
    ap.add_argument("--v2-tag", default="ota-v2")
    ap.add_argument(
        "--v2-log", required=True, help="plain BuildKit log of the v2 rebuild"
    )
    ap.add_argument("--update-note", default="model file replaced")
    ap.add_argument("--out", required=True, help="JSON report path")
    args = ap.parse_args()

    steps = parse_buildlog(open(args.v2_log, encoding="utf-8", errors="replace").read())
    report = build_report(
        image_manifest(args.registry, args.repo, args.v1_tag),
        image_manifest(args.registry, args.repo, args.v2_tag),
        steps,
        args.update_note,
    )
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    md = markdown(report)
    print(md)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as f:
            f.write(md + "\n")

    rerun = [s for s in report["v2_build_steps"] if s["result"] == "RE-RUN"]
    if not rerun or not all("training/models/" in s["instruction"] for s in rerun):
        print(
            f"::error::expected only the model COPY to re-run, got: {rerun}",
            file=sys.stderr,
        )
        return 1
    if report["layers_new_in_v2"] == 0:
        print(
            "::error::v2 adds no new layer - the model file did not change",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
