#!/usr/bin/env bash
# End-to-end check of one model variant inside the inference image:
#   container (MODEL_PATH=<model>) -> logibridge/trucks/<truck>/inference -> payload validation
#   + drift monitor builds a reference distribution from the same live payloads.
# Requires: a running broker container named "mosquitto" on localhost:1883 and the
# sensor simulator already publishing to it.
#
# Usage: scripts/integration_test.sh <image> <model_file> [truck_id] [messages]
set -euo pipefail

IMAGE="$1"
MODEL="$2"
TRUCK_ID="${3:-truck_001}"
MESSAGES="${4:-3}"
NAME="inference-${MODEL%%.*}"
OUT="${RUNNER_TEMP:-/tmp}/it-${MODEL%%.*}"
mkdir -p "$OUT"

cleanup() {
  docker logs "$NAME" > "$OUT/container.log" 2>&1 || true
  docker rm -f "$NAME" > /dev/null 2>&1 || true
  [ -n "${MONITOR_PID:-}" ] && kill "$MONITOR_PID" 2> /dev/null || true
}
trap cleanup EXIT

echo "=== $MODEL ==="
docker run -d --name "$NAME" --network host \
  -e MODEL_PATH="$MODEL" -e TRUCK_ID="$TRUCK_ID" \
  -e MQTT_BROKER=localhost -e MQTT_PORT=1883 \
  "$IMAGE" > /dev/null

for i in $(seq 1 120); do
  if grep -q "Connected successfully" <(docker logs "$NAME" 2>&1); then
    echo "connected after ${i}s"
    break
  fi
  if [ "$(docker inspect -f '{{.State.Running}}' "$NAME")" != "true" ]; then
    docker logs "$NAME"
    echo "::error::$MODEL: inference container exited"
    exit 1
  fi
  [ "$i" -eq 120 ] && { docker logs "$NAME"; echo "::error::$MODEL: no broker connection in 120s"; exit 1; }
  sleep 1
done

python monitoring/drift_monitor.py --mode reference --reference-size "$MESSAGES" \
  --truck-id "$TRUCK_ID" --reference "$OUT/reference_dist.json" > "$OUT/drift_monitor.log" 2>&1 &
MONITOR_PID=$!

docker exec mosquitto mosquitto_sub -h localhost \
  -t "logibridge/trucks/$TRUCK_ID/inference" -C "$MESSAGES" -W 120 > "$OUT/payloads.jsonl"

python - "$OUT/payloads.jsonl" "$TRUCK_ID" <<'PY'
import json, sys
rows = [json.loads(line) for line in open(sys.argv[1]) if line.strip()]
assert rows, "no payloads received"
for r in rows:
    assert {"truck_id", "status", "confidence", "probabilities", "timestamp"} <= set(r), r
    assert r["truck_id"] == sys.argv[2], r
    assert r["status"] in {"NORMAL", "WARNING", "CRITICAL"}, r
    p = r["probabilities"]
    assert isinstance(p, list) and len(p) == 3, r
    assert all(0.0 <= x <= 1.0 for x in p), r
    assert abs(sum(p) - 1.0) < 0.02, f"probabilities do not sum to 1: {r}"   # INT8 output step = 1/256
    assert abs(max(p) - r["confidence"]) < 1e-6, r
    assert r["status"] == ["NORMAL", "WARNING", "CRITICAL"][p.index(max(p))], r
    print(f'{r["status"]:<9} confidence={r["confidence"]:.3f}  P(Normal)={p[0]:.3f}')
print(f"OK: {len(rows)} valid payloads")
PY

for i in $(seq 1 30); do
  [ -s "$OUT/reference_dist.json" ] && break
  sleep 1
done
cat "$OUT/drift_monitor.log"
[ -s "$OUT/reference_dist.json" ] || { echo "::error::$MODEL: drift monitor built no reference from live payloads"; exit 1; }
echo "drift monitor reference: $(python -c "import json,sys; d=json.load(open(sys.argv[1])); print(d['counts'])" "$OUT/reference_dist.json")"
