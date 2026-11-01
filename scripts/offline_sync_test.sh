#!/usr/bin/env bash
set -euo pipefail

TRUCK_ID="${1:-truck_001}"
N_ALERTS="${2:-5}"
IMAGE=eclipse-mosquitto:2
WORK="${RUNNER_TEMP:-/tmp}/offline-sync"
OPS_PORT=1884
EDGE_PORT=1885
ALERT_TOPIC="logibridge/trucks/$TRUCK_ID/alerts"

rm -rf "$WORK"; mkdir -p "$WORK/ops-data" "$WORK/edge-data"
chmod 777 "$WORK/ops-data" "$WORK/edge-data"

cleanup() {
  for c in lg-ops-broker lg-edge-broker; do
    echo "::group::docker logs $c"; docker logs "$c" 2>&1 | tail -n 40 || true; echo "::endgroup::"
    docker rm -f "$c" > /dev/null 2>&1 || true
  done
}
trap cleanup EXIT

wait_port() {
  for _ in $(seq 1 30); do
    mosquitto_pub_c "$1" -t lg/probe -m x -q 0 2> /dev/null && return 0
    sleep 1
  done
  echo "::error::broker on port $1 did not come up"; return 1
}
mosquitto_pub_c() { local port="$1"; shift; docker run --rm --network host "$IMAGE" mosquitto_pub -h localhost -p "$port" "$@"; }
mosquitto_sub_c() { local port="$1"; shift; docker run --rm --network host "$IMAGE" mosquitto_sub -h localhost -p "$port" "$@"; }

printf 'listener %s\nallow_anonymous true\npersistence true\npersistence_location /mosquitto/data/\nlog_dest stdout\n' \
  "$OPS_PORT" > "$WORK/ops.conf"
sed -e "s/^listener 1883 127.0.0.1$/listener $EDGE_PORT/" \
    -e "s/^address .*/address localhost:$OPS_PORT/" \
    -e "s/^restart_timeout .*/restart_timeout 1 5/" \
    -e '/^bridge_\(cafile\|certfile\|keyfile\|tls_version\)/d' \
    deployment/mosquitto/mosquitto.conf > "$WORK/edge.conf"
grep -q "^address localhost:$OPS_PORT" "$WORK/edge.conf" && grep -q "^listener $EDGE_PORT" "$WORK/edge.conf" \
  || { echo "::error::could not adapt deployment/mosquitto/mosquitto.conf"; exit 1; }
echo "::group::edge broker config under test (from deployment/mosquitto/mosquitto.conf)"
grep -vE '^\s*(#|$)' "$WORK/edge.conf"
echo "::endgroup::"

start_ops()  { docker run -d --name lg-ops-broker  --network host -v "$WORK/ops.conf:/mosquitto/config/mosquitto.conf:ro"  -v "$WORK/ops-data:/mosquitto/data"  "$IMAGE" > /dev/null; }
start_edge() { docker run -d --name lg-edge-broker --network host -v "$WORK/edge.conf:/mosquitto/config/mosquitto.conf:ro" -v "$WORK/edge-data:/mosquitto/data" "$IMAGE" > /dev/null; }

echo "=== 1. Coverage available: start ops-centre and edge brokers, bridge connects ==="
start_ops;  wait_port "$OPS_PORT"
mosquitto_sub_c "$OPS_PORT" -i ops-centre-consumer -c -q 1 -t "logibridge/trucks/+/alerts" -E
start_edge; wait_port "$EDGE_PORT"
for i in $(seq 1 30); do
  docker logs lg-ops-broker 2>&1 | grep -q "New bridge connected" && { echo "bridge connected"; break; }
  [ "$i" -eq 30 ] && { echo "::error::edge bridge never connected to the ops broker"; exit 1; }
  sleep 1
done

echo "=== 2. Cellular gap: ops-centre link goes down ==="
docker stop lg-ops-broker > /dev/null
date -u +"gap started %H:%M:%SZ"

echo "=== 3. Truck raises $N_ALERTS alerts during the gap (QoS 1 to the local broker) ==="
for i in $(seq 1 "$N_ALERTS"); do
  payload=$(printf '{"truck_id":"%s","seq":%d,"event":"ESCALATED","status":"CRITICAL","raised_offline":true,"timestamp":%s}' \
            "$TRUCK_ID" "$i" "$(date +%s)")
  mosquitto_pub_c "$EDGE_PORT" -q 1 -t "$ALERT_TOPIC" -m "$payload"
  echo "raised offline: $payload"
done

echo "=== 4. Edge node reboots during the gap (queue must survive on disk) ==="
docker stop lg-edge-broker > /dev/null && docker rm lg-edge-broker > /dev/null
start_edge; wait_port "$EDGE_PORT"

echo "=== 5. Coverage returns: ops-centre link back up ==="
docker rm lg-ops-broker > /dev/null
start_ops; wait_port "$OPS_PORT"
date -u +"gap ended %H:%M:%SZ"

echo "=== 6. Ops centre receives the queued alerts ==="
mosquitto_sub_c "$OPS_PORT" -i ops-centre-consumer -c -q 1 -t "logibridge/trucks/+/alerts" \
  -C "$N_ALERTS" -W 60 | tee "$WORK/received.jsonl"

python3 - "$WORK/received.jsonl" "$N_ALERTS" <<'PY'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
n = int(sys.argv[2])
seqs = [r["seq"] for r in rows]
assert len(rows) == n, f"expected {n} alerts after the gap, got {len(rows)}: {seqs}"
assert seqs == list(range(1, n + 1)), f"alerts out of order or duplicated: {seqs}"
print(f"OK: {n}/{n} alerts raised offline were delivered in order after the gap and an edge reboot")
PY

if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then
  {
    echo "### Offline store-and-forward (brief section 2)"
    echo "Edge broker config: \`deployment/mosquitto/mosquitto.conf\` (bridge \`local_cleansession false\`, persistence on)."
    echo ""
    echo "| Check | Result |"
    echo "|---|---|"
    echo "| Alerts raised while the ops-centre link was down | $N_ALERTS |"
    echo "| Edge broker restarted during the gap | yes |"
    echo "| Alerts delivered after coverage returned | $N_ALERTS / $N_ALERTS, in order |"
  } >> "$GITHUB_STEP_SUMMARY"
fi