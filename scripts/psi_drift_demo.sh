#!/usr/bin/env bash
set -euo pipefail

IMAGE="$1"
MODEL="${2:-cold_chain_mlp_pruned_int8.tflite}"
TRUCK_ID="${3:-truck_001}"
OUT="${4:-${RUNNER_TEMP:-/tmp}/psi-demo}"
REF_SIZE="${REF_SIZE:-300}"
WINDOW="${WINDOW:-100}"
INTERVAL="${INTERVAL:-60}"
MIN_SAMPLES="${MIN_SAMPLES:-20}"
CLEAN_MIN="${CLEAN_MIN:-17}"
ALERT_DEADLINE_S="${ALERT_DEADLINE_S:-300}"
HOLD_AFTER_ALERT_S="${HOLD_AFTER_ALERT_S:-120}"
RECOVER_MAX_MIN="${RECOVER_MAX_MIN:-30}"
NAME=psi-demo-inference
mkdir -p "$OUT"
EVENTS="$OUT/events.csv"
MONLOG="$OUT/monitor.log"
echo "event,utc" > "$EVENTS"

event() { echo "$1,$(date -u +%Y-%m-%dT%H:%M:%S+00:00)" >> "$EVENTS"; echo ">>> $(date -u +%H:%M:%S) $1"; }
SIM_PID=""; MON_PID=""; TAIL_PID=""
start_sim() {
  [ -n "$SIM_PID" ] && { kill "$SIM_PID" 2> /dev/null || true; wait "$SIM_PID" 2> /dev/null || true; }
  python data_pipeline/simulator.py --anomaly "$1" >> "$OUT/simulator.log" 2>&1 &
  SIM_PID=$!
  echo "simulator restarted with --anomaly $1 (pid $SIM_PID)"
}
cleanup() {
  for p in "$TAIL_PID" "$MON_PID" "$SIM_PID"; do [ -n "$p" ] && kill "$p" 2> /dev/null || true; done
  docker logs "$NAME" > "$OUT/container.log" 2>&1 || true
  docker rm -f "$NAME" > /dev/null 2>&1 || true
}
trap cleanup EXIT

wait_for() {
  local regex="$1" after="$2" timeout="$3" start=$SECONDS line
  while (( SECONDS - start < timeout )); do
    line=$(tail -n +"$((after + 1))" "$MONLOG" | grep -m1 -E "$regex" || true)
    [ -n "$line" ] && { echo "$line"; return 0; }
    kill -0 "$MON_PID" 2> /dev/null || { echo "::error::drift monitor exited" >&2; return 2; }
    sleep 5
  done
  return 1
}

echo "=== E1 PSI drift demo | model=$MODEL truck=$TRUCK_ID | reference=$REF_SIZE window=$WINDOW interval=${INTERVAL}s ==="
docker run -d --name "$NAME" --network host \
  -e MODEL_PATH="$MODEL" -e TRUCK_ID="$TRUCK_ID" -e MQTT_BROKER=localhost -e MQTT_PORT=1883 \
  "$IMAGE" > /dev/null
start_sim none

echo "=== 1. Reference: $REF_SIZE clean inferences (one per 10 s) ==="
python monitoring/drift_monitor.py --mode reference --reference-size "$REF_SIZE" \
  --truck-id "$TRUCK_ID" --reference "$OUT/reference_dist.json" 2>&1 | tee "$OUT/reference.log"
cat "$OUT/reference_dist.json"; echo

echo "=== 2. Monitor: PSI on the last $WINDOW inferences every ${INTERVAL}s ==="
: > "$MONLOG"
python monitoring/drift_monitor.py --mode monitor --reference "$OUT/reference_dist.json" \
  --truck-id "$TRUCK_ID" --window "$WINDOW" --interval "$INTERVAL" --min-samples "$MIN_SAMPLES" \
  --log "$OUT/psi_log.csv" > "$MONLOG" 2>&1 &
MON_PID=$!
tail -n +1 -f "$MONLOG" &
TAIL_PID=$!
event monitor_start

echo "=== 3. Clean phase: $CLEAN_MIN min ==="
sleep $((CLEAN_MIN * 60))

echo "=== 4. Drift injection: --anomaly combined (PSI must exceed 0.25 within ${ALERT_DEADLINE_S}s) ==="
mark=$(wc -l < "$MONLOG")
start_sim combined
event inject_combined
if ! alert=$(wait_for '\[LOGIBRIDGE DRIFT ALERT\]' "$mark" "$ALERT_DEADLINE_S"); then
  echo "::error::no [LOGIBRIDGE DRIFT ALERT] within ${ALERT_DEADLINE_S}s of injection"; exit 1
fi
echo "::notice title=E1 drift detected::$alert"
sleep "$HOLD_AFTER_ALERT_S"

echo "=== 5. Recovery: --anomaly none (PSI must return below 0.10) ==="
mark=$(wc -l < "$MONLOG")
start_sim none
event restore_clean
if ! ok=$(wait_for 'PSI=0\.0[0-9]{2}( |$)' "$mark" $((RECOVER_MAX_MIN * 60))); then
  echo "::error::PSI did not return below 0.10 within $RECOVER_MAX_MIN min"; exit 1
fi
echo "::notice title=E1 recovered::$ok"
sleep $((2 * INTERVAL))
event end
kill "$MON_PID" 2> /dev/null || true; wait "$MON_PID" 2> /dev/null || true

python monitoring/psi_report.py --log "$OUT/psi_log.csv" --events "$EVENTS" \
  --json "$OUT/psi_summary.json" --png "$OUT/psi_timeline.png"