#!/usr/bin/env bash
set -euo pipefail

NEW_MODEL="$1"
OUT="${2:-${RUNNER_TEMP:-/tmp}/ota-demo}"
MODEL_IN_IMAGE=training/models/cold_chain_mlp_pruned_int8.tflite
ROLLBACK_MODEL=training/models/cold_chain_mlp_ptq_int8.tflite
REGISTRY=localhost:5000
REPO=logi-edge
mkdir -p "$OUT"

[ -f "$NEW_MODEL" ] || { echo "::error::new model not found: $NEW_MODEL"; exit 1; }
cp "$MODEL_IN_IMAGE" "$OUT/model_v1.tflite"
restore() { cp "$OUT/model_v1.tflite" "$MODEL_IN_IMAGE"; }
trap restore EXIT

if ! curl -sf "http://$REGISTRY/v2/" > /dev/null; then
  docker run -d --name ota-registry -p 5000:5000 registry:2 > /dev/null
  for _ in $(seq 1 30); do curl -sf "http://$REGISTRY/v2/" > /dev/null && break; sleep 1; done
fi

echo "::group::v1 build (cold: every step runs)"
DOCKER_BUILDKIT=1 docker build --progress=plain -f inference/Dockerfile -t "$REPO:ota-v1" . 2>&1 | tee "$OUT/build_v1.log"
echo "::endgroup::"

old_sha=$(sha256sum "$MODEL_IN_IMAGE" | cut -d' ' -f1)
new_sha=$(sha256sum "$NEW_MODEL" | cut -d' ' -f1)
if [ "$old_sha" != "$new_sha" ]; then
  NOTE="new M3 release: \`$(basename "$MODEL_IN_IMAGE")\` retrained in this CI run (sha256 ${old_sha:0:12} → ${new_sha:0:12})"
  cp "$NEW_MODEL" "$MODEL_IN_IMAGE"
else
  rb_sha=$(sha256sum "$ROLLBACK_MODEL" | cut -d' ' -f1)
  NOTE="retrained M3 is byte-identical to the committed one, so the update ships M2's weights (F3 rollback) in \`$(basename "$MODEL_IN_IMAGE")\` (sha256 ${old_sha:0:12} → ${rb_sha:0:12})"
  cp "$ROLLBACK_MODEL" "$MODEL_IN_IMAGE"
fi
echo "OTA update: $NOTE"
git status --short -- training/models inference data_pipeline

echo "::group::v2 rebuild after the model-only change"
DOCKER_BUILDKIT=1 docker build --progress=plain -f inference/Dockerfile -t "$REPO:ota-v2" . 2>&1 | tee "$OUT/build_v2.log"
echo "::endgroup::"

echo "=== docker history: v1 vs v2 (top = newest layer) ==="
docker history --format 'table {{.ID}}\t{{.Size}}\t{{.CreatedBy}}' --no-trunc "$REPO:ota-v1" | cut -c1-150 | tee "$OUT/history_v1.txt"
echo "---"
docker history --format 'table {{.ID}}\t{{.Size}}\t{{.CreatedBy}}' --no-trunc "$REPO:ota-v2" | cut -c1-150 | tee "$OUT/history_v2.txt"

echo "=== push both to the local registry (what a truck would pull from) ==="
for tag in ota-v1 ota-v2; do
  docker tag "$REPO:$tag" "$REGISTRY/$REPO:$tag"
  docker push "$REGISTRY/$REPO:$tag" 2>&1 | tee "$OUT/push_$tag.log"
done

python3 scripts/ota_report.py --registry "http://$REGISTRY" --repo "$REPO" \
  --v1-tag ota-v1 --v2-tag ota-v2 --v2-log "$OUT/build_v2.log" \
  --update-note "$NOTE" --out "$OUT/ota_layer_cache.json"