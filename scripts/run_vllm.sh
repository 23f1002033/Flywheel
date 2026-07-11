#!/usr/bin/env bash
# Launch vLLM on ROCm serving Gemma + the ACTIVE LoRA adapter from the registry.
# Requires: pip install vllm (ROCm build per AMD Developer Cloud docs),
# huggingface-cli login (Gemma license must be accepted on HF).
set -e
ACTIVE_ADAPTER=$(python - <<'PY'
from app.flywheel import registry
v = registry.active_version()
print(v["adapter_path"] if v else "")
PY
)
ARGS="--model google/gemma-2-2b-it --port 8001 --dtype bfloat16"
if [ -n "$ACTIVE_ADAPTER" ]; then
  echo "Serving with adapter: $ACTIVE_ADAPTER"
  ARGS="$ARGS --enable-lora --lora-modules flywheel=$ACTIVE_ADAPTER"
fi
python -m vllm.entrypoints.openai.api_server $ARGS