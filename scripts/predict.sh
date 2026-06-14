#!/bin/bash
# 预测脚本
set -e

CONFIG=${CONFIG:-"configs/settings.yaml"}

echo "Running predictions..."
echo "Config: ${CONFIG}"

CMD="uv run python -m src.prediction.run_predict --config ${CONFIG}"
if [ -n "${MODE}" ]; then
    CMD="${CMD} --mode ${MODE}"
fi

eval ${CMD}
