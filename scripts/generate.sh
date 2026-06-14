#!/bin/bash
# 数据生成脚本
set -e

CONFIG=${CONFIG:-"configs/settings.yaml"}

echo "Generating benchmark data..."
echo "Config: ${CONFIG}"

CMD="uv run python -m src.generation --config ${CONFIG}"
if [ -n "${OUTPUT}" ]; then
    CMD="${CMD} --output ${OUTPUT}"
fi
if [ -n "${SAMPLES}" ]; then
    CMD="${CMD} --samples ${SAMPLES}"
fi
if [ -n "${INTENTS}" ]; then
    CMD="${CMD} --intents ${INTENTS}"
fi
if [ -n "${THREADS}" ]; then
    CMD="${CMD} --threads ${THREADS}"
fi

eval ${CMD}
