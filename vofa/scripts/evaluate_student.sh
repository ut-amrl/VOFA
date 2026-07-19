#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VOFA_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG=${1:-logs/vofa/student_policy/config.yaml}
CHECKPOINT=${2:-logs/vofa/student_policy/nn/model.pth}

echo cd into ${VOFA_DIR}
cd "${VOFA_DIR}"

python run.py --alg FalconCamEvaluate --eval \
    --policy student \
    --num_envs 1000 \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT"
