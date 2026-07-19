#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VOFA_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo cd into ${VOFA_DIR}
cd "${VOFA_DIR}"
python run.py --alg FalconDAgger --config configs/train_student.yaml --expert_path logs/2026-01-15-21-25-31/nn/model_100000.pth
