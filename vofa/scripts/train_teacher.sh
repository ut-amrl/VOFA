#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VOFA_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo cd into ${VOFA_DIR}
cd "${VOFA_DIR}"
python run.py --alg FalconPPO --config configs/train_teacher.yaml
