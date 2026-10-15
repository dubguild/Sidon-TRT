# shellcheck shell=bash
# Sourced by every workflow: repo root, environment, shared defaults.
# Override any STRT_* variable from the environment.
set -euo pipefail
STRT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
if [ -f "${STRT_ROOT}/configs/env.local.sh" ]; then
  source "${STRT_ROOT}/configs/env.local.sh"
else
  source "${STRT_ROOT}/configs/env.example.sh"
fi
STRT_BUNDLE=${STRT_BUNDLE:-${STRT_ROOT}/bundle}              # converted weights
STRT_ENGINE_DIR=${STRT_ENGINE_DIR:-${STRT_ROOT}/engines}     # TensorRT engines
STRT_WORK=${STRT_WORK:-${TMPDIR:-/tmp}/sidon-trt-${USER:-user}}  # scratch outputs
# local sidon-v0.1 TorchScript files; unset = download from the Hub
TS_ARGS=()
if [ -n "${STRT_TORCHSCRIPT:-}" ]; then TS_ARGS=(--torchscript-dir "${STRT_TORCHSCRIPT}"); fi
mkdir -p "${STRT_WORK}"
cd "${STRT_ROOT}"
PY() { "${STRT_PYTHON}" -W ignore "$@"; }
