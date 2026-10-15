#!/bin/bash
# Listening material: official vs Sidon-TRT A/B pairs (named, blind,
# stereo) and stage-by-stage variants.
#   scripts/workflows/listening.sh output_dir input.wav [...]
# STRT_ENGINE_DIR_FP32 (all-fp32 engines), STRT_BF16_TAIL=1 add variants.
source "$(dirname "$0")/common.sh"
[ $# -gt 1 ] || { echo "usage: $0 output_dir input.wav [...]" >&2; exit 2; }
OUT=$1; shift
PY tools/make_listening_set.py --engine-dir "${STRT_ENGINE_DIR}" "${TS_ARGS[@]}" \
  --output-dir "${OUT}" "$@"
EXTRA=()
if [ -n "${STRT_ENGINE_DIR_FP32:-}" ]; then EXTRA+=(--engine-dir-fp32 "${STRT_ENGINE_DIR_FP32}"); fi
if [ -n "${STRT_BF16_TAIL:-}" ]; then EXTRA+=(--bf16-tail); fi
PY tools/make_listening_variants.py --engine-dir "${STRT_ENGINE_DIR}" "${TS_ARGS[@]}" \
  --bundle "${STRT_BUNDLE}" --output-dir "${OUT}/variants" "${EXTRA[@]}" "$@"
echo "LISTENING_OK"
