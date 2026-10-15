#!/bin/bash
# Convert (once), build the engines, benchmark, and check parity with the
# official demo output.
#   scripts/workflows/build_and_bench.sh [eval.wav ...]
# STRT_DECODER=cuda|trt  STRT_ENCODER_PRECISION=fp16|bf16|fp32
source "$(dirname "$0")/common.sh"
if [ ! -f "${STRT_BUNDLE}/encoder.safetensors" ]; then
  PY tools/convert_torchscript.py "${TS_ARGS[@]}" --output "${STRT_BUNDLE}"
fi
WAV_ARGS=()
if [ $# -gt 0 ]; then WAV_ARGS=(--wav "$1"); fi
PY scripts/build_engines.py --bundle "${STRT_BUNDLE}" \
  --engine-dir "${STRT_ENGINE_DIR}" --onnx-dir "${STRT_WORK}/onnx" "${WAV_ARGS[@]}" \
  --encoder-precision "${STRT_ENCODER_PRECISION:-fp16}" --decoder "${STRT_DECODER:-cuda}"
PY scripts/bench_e2e.py --engine-dir "${STRT_ENGINE_DIR}" "${WAV_ARGS[@]}" \
  --output "${STRT_ENGINE_DIR}/bench_e2e.json"
if [ $# -gt 0 ]; then
  PY tools/compare_demo.py --engine-dir "${STRT_ENGINE_DIR}" "${TS_ARGS[@]}" \
    --output-dir "${STRT_ENGINE_DIR}/compare" "$@"
fi
echo "BUILD_AND_BENCH_OK"
