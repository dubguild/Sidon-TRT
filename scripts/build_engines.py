#!/usr/bin/env python3
"""Export ONNX and build the encoder + decoder-head TensorRT engines (and
the CUDA-tail weights) for bucketed / batched execution.

    build_engines.py --bundle B --engine-dir E --onnx-dir /tmp/x
    build_engines.py --bundle B --engine-dir E/short --onnx-dir /tmp/y \
        --max-batch 16 --max-seconds 32          # short-clip set

The long set covers the official demo chunk (96 s; the decoder takes one
extra carried frame). Requires a GPU and NVIDIA_TF32_OVERRIDE=0.
Validation (build_report.json):
- accuracy of runs vs the exact-length fp32 torch reference;
- padding invariance: the same clip run in two different bucket paddings
  must give the same valid output (the masking contract);
- batching: clips of different lengths in one padded batch vs one by one.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sidon_trt import (  # noqa: E402
    cuda_tail,
    flexrel_triton,
    frontend,
    runtime,
    snake_triton,
)
from sidon_trt.audio_io import read_wav  # noqa: E402
from sidon_trt.engines import (  # noqa: E402
    build_engine,
    build_strong_engine,
    errors,
)
from sidon_trt.export_onnx import (  # noqa: E402
    FULL_MASKS,
    HEAD_MASKS,
    ExportableDACDecoder,
    ExportableDACHead,
    ExportableDACMasked,
    export_encoder,
    export_head,
    head_mask_lengths,
)
from sidon_trt.modeling import (  # noqa: E402
    load_bundle,
    replace_attention_with_flex,
)

MIN_SEQ = runtime.MIN_SEQ


def test_waveform(wav: Path | None, seconds: float) -> torch.Tensor:
    """16 kHz preprocessed audio, tiled up to `seconds`."""
    total = int(seconds * 16_000)
    if wav is None:
        torch.manual_seed(0)
        return torch.randn(total) * 0.1
    data, rate = read_wav(wav)
    wav16 = frontend.preprocess(torch.from_numpy(data), rate)
    return wav16.repeat(-(-total // wav16.numel()))[:total].contiguous()


@torch.no_grad()
def reference(bundle, front, chunk: torch.Tensor):
    """exact-length fp32 torch: features, hidden, audio [960 L - 26]."""
    torch.set_float32_matmul_precision("highest")
    features = front(chunk.cuda().unsqueeze(0))
    hidden = bundle.encoder(features).float()
    audio = bundle.decoder(hidden.transpose(1, 2)).float().view(-1)
    return features, hidden, audio


@torch.no_grad()
def run_bucketed(engines: runtime.EngineSet, clips: list, bucket: int):
    """clips (16 kHz) zero-padded to `bucket` as one batch -> per clip
    (hidden [L, 1024], audio [960 L - 26]) of the valid region."""
    batch = len(clips)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        wave = engines.wave_view(batch, bucket)
        wave.zero_()
        for row, clip in enumerate(clips):
            wave[row, : clip.numel()].copy_(clip)
        engines.wave_valid[:batch].copy_(
            torch.tensor([c.numel() for c in clips], dtype=torch.int32)
        )
        engines.enqueue(batch, bucket, cached=False)
    torch.cuda.synchronize()
    seq = frontend.num_features(bucket)
    hidden = engines.hidden[: batch * seq * 1024].view(batch, seq, 1024)
    audio = engines.audio_view(batch, bucket, cached=False)
    out = []
    for row, clip in enumerate(clips):
        frames = frontend.num_features(clip.numel())
        out.append(
            (hidden[row, :frames].clone(),
             audio[row, : runtime.HOP * frames - runtime.TRIM].clone())
        )
    return out


def ceil_to(value: int, unit: int) -> int:
    return -(-value // unit) * unit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--engine-dir", type=Path, required=True)
    parser.add_argument("--onnx-dir", type=Path, required=True)
    parser.add_argument("--wav", type=Path, default=None)
    parser.add_argument(
        "--encoder-precision", choices=["bf16", "fp16", "fp32"],
        default="fp16",
        help="w2v-BERT engine dtype (fp16: bf16 was the main deviation from "
        "official Sidon; fp32 ~5x slower)",
    )
    parser.add_argument(
        "--decoder", choices=["cuda", "trt"], default="cuda",
        help="cuda: TRT head (blocks 0-1) + fused CUDA tail (default, "
        "fastest); trt: the whole DAC decoder in TensorRT (fallback, no "
        "custom kernels; ~1.7x slower decoder)",
    )
    parser.add_argument(
        "--decoder-fp32", default="final",
        help="--decoder trt: strongly typed fp16 decoder with these parts in "
        "fp32 (comma list of final, residual, block0..4; 'none' = all fp16; "
        "'weak' = weakly typed fp16 build). Default final: fp32 final "
        "Snake + conv + tanh brings the log-mel deviation from official "
        "0.28 -> 0.10 dB for +1%% time",
    )
    parser.add_argument("--max-batch", type=int, default=1)
    parser.add_argument(
        "--max-seconds", type=float, default=runtime.DEMO_CHUNK_SECONDS,
        help="longest chunk (default: the demo's 96 s chunk)",
    )
    parser.add_argument(
        "--validate-seconds",
        type=lambda s: [float(v) for v in s.split(",")],
        default=None,
        help="clip lengths to validate (default: 15, 60, max for the long "
        "set; 3, 11.3, max-3 for a batched set)",
    )
    args = parser.parse_args()
    args.onnx_dir.mkdir(parents=True, exist_ok=True)
    args.engine_dir.mkdir(parents=True, exist_ok=True)
    max_seq = int(args.max_seconds * 50)
    max_batch = args.max_batch
    opt_batch = max(1, max_batch // 2)
    opt_seq = max_seq if max_batch == 1 else max_seq // 2

    device = torch.device("cuda")
    # register BEFORE any TensorRT builder/runtime work (segfault otherwise)
    flexrel_triton.register_trt_plugin_qr_dyn()
    snake_triton.register_trt_plugin()
    bundle = load_bundle(args.bundle, device="cuda")
    front = frontend.SeamlessFbank().to(device).eval()
    waveform = test_waveform(args.wav, args.max_seconds + 4)
    report = {
        "gpu": torch.cuda.get_device_name(0),
        "encoder_precision": args.encoder_precision,
        "decoder": args.decoder,
        "max_batch": max_batch,
        "max_seconds": args.max_seconds,
        "engines": {},
        "validation": {},
    }

    features, hidden, _ = reference(bundle, front, waveform[: 600 * 320])
    encoder_onnx = args.onnx_dir / "encoder.onnx"
    encoder_copy = copy.deepcopy(bundle.encoder)
    replace_attention_with_flex(encoder_copy.model)
    export_encoder(
        encoder_copy, features, encoder_onnx,
        dtype={
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "fp32": torch.float32,
        }[args.encoder_precision],
    )
    del encoder_copy
    _, seconds = build_strong_engine(
        encoder_onnx,
        {
            "features": (
                (1, MIN_SEQ, 160),
                (opt_batch, opt_seq, 160),
                (max_batch, max_seq, 160),
            ),
            "lens": ((1,), (opt_batch,), (max_batch,)),
        },
        args.engine_dir / runtime.ENGINE_FILES["encoder"],
    )
    report["engines"]["encoder"] = {"build_seconds": seconds}
    print(f"encoder engine built in {seconds:.0f}s", flush=True)

    exportable = ExportableDACDecoder(
        copy.deepcopy(bundle.decoder), gemm_upsample=True, snake_plugin=True,
    )
    fp32_parts = None
    if args.decoder == "trt" and args.decoder_fp32 != "weak":
        fp32_parts = tuple(
            p for p in args.decoder_fp32.split(",") if p and p != "none"
        )
        report["decoder_fp32"] = list(fp32_parts)
    head = (
        ExportableDACHead(exportable) if args.decoder == "cuda"
        else ExportableDACMasked(exportable.to(device), fp32_parts)
    ).to(device).eval()
    from_block = 2 if args.decoder == "cuda" else 4
    mask_names = HEAD_MASKS if args.decoder == "cuda" else FULL_MASKS
    head_onnx = args.onnx_dir / "decoder_head.onnx"
    export_head(head, hidden, head_onnx)
    head_profile = {
        "hidden": (
            (1, MIN_SEQ, 1024),
            (opt_batch, opt_seq + 1, 1024),
            (max_batch, max_seq + 1, 1024),
        ),
    }
    for name, low, mid, high in zip(
        mask_names,
        head_mask_lengths(MIN_SEQ, from_block),
        head_mask_lengths(opt_seq + 1, from_block),
        head_mask_lengths(max_seq + 1, from_block),
    ):
        head_profile[name] = ((1, low), (opt_batch, mid), (max_batch, high))
    if fp32_parts is None:
        _, seconds = build_engine(
            head_onnx,
            head_profile,
            args.engine_dir / runtime.ENGINE_FILES["decoder"],
            precision="fp16",
        )
    else:
        _, seconds = build_strong_engine(
            head_onnx, head_profile,
            args.engine_dir / runtime.ENGINE_FILES["decoder"],
        )
    report["engines"]["decoder_head"] = {"build_seconds": seconds}
    print(f"decoder head engine built in {seconds:.0f}s", flush=True)
    tail_path = args.engine_dir / cuda_tail.TAIL_FILE
    tail = None
    if args.decoder == "cuda":
        tail = cuda_tail.CudaDecoderTail(bundle.decoder).to(device).eval()
        cuda_tail.save_tail(tail, bundle.decoder_config, tail_path)
    else:
        tail_path.unlink(missing_ok=True)

    engines = runtime.EngineSet(args.engine_dir, tail, front)
    lengths = args.validate_seconds or (
        [15.0, 60.0, args.max_seconds] if max_batch == 1
        else [3.0, 11.3, args.max_seconds - 3.0]
    )
    unit = 2 * frontend.FRAME_SHIFT
    for seconds_ in lengths:
        n = int(seconds_ * 16_000) - 37  # deliberately not frame-aligned
        clip = waveform[:n].cuda()
        _, ref_hidden, ref_audio = reference(bundle, front, clip)
        tight = run_bucketed(engines, [clip], ceil_to(n, unit))[0]
        loose_bucket = min(ceil_to(n, unit) + 3 * 16_000, engines.max_samples)
        loose = run_bucketed(engines, [clip], loose_bucket)[0]
        entry = {
            "samples": n,
            "encoder_vs_fp32": errors(ref_hidden[0], tight[0]),
            "audio_vs_fp32": errors(ref_audio, tight[1]),
            "padding_invariance_hidden": errors(tight[0], loose[0]),
            "padding_invariance_audio": errors(tight[1], loose[1]),
        }
        report["validation"][f"{seconds_:g}s"] = entry
        print(f"{seconds_:g}s: {json.dumps(entry)}", flush=True)

    if max_batch > 1:  # mixed lengths in one batch vs one by one
        sizes = [int(s * 16_000) + k for k, s in enumerate(
            [2.1, 7.5, 13.0, 19.9, args.max_seconds - 4.0]
        )][:max_batch]
        clips = [waveform[i * 50_000 : i * 50_000 + n].cuda()
                 for i, n in enumerate(sizes)]
        bucket = ceil_to(max(sizes), unit)
        batched = run_bucketed(engines, clips, bucket)
        report["validation"]["batch"] = {}
        for clip, (b_hidden, b_audio) in zip(clips, batched):
            alone = run_bucketed(
                engines, [clip], ceil_to(clip.numel(), unit)
            )[0]
            entry = {
                "hidden": errors(alone[0], b_hidden),
                "audio": errors(alone[1], b_audio),
            }
            report["validation"]["batch"][str(clip.numel())] = entry
            print(f"batch {clip.numel()}: {json.dumps(entry)}", flush=True)

    (args.engine_dir / "build_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(f"BUILD_ENGINES_OK engine_dir={args.engine_dir}", flush=True)


if __name__ == "__main__":
    main()
