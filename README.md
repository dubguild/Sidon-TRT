# Sidon-TRT

Sidon-TRT is a fast inference implementation of
[Sidon v0.1](https://huggingface.co/sarulab-speech/sidon-v0.1) speech
restoration (any-rate noisy speech in, 48 kHz restored speech out),
implemented by Rikuto Kotoge. It is built by **converting the released
weights — no retraining** — and matches the official demo's output to
within ~0.1 dB log-mel.

Two decoder back-ends, selected at engine build time:

- `--decoder cuda` (default): DAC decoder head in TensorRT + fused CUDA
  kernels for the 48 kHz tail. Fastest and closest to the official output;
  needs Hopper / datacenter-Blackwell shared memory.
- `--decoder trt`: the whole decoder in TensorRT (strongly typed fp16
  with an fp32 output layer, Snake as a Triton plugin), no custom CUDA
  kernels. Same accuracy, ~1.7x slower; fallback for other GPUs or if the
  kernels are not wanted.

## Official Sidon vs Sidon-TRT (H100 SXM, batch 1)

Official = the released TorchScript (fp32) with the demo's CPU fbank, run
as the demo runs it — no per-stage synchronization, so the CPU fbank of
the next chunk overlaps the GPU work of the previous one
(`scripts/bench_baseline.py`; with a sync after every stage, for the
breakdown, it takes 3462 ms). Sidon-TRT = `scripts/bench_e2e.py`, one CUDA
graph per chunk (front-end + encoder + decoder).

| | official | Sidon-TRT `--decoder trt` | Sidon-TRT `--decoder cuda` |
|---|---|---|---|
| 600 s input, end to end | 2677 ms | 404 ms (6.6x) | **236 ms (11.3x)** |
| 128 clips 1.5-25 s (29 min) | — | 1599 ms | **1065 ms** |
| device memory after init | — | 4.7 GB | 3.0 GB |
| log-mel L1 vs official (5 noisy mixtures) | 0 | 0.08-0.13 dB | **0.08-0.13 dB** |
| custom CUDA kernels / GPU limit | no | no / any TRT GPU | yes / H100, B200 class |

Per chunk length (milliseconds; official: GPU stages + CPU fbank of that
chunk, serial):

| chunk | official enc / dec / CPU fbank | **official total** | trt: enc / dec | **trt total** | cuda: enc / head / tail | **cuda total** | cuda RTF |
|---|---|---|---|---|---|---|---|
| 15 s | 13.8 / 17.8 / 23.7 | **55** | 2.5 / 9.2 | **11.3** | 2.5 / 2.3 / 2.9 | **7.2** | 4.8e-4 |
| 30 s | 33.8 / 32.7 / 45.2 | **112** | 3.5 / 17.3 | **20.4** | 3.5 / 3.9 / 5.3 | **12.3** | 4.1e-4 |
| 60 s | 102.7 / 63.9 / 91.1 | **258** | 5.5 / 33.6 | **38.9** | 5.5 / 7.4 / 10.3 | **22.7** | 3.8e-4 |
| 96 s | 234.3 / 109.9 / 144.3 | **489** | 9.3 / 53.6 | **62.7** | 9.2 / 11.4 / 15.8 | **36.4** | 3.8e-4 |

(The front-end — fbank on GPU — is 0.33 ms in every Sidon-TRT row.
Official preprocessing of a 600 s file adds 410 ms CPU; Sidon-TRT's GPU
preprocessing takes 3 ms.)

File throughput with `scripts/run.py` (decode / encode in worker
processes, `--decoder cuda`): ~1040x realtime disk to disk over 5323
speech files (FLAC, 8.6 h in total), GPU busy 98%.

**Why batch 1.** From 15 s up, a single chunk already gives every kernel
far more parallel work than the GPU has SMs (the 48 kHz decoder stages
alone are millions of rows) and CUDA graphs remove the launch overhead,
so there is nothing left for batching to fill: the cost per audio second
is flat from 30 s to 96 s chunks (RTF 4.1e-4 → 3.8e-4 above). Batching
mixed lengths also pads every item to the longest one. Measured with a
batched engine set: no gain on mixed 1.5-25 s clips or long files, ~1.3x
only on uniformly short clips (~5.8 s average), for +14 GB of memory —
so it was removed.

**Chunks longer than 96 s.** The engines accept longer chunks
(`build_engines.py --max-seconds`, `run.py --chunk-seconds`), but this is
not validated in the current version and gains little: with the cost per
second already flat, a longer chunk only removes a few chunk boundaries
(an earlier 120 s build ran at RTF 3.7e-4 vs 3.8e-4 at 96 s), while
moving the output further from the official 96 s chunking.

## Accuracy

- Conversion is exact: encoder 126 dB, decoder bit-identical vs the
  released TorchScript (fp32), `tools/convert_torchscript.py`.
- Engines vs the fp32 torch reference: encoder hidden ~52-53 dB (fp16 with
  an fp32 feature-projection island); audio waveform SNR 26-27 dB for both
  decoders (fp16 phase noise; the log-mel distance is the meaningful
  number).
- `--decoder cuda` vs the official demo output (log-mel L1 on speech
  frames): **mean 0.11 / p95 0.17 / max 0.42 dB** over 200 random
  speech clips; 0.08-0.09 dB on 7-15 min 2-channel telephone
  recordings (`--per-channel`).
- `--decoder trt` precision (`--decoder-fp32`, log-mel L1 vs official on
  the 5 mixtures / 96 s chunk time):

  | TRT decoder | log-mel L1 | 96 s chunk |
  |---|---|---|
  | weakly typed fp16 (`weak`) | 0.27-0.32 dB | 61.9 ms |
  | strongly typed, all fp16 (`none`) | 0.19-0.23 dB | 60.6 ms |
  | + fp32 final Snake/conv/tanh (`final`, **default**) | **0.08-0.13 dB** | 62.7 ms |
  | + fp32 residual stream (`final,residual`) | 0.08-0.13 dB | 75.6 ms |
  | + fp32 block 4, 48 kHz (`final,block4`) | 0.08-0.13 dB | 80.5 ms |

  The deviation came almost entirely from the last 48-channel conv
  (48x7 taps summed into one sample) and tanh in fp16; the CUDA tail
  already computes them in fp32. bf16 would not help: fewer mantissa
  bits than fp16, and TensorRT has no fast bf16 tactics for this decoder.
- For scale: re-chunking the official model itself (20 s instead of 96 s
  windows) moves its output by 1.4-3.2 dB.
- Large-scale run (5323 clips + long 2-channel recordings): no non-finite,
  clipped or wrong-length outputs; bit-identical across repeated runs.
- Listening: `tools/make_listening_set.py` writes named and blind A/B pairs
  (official vs Sidon-TRT, also as stereo L=official / R=Sidon-TRT);
  `tools/make_listening_variants.py` adds fp32 / bf16 variants. fp16 was
  chosen after listening.

## How

```
wav (any rate/channels)
 └ GPU preprocess: peak 0.9, 50 Hz high-pass (fp64 FFT conv), 16 kHz resample
 └ demo chunking: +1.5 s tail, 96 s chunks, 1-frame feature carry-over
   └ fp32 SeamlessM4T fbank (torch)                         ┐
   └ encoder — TensorRT fp16: w2v-BERT 8 layers, LoRA       │ one CUDA graph
       merged, fp32 feature projection, fused Triton        │ per length bucket
       relative_key attention plugin with key-length mask   │ (1 s steps <= 32 s,
   └ decoder (--decoder cuda):                              │  4 s above)
       head — TensorRT fp16: input conv, blocks 0-1,        │
         block 2 upsample (time-major out)                  │
       tail — fused CUDA kernels (csrc/dac_kernels.cu):     │
         each residual unit of blocks 2-4 is one kernel     │
         (Snake -> dilated conv -> Snake -> 1x1 -> residual,│
         mma.sync, weights resident or streamed), Snake +   │
         ConvTranspose upsamples, Snake + conv + tanh       │
     decoder (--decoder trt): all of it in TensorRT fp16    ┘
```

- **Exact bucketing**: a chunk zero-padded to its bucket gives the
  exact-length output — fbank statistics over valid frames only, attention
  keys past the valid length masked, every decoder conv re-zeroed past it
  (validity masks are engine inputs). Any mix of input lengths reuses
  ~30-50 captured graphs.
- **File throughput**: `sidon_trt.stream.FileStreamer` decodes and encodes
  in spawned worker processes overlapped with the GPU.

## GPU support

Built and measured on H100 only; engines are architecture-specific (build
them on the deployment GPU). The CUDA tail kernels need up to ~210 KB
shared memory per block (Hopper / datacenter Blackwell); on A100 (164 KB)
or Ada such as RTX 4090 (~99 KB) build with `--decoder trt`. The kernels
are compiled by nvcc on first use for the visible GPU.

## Quick start

```bash
git clone <this repo> Sidon-TRT && cd Sidon-TRT
python -m venv .venv && source .venv/bin/activate && pip install -e .
source configs/env.example.sh       # NVIDIA_TF32_OVERRIDE=0 (required), TensorRT libs
# nvcc on PATH (only for the default --decoder cuda)
python tools/convert_torchscript.py --output bundle/   # downloads sidon-v0.1 from the Hub
python scripts/build_engines.py --bundle bundle/ --engine-dir engines/ \
    --onnx-dir /tmp/sidon-trt-onnx [--decoder trt]
python scripts/run.py --engine-dir engines/ --input-dir corpus/ \
    --output-dir restored/ --format flac [--per-channel]
```

Or in one go, with a parity check against the official model on your own
inputs: `scripts/workflows/build_and_bench.sh eval1.wav eval2.wav`.
See `docs/SETUP.md` for the environment, the other workflows and pitfalls.

The runtime detects the decoder type from the engine directory.

```python
from sidon_trt.runtime import TRTPipeline
pipeline = TRTPipeline("engines/")
audio48k = pipeline.restore(waveform, sample_rate)              # demo chunking
stereo = pipeline.restore(two_channel, sr, per_channel=True)    # 1 speaker / channel
sliding = pipeline.restore_sliding(waveform, sr, window_seconds=20)

from sidon_trt.stream import FileStreamer                       # many files
with FileStreamer(pipeline, readers=8, writers=8) as streamer:  # spawned workers:
    stats = streamer.run(paths, "restored/", fmt="flac")        # use a __main__ guard
```

`--per-channel` restores each channel separately (2-channel telephone
dialogue, channel layout kept); the default downmixes like the official
demo. See `docs/SETUP.md` for setup details and pitfalls.

## Design notes (measured and dropped)

| tried | outcome |
|---|---|
| TensorRT decoder with Snake exported as ops | TRT does not fuse Snake: 57% of decoder time |
| `sidon::Snake` Triton TRT plugin | kept for the TensorRT decoder parts (`--decoder trt`) |
| fused residual unit as Triton TRT AOT plugin | slower than TRT convs (pow2 padding, 48 KiB AOT smem cap); removed |
| CUDA residual units, WMMA, channel-major | 8-10 ms/unit; replaced by time-major + mma.sync (1-2 ms/unit) |
| bf16 encoder | 0.4-1.1 dB off official; fp16 + fp32 projection: 0.1 dB |
| fp16 feature projection | up to 14 dB off on short clips (silent frames: LayerNorm variance ~0) |
| bf16 CUDA tail | 18 dB less accurate than fp16 at the same speed |
| TensorRT bf16 decoder | as slow as fp32 (no fast bf16 conv tactics); strongly typed fails in Myelin |
| more fp32 in the TRT decoder (block 4, residual stream) | no accuracy gain over the fp32 output layer, +20-30% time |
| all-fp32 engines | 0.01-0.03 dB off official, 6.3x slower than fp16 |
| in-graph length masks in TensorRT | TRT 10.16 fp16 produced all-zero output -> masks are inputs |
| batching short clips | ~1.3x on real short clips, none on mixed lengths, +14 GB; removed |
| 20 s sliding windows (training length) | 16% slower; differs from 96 s chunking by 1.4-3.2 dB (window effect) |
| wgmma/TMA tail kernels | not done: est. ~2x on the tail, ~1.2x overall |

## Repository layout

| path | |
|---|---|
| `src/sidon_trt/frontend.py` | demo preprocessing (CPU + GPU) and fbank, incl. bucket masking |
| `src/sidon_trt/modeling.py` | encoder / DAC decoder rebuilt as torch modules |
| `src/sidon_trt/export_onnx.py` | encoder, decoder head and TRT-only decoder ONNX export |
| `src/sidon_trt/runtime.py` | `TRTPipeline`: bucketed CUDA graphs, demo chunking, sliding windows |
| `src/sidon_trt/cuda_tail.py`, `csrc/dac_kernels.cu` | fused CUDA decoder tail |
| `src/sidon_trt/cuda_ext.py` | nvcc build + load of the kernels (no ninja) |
| `src/sidon_trt/stream.py`, `fileio.py`, `audio_io.py` | overlapped decode / restore / encode |
| `src/sidon_trt/flexrel_triton.py`, `snake_triton.py` | TensorRT plugins (attention, Snake) |
| `src/sidon_trt/engines.py` | engine build / TRT stage helpers |
| `src/sidon_trt/demo.py` | the official demo pipeline (baseline + parity reference) |
| `tools/convert_torchscript.py` | TorchScript -> bundle with parity check |
| `tools/compare_demo.py`, `make_listening_*.py`, `make_window_compare.py` | parity and listening sets |
| `tools/stress_test.py`, `test_cuda_tail.py` | large-scale and kernel tests |
| `scripts/` | `build_engines.py`, `run.py`, `bench_e2e.py`, `bench_baseline.py` |
| `scripts/workflows/`, `scripts/examples/` | shell workflows (build, benchmarks, tests, listening) and a batch-job template |
| `src/sidon_trt/hub.py` | download of the released TorchScript files (pinned revision) |

## Acknowledgements

Sidon-TRT is an inference port of **Sidon** by sarulab-speech
(Wataru Nakata, Yuki Saito, Yota Ueda, Hiroshi Saruwatari); all model weights and the method are
theirs, and this repository only converts and accelerates the released
v0.1 model without retraining. If you use it, please cite the Sidon paper:

- Paper: [Sidon: Fast and Robust Open-Source Multilingual Speech
  Restoration for Large-scale Dataset Cleansing (ICASSP 2026, arXiv:2509.17052)](https://arxiv.org/abs/2509.17052)
- Model: [sarulab-speech/sidon-v0.1](https://huggingface.co/sarulab-speech/sidon-v0.1) (MIT)
- Demo whose preprocessing and chunking are reproduced here:
  [sarulab-speech/sidon_demo_beta](https://huggingface.co/spaces/sarulab-speech/sidon_demo_beta)

```bibtex
@inproceedings{nakata2026sidon,
  title={Sidon: Fast and robust open-source multilingual speech restoration for large-scale dataset cleansing},
  author={Nakata, Wataru and Saito, Yuki and Ueda, Yota and Saruwatari, Hiroshi},
  booktitle={ICASSP 2026-2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={17617--17621},
  year={2026},
  organization={IEEE}
}
```

Sidon builds on [w2v-BERT 2.0](https://huggingface.co/facebook/w2v-bert-2.0)
(Meta) and the [DAC](https://github.com/descriptinc/descript-audio-codec)
decoder architecture (Descript).
<!-- The fused relative-position attention
TensorRT plugin is shared with FastDialogueSidon. -->

## License

MIT License, Copyright (c) 2026 Rikuto Kotoge (see `LICENSE`). The Sidon
model weights are released by sarulab-speech under their own MIT license.
