# Language Model Benchmarks

This file tracks training accuracy for language model experiments run on Google Colab.

## Contents

- [Runtime Environment](#runtime-environment)
- [Accuracy Comparison](#accuracy-comparison)
- [Summary Comparison](#summary-comparison)
- [Transformer Experiment Notes](#transformer-experiment-notes)
  - [torch.compile Cold vs Warm Run](#torchcompile-cold-vs-warm-run-tinytransformer-2-layers-context_size8)
  - [Layer Depth Comparison (2 vs 4 layers)](#tinytransformer-layer-depth-comparison-2-vs-4-layers)
  - [Context Size Comparison](#tinytransformer-context-size-accuracy-comparison)
  - [bfloat16 vs float16 on T4](#bfloat16-vs-float16-on-t4)
- [Generated Samples](#generated-samples)

## Runtime Environment

- Platform: Google Colab
- GPU: T4 GPU
- Backend: Python 3 (Google Compute Engine, GPU)
- System RAM: 12.7 GB
- GPU RAM: 15.0 GB
- Disk: 112.6 GB

## Accuracy Comparison

**Key:** TT = TinyTransformer.py, TTC = TinyTransformerClass.py, µGPT = microgpt_lite.py

| Epoch | NameSLP.py | TinyMLP.py | TorchMLP.py | TT (2 layers) | TTC | µGPT | LlamaLite | TT (4 layers) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 3.5% | 4.7% | 21.4% | 19.3% | 19.3% | 1.7% | 19.6% | 19.3% |
| 200 | 37.1% | 44.8% | 54.3% | 54.8% | 54.7% | 53.6% | 47.3% | 56.8% |
| 400 | 38.2% | 48.9% | 58.0% | 58.3% | 58.7% | 65.2% | 53.7% | 60.7% |
| 600 | 38.6% | 52.3% | 59.1% | 60.4% | 60.6% | 68.6% | 57.1% | 62.1% |
| 800 | 38.9% | 55.0% | 59.9% | 63.2% | 63.9% | 71.4% | 58.3% | 64.6% |
| 1000 | 39.1% | 56.4% | 60.8% | 65.4% | 65.1% | 71.9% | 60.9% | 65.9% |
| 1200 | 39.2% | 56.7% | 61.4% | 65.5% | 64.9% | 73.3% | 62.6% | 66.6% |
| 1400 | 39.4% | 58.2% | 60.8% | 66.0% | 66.8% | 74.6% | 63.0% | 67.6% |
| 1600 | 39.5% | 58.3% | 61.8% | 67.0% | 66.8% | 76.0% | 64.1% | 68.0% |
| 1800 | 39.5% | 59.2% | 61.1% | 67.7% | 67.8% | 75.9% | 66.4% | 69.0% |
| 2000 | 39.6% | 59.4% | 62.4% | 67.4% | 68.1% | 77.0% | 65.6% | 68.9% |
| 2200 | - | - | - | - | - | - | - | 72.8% |
| 2400 | - | - | - | - | - | - | - | 71.6% |
| 2600 | - | - | - | - | - | - | - | 70.6% |
| 2800 | - | - | - | - | - | - | - | 72.0% |
| 3000 | - | - | - | - | - | - | - | 72.0% |
| 3200 | - | - | - | - | - | - | - | 72.5% |
| 3400 | - | - | - | - | - | - | - | 73.1% |
| 3500 | - | - | - | - | - | 79.4% | - | - |

## Summary Comparison

| Model | Best Accuracy | Step | Training Time |
|---|---:|---:|---:|
| NameSLP.py | 39.6% | 2000 | 35.1s |
| TinyMLP.py | 59.4% | 2000 | 3.9s |
| TorchMLP.py | 62.4% | 2000 | 3.6s |
| TinyTransformer.py (2 layers, cold start) | 67.9% | 2000 | 46.3s |
| TinyTransformer.py (2 layers, warm start) | 68.4% | 2000 | 19.7s |
| TinyTransformer.py (`context_size=64`) | 68.5% | 1800 | 197.5s |
| TinyTransformer.py (4 layers, 3,193,920 params) | 73.1% | 3400 | 79.9s |
| TinyTransformerClass.py (1,614,400 params) | 68.1% | 2000 | 19.3s |
| microgpt_lite.py | 79.4% | 3500 | 202.0s |
| LlamaLite (`context_size=32`, 1.59M params) | 66.4% | 1800 | 62.7s |
| TinyTransformer.py (bfloat16, T4) | 68.6% | 2000 | 82.0s |

## Transformer Experiment Notes

- `5000` TinyStories often causes CUDA OOM/crash in this setup.
- Increasing transformer context window to `64` made training much slower, with only a small accuracy gain in this pair of runs.
- First cold run of 4-layer model was 71.9s due to Colab initialisation overhead; warm runs settle at ~41.4s.
- Running 4 layers to 3500 steps (79.9s) reaches 73.1%, closing the gap with µGPT (79.4%) at a fraction of the training time (202.0s).
- `torch.compile` causes a ~26s cold-start overhead on the first run (46.3s total) as PyTorch's Inductor compiles and caches CUDA kernels. Subsequent warm runs reuse the compiled kernel cache and run at 19.7s — the true steady-state training speed.

### torch.compile Cold vs Warm Run (TinyTransformer, 2 layers, context_size=8)

| Step | Accuracy | Cold Time | Warm Time |
|---:|---:|---:|---:|
| 0 | 19.3% | 27.2s | 0.0s |
| 200 | 54.8% | 29.1s | 2.1s |
| 400 | 58.7% | 31.0s | 4.0s |
| 600 | 60.6% | 32.9s | 5.9s |
| 800 | 62.6% | 34.8s | 7.9s |
| 1000 | 65.5% | 36.7s | 9.8s |
| 1200 | 65.3% | 38.6s | 11.8s |
| 1400 | 66.8% | 40.5s | 13.8s |
| 1600 | 67.3% | 42.4s | 15.7s |
| 1800 | 67.3% | 44.4s | 17.7s |
| 2000 | 67.9% | 46.3s | 19.7s |

The ~26s cold-start overhead is entirely front-loaded at Step 0 (27.2s vs 0.0s). Per-step speed is identical (~1.9s/200 steps) in both runs. The Inductor kernel cache persists between runs within the same session (or to disk via `TORCHINDUCTOR_FX_GRAPH_CACHE=1`).

### TinyTransformer Layer Depth Comparison (2 vs 4 layers)

| Step | 2 layers (1,614,400 params) | 4 layers (3,193,920 params) |
|---:|---:|---:|
| 0 | 19.3% | 19.3% |
| 200 | 54.8% | 56.8% |
| 400 | 58.3% | 60.7% |
| 600 | 60.4% | 62.1% |
| 800 | 63.2% | 64.6% |
| 1000 | 65.4% | 65.9% |
| 1200 | 65.5% | 66.6% |
| 1400 | 66.0% | 67.6% |
| 1600 | 67.0% | 68.0% |
| 1800 | 67.7% | 69.0% |
| 2000 | 67.4% | 68.9% |

Training time (2000 steps): `21.1s` (2 layers) vs `45.7s` (4 layers) → ~2.2x slower.

Quick comparison (at 2000 steps):

- Best accuracy: `67.7%` (2 layers) vs `68.9%` (4 layers) → `+1.2` points.
- Training time: `21.1s` (2 layers) vs `45.7s` (4 layers) → about `2.2x` slower with 4 layers.
- Parameters: `1,614,400` (2 layers) vs `3,193,920` (4 layers) → ~2x more parameters.
- At 3400 steps, 4 layers reaches `73.1%` in `79.9s` → a much better result given more time.

### TinyTransformer Context Size Accuracy Comparison

| Epoch | `context_size=8` | `context_size=64` |
|---:|---:|---:|
| 0 | 19.8% | 19.0% |
| 200 | 55.7% | 55.5% |
| 400 | 59.4% | 58.4% |
| 600 | 60.7% | 61.3% |
| 800 | 62.7% | 64.4% |
| 1000 | 63.1% | 64.8% |
| 1200 | 66.3% | 64.4% |
| 1400 | 66.8% | 68.3% |
| 1600 | 67.4% | 67.6% |
| 1800 | 66.9% | 68.5% |
| 2000 | 67.0% | 68.4% |

Training time:

- `context_size=8`: `25.4s`
- `context_size=64`: `197.5s`

Quick comparison:

- Best accuracy: `67.4%` (`context_size=8`) vs `68.5%` (`context_size=64`) -> `+1.1` points.
- Training time: `25.4s` (`context_size=8`) vs `197.5s` (`context_size=64`) -> about `7.8x` slower at `context_size=64`.

### bfloat16 vs float16 on T4

| Step | Accuracy (bfloat16) | Time (bfloat16) |
|---:|---:|---:|
| 0 | 19.3% | 0.2s |
| 200 | 55.3% | 8.4s |
| 400 | 58.5% | 16.5s |
| 600 | 61.1% | 24.8s |
| 800 | 64.4% | 33.1s |
| 1000 | 65.8% | 41.3s |
| 1200 | 66.0% | 49.5s |
| 1400 | 66.6% | 57.6s |
| 1600 | 67.2% | 65.7s |
| 1800 | 68.2% | 73.9s |
| 2000 | 68.6% | 82.0s |

**Conclusion:** bfloat16 is **~4.2× slower** than float16 on the T4 GPU (82.0s vs ~19.7s warm). The T4 (Turing architecture) has no native bfloat16 tensor cores — it falls back to float32 compute internally, losing all speed benefit. bfloat16 is only advantageous on Ampere (A100) or Hopper (H100) GPUs. **Use float16 on T4.**

## Generated Samples

### NameSLP.py

- emma
- osola
- riganna
- ahala
- horme
- rayly
- etannoye
- toraeyn
- alose
- gettel
- yandilon
- ceamira
- anleiph
- kafrin
- melia
- j

### TinyMLP.py

```text
Once tichec. Ther.
She said outned. Sker to. Hif even very the box. It. I mesis momors. He day.
"Se! smiled in outsy lows.
They played, it and said, "Yes, I wast, hure ats a creany five a bind. She saidy
```

### TorchMLP.py

```text
Once upon a time, to mak,""
The learry tried that her the corne but he saw two learned. She chess smal wife sell best couldn't my her and ran was a big for naughed loved clean withing. Mommy!"I will magin
```

### TinyTransformer.py (2 layers)

```text
Once there. She wise her bird was family face on on the thought it was so happy and put the tent down and said, "Mom, Tim, and they also much fun. They are red back well. One day, a big boy named Tim.
```

### TinyTransformer.py (4 layers, 3500 steps)

```text
Once there was a little girl named Sam. Sam was so happy and started to play with the camera. They made a big hill and the birds fly something shine and saw a big tree.
```

### TinyTransformerClass.py

```text
Once there. She was playing nap. They laughed a more a little girl named Fluffy was a great time her talked away. Tom looked a bunny he did not like to get reache. They loved to play with the balls
```

### microgpt_lite.py

```text
Once upon a time, there was a little boy named Tim. He loved to measure his favorite toy. One day, he saw a big, deep broken shirt. He thought it would be fun to play with it.
```

### LlamaLite (`context_size=32`)

```text
Once there was a little boy named Tiny. The marked were very share hugged something and wanted to the walked to see a difffort to play outside. The little girl got red and got. Lily loved to play with his pretty
```
