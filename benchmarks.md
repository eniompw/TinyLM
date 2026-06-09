# Language Model Benchmarks

This file tracks training experiments on character-level language models trained on TinyStories. The baseline model is **TinyTransformer.py** (2-layer transformer, float16, ReLU, learned pos_embed, `context_size=8`). All runs use Google Colab T4 GPU unless noted.

> ⚠️ **T4 Session Variance:** Colab assigns T4s from a shared pool. Warm run times can vary from ~19.7s to ~27.3s (~1.9s vs ~2.7s per 200 steps) depending on which physical GPU is assigned. Always run at least twice when benchmarking.

## Contents

- [Runtime Environment](#runtime-environment)
- [Model Comparison](#model-comparison)
- [Ablation Summary](#ablation-summary)
- [Step-by-Step Accuracy](#step-by-step-accuracy)
- [Experiment Details](#experiment-details)
  - [torch.compile Cold vs Warm Run](#torchcompile-cold-vs-warm-run)
  - [Layer Depth Comparison](#layer-depth-comparison-2-vs-4-layers)
  - [Context Size Comparison](#context-size-comparison)
  - [bfloat16 vs float16](#bfloat16-vs-float16)
  - [Weight Tying](#weight-tying)
  - [GELU vs ReLU Activation](#gelu-vs-relu-activation)
  - [Positional Embedding Ablation](#positional-embedding-ablation)
- [Generated Samples](#generated-samples)

## Runtime Environment

- Platform: Google Colab
- GPU: T4 GPU
- Backend: Python 3 (Google Compute Engine, GPU)
- System RAM: 12.7 GB
- GPU RAM: 15.0 GB
- Disk: 112.6 GB

## Model Comparison

Canonical model results — best configuration per architecture.

| Model | Best Accuracy | Step | Training Time |
|---|---:|---:|---:|
| NameSLP.py | 39.6% | 2000 | 35.1s |
| TinyMLP.py | 59.4% | 2000 | 3.9s |
| TorchMLP.py | 62.4% | 2000 | 3.6s |
| TinyTransformer.py (2 layers, warm start) | 68.4% | 2000 | 19.7s |
| TinyTransformer.py (`context_size=64`) | 68.5% | 1800 | 197.5s |
| TinyTransformer.py (4 layers, 3,193,920 params) | 73.1% | 3400 | 79.9s |
| TinyTransformerClass.py (1,614,400 params) | 68.1% | 2000 | 19.3s |
| LlamaLite (`context_size=32`, 1.59M params) | 66.4% | 1800 | 62.7s |
| microgpt_lite.py | 79.4% | 3500 | 202.0s |

## Ablation Summary

Single-change experiments on TinyTransformer.py baseline (~68% accuracy, ~21s warm). All run on T4 GPU, 2000 steps warm start.

| Change | Accuracy Δ | Speed Δ | Verdict |
|---|---:|---:|---|
| float16 → bfloat16 | +0.2% | ~4.2× slower | ❌ T4 has no native bf16 tensor cores |
| Weight tying (`linear.weight = tok_embed.weight`) | −3.0% | neutral | ❌ Init mismatch + small vocab |
| ReLU → GELU activation | neutral | ~14% slower | ❌ `erf()` overhead not offset by compile |
| Remove `pos_embed` | −7.7% | negligible | ❌ Breaks token order — essential at all context sizes |
| Flash Attention (`F.scaled_dot_product_attention`) | TBD | TBD | ⏳ Next experiment |

## Step-by-Step Accuracy

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

## Experiment Details

### torch.compile Cold vs Warm Run

**Change:** First vs subsequent runs with `torch.compile` enabled.

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

**Conclusion:** The ~26s cold-start penalty is entirely front-loaded at Step 0 (27.2s vs 0.0s). Per-step speed is identical (~1.9s/200 steps) in both runs. The Inductor kernel cache persists between runs within the same session, or to disk via `TORCHINDUCTOR_FX_GRAPH_CACHE=1`.

### Layer Depth Comparison (2 vs 4 layers)

**Change:** `n_layers = 2` → `n_layers = 4` (params: 1,614,400 → 3,193,920).

| Step | 2 layers | 4 layers |
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
| 3400 | - | 73.1% |

**Conclusion:** 4 layers gives +1.2% at 2000 steps but is 2.2× slower (21.1s vs 45.7s). At 3400 steps (79.9s total) it reaches 73.1%, closing the gap with µGPT (79.4%) at a fraction of µGPT's 202.0s training time.

### Context Size Comparison

**Change:** `context_size = 8` → `context_size = 64`.

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

**Conclusion:** `context_size=64` gives only +1.1% accuracy but is **7.8× slower** (197.5s vs 25.4s). This is O(T²) attention scaling — the primary motivation for implementing Flash Attention.

### bfloat16 vs float16

**Change:** `torch.autocast(..., dtype=torch.float16)` → `torch.bfloat16` (2 occurrences).

| Step | Accuracy (float16) | Accuracy (bfloat16) | Time (bfloat16) |
|---:|---:|---:|---:|
| 0 | 19.3% | 19.3% | 0.2s |
| 200 | 54.8% | 55.3% | 8.4s |
| 400 | 58.3% | 58.5% | 16.5s |
| 600 | 60.4% | 61.1% | 24.8s |
| 800 | 63.2% | 64.4% | 33.1s |
| 1000 | 65.4% | 65.8% | 41.3s |
| 1200 | 65.5% | 66.0% | 49.5s |
| 1400 | 66.0% | 66.6% | 57.6s |
| 1600 | 67.0% | 67.2% | 65.7s |
| 1800 | 67.7% | 68.2% | 73.9s |
| 2000 | 67.4% | 68.6% | 82.0s |

**Conclusion:** bfloat16 is **~4.2× slower** on T4 (82.0s vs ~19.7s warm). The T4 (Turing architecture) has no native bfloat16 tensor cores and falls back to float32 compute internally. bfloat16 is only advantageous on Ampere (A100) or Hopper (H100) GPUs. **Use float16 on T4.**

### Weight Tying

**Change:** Added `linear.weight = tok_embed.weight` after model definition (1 line).

| Step | Accuracy (baseline) | Accuracy (weight tied) |
|---:|---:|---:|
| 0 | 19.3% | 1.9% |
| 200 | 55.5% | 41.5% |
| 400 | 59.6% | 47.8% |
| 600 | 60.3% | 51.4% |
| 800 | 64.1% | 57.1% |
| 1000 | 66.1% | 60.0% |
| 1200 | 65.4% | 61.0% |
| 1400 | 67.5% | 62.6% |
| 1600 | 66.8% | 63.7% |
| 1800 | 68.4% | 65.2% |
| 2000 | 68.2% | 65.2% |

Training time: `27.3s` (baseline) vs `27.2s` (weight tied) — neutral speed.

**Conclusion:** Weight tying is **net negative** (−3% accuracy). Causes:
1. **Init mismatch:** `nn.Linear` uses Kaiming uniform; `nn.Embedding` uses standard normal — tying forces the linear layer into a poor initialisation, causing loss to explode to 256 at step 0.
2. **Small vocab:** Weight tying benefits large-vocab models (e.g. GPT-2, 50K tokens). On a ~65-char vocab the gradient signal improvement is negligible.
3. **Double-counting risk:** Passing both `tok_embed.parameters()` and `linear.parameters()` to AdamW when tied may double-count gradients.

### GELU vs ReLU Activation

**Change:** Added `activation='gelu'` to `nn.TransformerEncoderLayer` (1 argument, default is `'relu'`).

| Step | Accuracy (ReLU) | Time (ReLU) | Accuracy (GELU) | Time (GELU) |
|---:|---:|---:|---:|---:|
| 0 | 19.3% | 0.0s | 19.2% | 0.0s |
| 200 | 54.7% | 2.1s | 53.4% | 2.4s |
| 400 | 59.3% | 4.2s | 58.1% | 4.8s |
| 600 | 60.4% | 6.3s | 60.7% | 7.1s |
| 800 | 64.8% | 8.3s | 63.8% | 9.5s |
| 1000 | 64.8% | 10.4s | 65.5% | 11.9s |
| 1200 | 66.1% | 12.5s | 65.3% | 14.3s |
| 1400 | 66.6% | 14.6s | 66.4% | 16.7s |
| 1600 | 67.0% | 16.8s | 67.0% | 19.1s |
| 1800 | 67.2% | 18.9s | 67.2% | 21.5s |
| 2000 | 68.1% | 21.0s | 68.1% | 23.9s |

Run 3× warm. ReLU per-step: ~2.1s/200 steps. GELU per-step: ~2.4s/200 steps.

**Conclusion:** GELU is **consistently ~14% slower** with **identical final accuracy**. GELU's `erf()` computation is not optimised away by `torch.compile` at this batch/context size on the T4. **Keep default ReLU.**

### Positional Embedding Ablation

**Change:** Removed `pos_embed = nn.Embedding(context_size, embed_dim)` and all three `+ pos_embed(torch.arange(context_size))` additions (train loop, eval, generate). Params: 1,614,400 → 1,612,352 (−2,048 = `context_size × embed_dim`).

| Step | Accuracy (with pos_embed) | Accuracy (no pos_embed) |
|---:|---:|---:|
| 0 | 19.3% | 21.0% |
| 200 | 53.7% | 44.7% |
| 400 | 58.9% | 49.0% |
| 600 | 60.7% | 50.7% |
| 800 | 63.6% | 54.4% |
| 1000 | 64.9% | 54.6% |
| 1200 | 65.5% | 56.2% |
| 1400 | 66.6% | 56.9% |
| 1600 | 67.1% | 59.5% |
| 1800 | 67.8% | 60.0% |
| 2000 | 67.5% | 59.8% |

Training time: `21.4s` (with) vs `21.1s` (without) — negligible difference.

**Conclusion:** Removing positional embeddings costs **−7.7% accuracy** for 0.3s saved. Transformer self-attention is permutation-invariant — without positional encoding the model cannot distinguish token order, producing near-gibberish output. **Positional embeddings are essential even at `context_size=8`.**

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
