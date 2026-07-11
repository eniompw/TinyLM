# 🧪 AI Lab Notebook: Training Tiny Language Models

This notebook traces a complete journey from a basic MLP to a BPE transformer — five models, five phases, and ~65 experiments, all trained on TinyStories in under 2 minutes on a free Colab T4 GPU.

## 🗺️ The Journey at a Glance

| Phase | Model | Key Idea | Best Result |
| :--- | :--- | :--- | :--- |
| 1 | [TorchMLP.py](TorchMLP.py) | Feedforward baseline — no attention | 70.7% char-level |
| 2–3 | [SimpleTransformer.py](SimpleTransformer.py) → [TinyTransformer.py](TinyTransformer.py) | Add attention, Keller optimizations | 70.0% (genuine learning) |
| 4 | [SimpleBPE.py](SimpleBPE.py) | Swap tokenizer only — minimal code | 44.4%† in 72.9s |
| 5 | [TinyBPE.py](TinyBPE.py) | Custom vocab, scaled data | 45.9%† in 140s |

*† BPE accuracy is not comparable to character-level accuracy. Judge it by the [Generated Samples](#-generated-samples-seeing-is-believing).*

This notebook documents experiments across **five models** — from a pure MLP to a BPE transformer — each building on the last. Every model is trained on the [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset on a standard Google Colab T4 GPU, with techniques drawn from [Keller Jordan's modded-nanogpt speedrun](https://github.com/KellerJordan/modded-nanogpt) and tested at each stage.

> **Just want the results?** → [Model Leaderboard](#-the-leaderboard-model-comparison) · [Generated Samples](#-generated-samples-seeing-is-believing)  
> **Want to understand why?** → Follow the five complexity-ordered phases below, then read [The Memorization Trap](#️-the-memorization-trap).  
> **Reproducing a run?** → Each Phase header lists the exact config and the single change made.

---

## 📌 Contents

- [🔬 Scientific Controls](#-scientific-controls)
- [🧬 Model Lineage](#-model-lineage-from-mlp-digits-to-tinytransformer)
- [🧠 Phase 1: TorchMLP Optimisation](#-phase-1-torchmlp-optimisation-adamw-context-capacity-data)
- [🔀 Phase 2: Transformer Baseline Comparison](#-phase-2-transformer-baseline-comparison)
- [🔧 Phase 3: Character-Level Transformer Experiments](#-phase-3-character-level-transformer-experiments)
- [⚡ Phase 4: SimpleBPE Baseline and AMP](#-phase-4-simplebpe-baseline-and-amp)
- [🚀 Phase 5: BPE Transformer Experiments](#-phase-5-bpe-transformer-experiments)
- [🔧 Optimization Stack](#-the-optimization-stack-simpletransformer--tinytransformer)
- [🔧 SimpleTransformer Baseline Tuning](#-simpletransformer-baseline-tuning-character-level)
- [📊 Model Leaderboard](#-the-leaderboard-model-comparison)
- [⚠️ The Memorization Trap](#️-the-memorization-trap)
- [🔬 Ablation & Experiment Summary](#-ablation--experiment-summary)
- [📈 Step-by-Step Accuracy Data](#-step-by-step-accuracy-data)
- [📝 Experiment & Ablation Details](#-experiment--ablation-details)
- [📖 Generated Samples](#-generated-samples-seeing-is-believing)

---

## 🔬 Scientific Controls

In AI it's very easy to fool yourself. Three rules keep our experiments valid:

- **🎲 Training Seed (`torch.manual_seed`):** Neural networks start with random weights. We fix the seed so experiments are reproducible.
- **🎯 Eval Seed:** We evaluate on the same fixed 4,096 stories every 200 steps (using a dedicated `eval_rng`), eliminating accuracy wobble from random sampling.
- **✂️ One Change at a Time:** If we add a layer AND double the batch size and the model improves, we won't know which caused it.

> ⚠️ **The Colab Lottery:** Google Colab assigns T4 GPUs from a shared pool — sometimes fast, sometimes slow. To avoid "hardware luck" tainting timing results, we report **Relative Speed Ratios**: the 2-Layer Baseline is the Control (1.0×). If an experiment takes twice as long, its speed is **2.0×**. This ratio holds true across any GPU.
>
> *Note: This affects speed, not accuracy — a study of 65 runs across 6 GPU/TPU types found only ~0.05%–0.3% accuracy variance between runs (T4 stdev ≈ 0.05%).*

> ⚠️ **Accuracy is not comparable across tokenizers.** A character-level model predicts 1 of 65 tokens; a BPE model predicts 1 of 50,257. From Phase 4 onwards, lower BPE accuracy does **not** mean a worse model. Always judge quality by the generated samples.

---

## 🧬 Model Lineage: From MLP-Digits to TinyTransformer

[TinyTransformer.py](TinyTransformer.py) evolved through three generations, each building directly on the last.

| | `MLP-Digits-Classifier` | `TorchMLP` | `TinyTransformer` |
| :--- | :--- | :--- | :--- |
| **Inspiration** | — | Based on MLP-Digits | Built on TorchMLP |
| **Framework** | scikit-learn | PyTorch | PyTorch |
| **Architecture** | 2-layer MLP on MNIST digits | 3-layer MLP + embeddings | 2-layer transformer + attention |
| **Optimizer** | LBFGS (sklearn default) | SGD | AdamW + cosine LR + GradScaler |
| **Custom forward** | No | Yes (embed + flatten) | Yes (full transformer loop) |
| **`torch.compile`** | No | No | Yes |

The story starts with [MLP-Digits-Classifier](https://github.com/eniompw/MLP-Digits-Classifier), a minimal scikit-learn MLP on MNIST. `TorchMLP` reimplemented the same idea in pure PyTorch, adding `nn.Module`, `nn.Embedding`, and a proper training loop. `TinyTransformer` layered attention on top. See [TinyTransformer-explained.md](TinyTransformer-explained.md) for the full walkthrough.

### What TinyTransformer Inherited Unchanged

- `embed_dim = 256`, `torch.manual_seed(42)`, `batch_size = 1024`
- 2001 training steps, evaluated every 200 steps
- Automatic device selection via `torch.set_default_device(...)`
- The same `load_tinystories(...)` data pipeline and sliding-window generation loop

### What Changed in the Transition from TorchMLP

Only two hyperparameters changed:

| Setting | TorchMLP | TinyTransformer |
| :--- | :--- | :--- |
| `context_size` | 4 | 8 |
| `num_stories` | 200 | 1000 |

Everything else that's new — 2-layer encoder (4 heads, `ffn_dim=1024`), `torch.compile`, float16 autocast + `GradScaler`, fused AdamW, `zero_grad(set_to_none=True)`, cosine LR (`eta_min=1e-4`), gradient clipping (`1.0`), and inference temperature (`0.7`) — came from [Keller Jordan's modded-nanogpt speedrun](https://github.com/KellerJordan/modded-nanogpt).

### 🔗 The Keller Jordan Influence

| Feature | In TinyTransformer? | Origin |
| :--- | :--- | :--- |
| `torch.compile` | ✅ Yes | Keller record #1 |
| `AdamW betas=(0.9, 0.95)` | ✅ Yes | llm.c baseline, refined by Keller |
| `fused=True` optimizer | ✅ Yes | Keller training loop |
| `float16` mixed precision | ✅ Yes | Keller record #10 |
| `CosineAnnealingLR` + `eta_min=1e-4` | ✅ Yes | Keller record #19 |
| Pre-LN (`norm_first=True`) | ✅ Yes | Keller modernized architecture |
| `bfloat16` | ❌ Tried, failed | T4 has no native bfloat16 hardware |
| Flash Attention | ❌ Tried, marginal | Model too small to benefit |
| Muon optimizer | ❌ Not tried | Too complex for educational scope |
| RoPE embeddings | ❌ Not tried | Learned positional embeddings kept for clarity |

> 🔬 A technique that wins at GPT-2 scale isn't guaranteed to help a 2M-parameter model trained in two minutes on a T4. Every Keller-lineage feature above was re-tested here via a dedicated ablation.

---

## 🧠 Phase 1: TorchMLP Optimisation (AdamW, Context, Capacity, Data)

*Goal: Start with [TorchMLP.py](TorchMLP.py), the pure feedforward ancestor of TinyTransformer, before introducing attention.*

*Baseline: SGD lr=0.5, ctx=4, hidden=150, batch=1024, 200 stories, 2001 steps → 62.3%.*

> ⚠️ **Eval method changed mid-phase.** Early runs used full-dataset eval. From `embed_dim=512` onward (OOM risk), eval switched to a fixed 4096-sample subset with `torch.Generator(device='cuda')`. Accuracy numbers before and after this boundary are not directly comparable.

### Experiment Log

| Experiment | Change | Acc | Time | s/200 steps | Verdict |
| :--- | :--- | ---: | ---: | ---: | :--- |
| Baseline (SGD, lr=0.5) | — | 62.3% | 4.1s | ~0.4s | Control |
| AdamW, lr=1e-3 | optimizer swap | 62.3% | 4.8s | ~0.5s | ❌ No gain at baseline scale |
| + stories=1000, ctx=8 | more data + context | 64.2% | 7.3s | ~0.7s | ✅ +1.9% |
| + batch=2048, n_steps=3001 | larger batch + more steps | 66.1% | 11.0s | ~0.7s | ✅ +1.9% |
| + hidden=512, n_steps=5001 | more capacity | 71.8% | 38.5s | ~1.3s | ✅ +5.7% — hidden was the bottleneck |
| hidden=1024, n_steps=3001 | double hidden | 72.8% | 40.2s | ~2.5s | ⚠️ +1% but 2× slower/step |
| hidden=512, n_steps=5001 (reconfirm) | sweet spot check | 71.8% | 38.5s | ~1.3s | ✅ 512 confirmed as optimal |
| embed_dim=512 | wider embeddings | OOM | — | — | ❌ embed_dim is not the bottleneck |
| ctx=16, hidden=512, n_steps=5001 | wider context | 75.5% | 38.7s | ~1.3s | ✅ +3.7% — context is king |
| ctx=32, hidden=512, n_steps=5001 | double context | 75.6% | 84.0s | ~3.3s | ❌ Same acc, 2.2× slower |
| ctx=16, hidden=512, n_steps=8001 | find plateau | **77.9%** ⭐ | 66.7s | ~1.6s | ✅ Peak at step 7800 — memorizing |
| ctx=16, hidden=512, stories=5000, n_steps=8001 | break memorization | 70.7% | 66.9s | ~1.6s | ✅ Genuinely learning |

### Step-by-Step Data

#### ctx=16, hidden=512, stories=1000, n_steps=8001 (Peak Raw Score)

| Step | Loss | Acc | Time |
| ---: | ---: | ---: | ---: |
| 0 | 4.1329 | 19.3% | 0.0s |
| 2000 | 0.9618 | 71.6% | 16.5s |
| 4000 | 0.7999 | 74.8% | 33.4s |
| 6000 | 0.7741 | 76.2% | 50.2s |
| 7800 | 0.7128 | **77.9%** ⭐ | 65.0s |
| 8000 | 0.7648 | 77.3% | 66.7s |

#### ctx=16, hidden=512, stories=5000, n_steps=8001 (Genuine Learning)

| Step | Loss | Acc | Time |
| ---: | ---: | ---: | ---: |
| 0 | 4.2901 | 19.0% | 0.0s |
| 2000 | 1.0472 | 67.7% | 16.7s |
| 5000 | 0.9608 | 69.7% | 42.0s |
| 8000 | 0.9333 | 70.4% | 66.9s |

### Key Findings

> 💡 **`hidden_dim` was the biggest single lever (+5.7%).** The original `hidden=150` was severely undersized — the MLP input is `context_size × embed_dim` features wide, so the hidden layer was a dramatic bottleneck. Doubling to `hidden=512` unlocked the capacity. `hidden=1024` gave only +1% more but cost 2× per step.

> 💡 **`context_size=16` beats `context_size=32` for the MLP.** Transformer attention scales efficiently with context; the MLP input layer size is `context_size × embed_dim`, so doubling context doubles the first linear layer. ctx=32 matched ctx=16's accuracy but took 2.2× longer per step — not worth it.

> 💡 **The memorization trap hits the MLP exactly as hard as the transformer.** On 1k stories the MLP peaks at 77.9% then wobbles. On 5k stories it plateaues at 70.4% with a smooth, stable curve.

> 💡 **Pure MLP matches the transformer at the same data/context scale.** Both architectures hit ~70% on 5k stories with ctx≈16–32. Attention adds no measurable advantage once the dataset is large enough to prevent memorization — at this tiny scale, the information bottleneck dominates everything.

### Canonical [TorchMLP.py](TorchMLP.py) Config

```python
num_stories  = 5000
context_size = 16
embed_dim    = 256
hidden_dim   = 512
batch_size   = 2048
lr           = 1e-3
n_steps      = 8001
# eval: fixed 4096-sample subset, torch.Generator(device='cuda')
```
**Result: 70.7% in 66.9s** — 8.4 percentage points above the original 62.3% baseline, using a pure two-linear-layer MLP with no attention.

---

## 🔀 Phase 2: Transformer Baseline Comparison

The MLP learning curve is now established in Phase 1. This comparison isolates the architectural transition: adding attention to the character-level baseline.

*Control: [TinyTransformer.py](TinyTransformer.py) is a 2-layer transformer with float16 precision, ReLU activation, learned positional embeddings, and an 8-character context window.*

| Model | Best Accuracy | Steps |
| :--- | ---: | ---: |
| [TinyMLP.py](TinyMLP.py) | 59.4% | 2000 |
| [TorchMLP.py](TorchMLP.py) | 62.4% | 2000 |
| [SimpleTransformer.py](SimpleTransformer.py) | 67.2% | 2000 |
| [TinyTransformer.py](TinyTransformer.py) (2 layers) | 68.4% | 2000 |

### First Transformer Learning Curves
*Goal: Does a basic transformer beat the character-level [SimpleTransformer.py](SimpleTransformer.py) baseline?*

| Step | SimpleTrans | **2L (Baseline)** |
| ---: | ---: | ---: |
| 0 | 4.0% | 19.3% |
| 200 | 53.5% | 54.8% |
| 800 | 62.4% | 63.2% |
| 1600 | 66.2% | 67.0% |
| 2000 | **67.2%** ⭐ | 67.4% |

---

## 🔧 The Optimization Stack: SimpleTransformer → TinyTransformer

[TinyTransformer.py](TinyTransformer.py)'s canonical config adds five optimizations absent from [SimpleTransformer.py](SimpleTransformer.py). Each costs 1–2 lines of code:

| Component | [SimpleTransformer.py](SimpleTransformer.py) | [TinyTransformer.py](TinyTransformer.py) | Accuracy Impact | Speed Impact | Proven By |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`torch.compile`** | ✅ Present | ✅ Same | Neutral | ~1.2× faster (after ~32s one-time compile tax) | Phase 3.7: Cold vs Warm ablation |
| **float16 autocast** | ❌ float32 | ✅ `torch.autocast` on forward + eval | Neutral | Major — halves memory bandwidth; enables batch=1536 + ctx=32 in <2 min | bfloat16 ablation: 4.2× slower for +0.2% |
| **`CosineAnnealingLR`** | ❌ Flat LR | ✅ `CosineAnnealingLR(T_max=n_steps, eta_min=1e-4)` | Smooths final convergence | Negligible | Phase 3.4: warmup on top gained only +0.6% |
| **AdamW** | ❌ `Adam(params, lr)` | ✅ `AdamW(..., betas=(0.9, 0.95), weight_decay=0.01, fused=True)` | Neutral; `weight_decay` stops repetitive output | `fused=True` speeds up GPU optimizer kernel | Experiment #9: `weight_decay` acts as grammar regularizer |
| **Fixed `eval_rng`** | ❌ Full dataset eval | ✅ Dedicated `eval_rng`, 4096-sample subset | Eliminates accuracy wobble | Faster per-eval | Scientific Controls section |
| **Inference temperature** | 0.7 (hardcoded) | 0.5 (parameterized) | N/A | N/A | Eliminates invented words ("throbe" → "robe") |

> 💡 **The takeaway:** `torch.compile` + `float16` are the **speed engine**. `CosineAnnealingLR` + `AdamW`/`weight_decay` are the **quality polish**. `eval_rng` is the **scientific control**. All five originate from Keller Jordan's modded-nanogpt, but each earned its place only after local ablations confirmed it at this tiny scale.

---

## 🔧 Phase 3: Character-Level Transformer Experiments

This phase introduces attention while retaining character tokens. It starts with the minimal [SimpleTransformer.py](SimpleTransformer.py) baseline, then scales up through the detailed TinyTransformer experiments below.

### SimpleTransformer Baseline Tuning (Character-Level)

*Goal: Squeeze maximum accuracy from [SimpleTransformer.py](SimpleTransformer.py) within a 2-minute Colab budget, changing only hyperparameters — zero new lines of code.*

*Baseline: 2L, ctx=8, 200 stories, batch=1024, lr=1e-3 → 67.2% at step 2000, 36.6s*

| Change | Acc | Time | Verdict |
| :--- | ---: | ---: | :--- |
| Baseline (original) | 67.2% | 36.6s | Control |
| 3L + batch=1536 + lr=2e-3 + 1000 stories | 67.3% | 36.6s | ✅ Matches baseline, no extra time |
| + ctx=32 + 5000 stories + eval subsample | 66.6% | 151.1s | ✅ Much cleaner text, no memorisation |
| + weight_decay=0.01 | 46.0% | 200.8s | ❌ Crushes 128d model — too small for regularisation |
| + embed_dim=256, ffn=512 (1.6M params) | ~63% | >240s | ❌ Over budget — compile tax ~28s alone |

**New canonical [SimpleTransformer.py](SimpleTransformer.py) config:** `3L, ctx=32, 5000 stories, batch=1536, lr=2e-3, n_steps=1801, temp=0.5` → **~66.6% in ~151s**

> 💡 **`weight_decay` doesn't transfer.** At `embed_dim=128` with only 420K params, `weight_decay=0.01` collapses accuracy to ~46% — the regularisation overwhelms a model this small.

> 💡 **Eval OOM fix.** With `ctx=32` and 5000 stories, evaluating on the full dataset tries to allocate ~62GB. Fix: subsample 4096 rows with a fixed seed (`manual_seed(0)`) — eliminates OOM and stabilises the accuracy curve.

> 💡 **Capacity ceiling confirmed at 420K params.** Accuracy plateaued at ~66–67% from step 1600 onward regardless of further steps. Breaking it requires graduating to [TinyTransformer.py](TinyTransformer.py)'s optimizer stack (AdamW + cosine LR + float16).

---

## ⚡ Phase 4: SimpleBPE Baseline and AMP

This phase keeps the minimal transformer style, but introduces subword BPE tokenization and mixed-precision training.

### [SimpleBPE.py](SimpleBPE.py) — Minimal BPE Baseline

*Goal: Prove the tokenizer swap alone (character → BPE) improves text quality, using [SimpleTransformer.py](SimpleTransformer.py)'s minimal code style with zero architectural changes.*

| Model | Params | Data | Context | Vocab | d_model / heads / layers / FFN | Optimizer | Steps | Time | Loss | Fixed train-sample accuracy |
|---|---:|---:|---:|---:|---|---|---:|---:|---:|---:|
| SimpleBPE | 1,429,536 | 5,000 TinyStories | 32 BPE tokens | 4,000 | 128 / 4 / 3 / 256 | AdamW, lr=2e-3 | 1,801 | 133.5s | 2.6078 | 42.6% |

### Run Notes

- Seed: `torch.manual_seed(0)`. Evaluation: one fixed random sample of 4,096 contexts, created before training.
- Architecture: 3-layer PyTorch `TransformerEncoder`, Pre-LN (`norm_first=True`), zero dropout.
- Optimizer: `AdamW(params, lr=2e-3)`; PyTorch default `weight_decay=0.01`.
- The reported metric is sampled training-context next-token accuracy — **not** held-out validation accuracy. It reliably shows within-run learning progress only.
- Float32 throughput: 13.5 steps/s. AMP + GradScaler throughput: 24.7 steps/s warm-compiled.
- Learning curve: 0.5% → 30.6% → 39.4% → 42.6% at steps 0, 200, 1200, 1800.
- Qualitative sample: coherent short-story syntax, but semantic errors remain (e.g., "bouncy glass").

> ⚠️ **Environment note:** Record the GPU model, PyTorch/CUDA versions, and whether `torch.compile` was enabled alongside future benchmark results. A seed improves repeatability but does not guarantee identical results across platforms or CUDA versions.
>
> Timing convention: always report both cold total time (includes `torch.compile`) and warm-compiled time; never compare them as if they were the same measurement.

### Optimisation Attempt: batch=2048 + cosine LR, n_steps=1601

| Step | Loss | Acc | Time |
| ---: | ---: | ---: | ---: |
| 0 | 8.5487 | 2.2% | 0.2s |
| 200 | 3.5808 | 31.0% | 24.5s |
| 800 | 2.8120 | 39.3% | 94.2s |
| 1200 | 2.6165 | 42.4% | 141.2s |
| 1600 | 2.6645 | **42.9%** ⭐ | 187.6s |

**Training time: 187.6s** *(over budget)*

> 💡 **batch=2048 is over budget on SimpleBPE.** Unlike [TinyBPE.py](TinyBPE.py) which uses float16, [SimpleBPE.py](SimpleBPE.py) runs float32 — so batch=2048 adds ~50% wall-clock time per step for only +0.7% accuracy. Not worth it without float16.

> 💡 **Cosine LR smoothed the tail** (step 1600 loss: 2.66 flat → 2.58 cosine), but the gain was absorbed by the budget overrun.

> 💡 **The tokenizer swap is the story.** Same architecture, same training loop, one import changed — yet generated text jumps from broken clauses to coherent multi-sentence paragraphs. The ~42% vs ~67% raw accuracy gap is meaningless; BPE predicts 1 of 4,000 tokens vs 1 of 65 characters.

---

### AMP Optimisation

*Goal: Test whether float16 mixed precision makes SimpleBPE faster within its minimal-code philosophy.*

*Control: float32, batch=1536, flat AdamW lr=2e-3, 1801 steps → 42.6% in 133.5s.*

### 1. AMP + GradScaler + batch=2048

*Combined change: float32 → float16 AMP + GradScaler; batch 1536 → 2048. Does not isolate the separate effects of each.*

| Step | Loss | Acc | Time |
| ---: | ---: | ---: | ---: |
| 0 | 8.8046 | 0.3% | 0.1s |
| 200 | 3.5113 | 31.1% | 7.9s |
| 800 | 2.8266 | 38.9% | 32.3s |
| 1200 | 2.6808 | 40.7% | 48.8s |
| 1800 | 2.5454 | **44.4%** ⭐ | 72.9s |

**Training time: 72.9s** *(warm-compiled)*

> 💡 **New SimpleBPE best.** float16 AMP + batch=2048 reaches **44.4%** vs **42.6%** for the float32/batch-1536 control, in **1.83× less time** (72.9s vs 133.5s).

### 2. Ablation: Remove GradScaler

*Single change from above: replace scaled backward/step with standard `loss.backward()` + `optimizer.step()`. Everything else unchanged.*

| Step | Loss | Acc | Time |
| ---: | ---: | ---: | ---: |
| 0 | 8.8046 | 0.3% | 0.1s |
| 800 | 2.8601 | 38.7% | 31.9s |
| 1800 | 2.5371 | **43.3%** ⭐ | 70.7s |

**Training time: 70.7s** *(warm-compiled)*

> 💡 **GradScaler is worth keeping.** Removing it saves only **2.2s** but drops final accuracy by **1.1 percentage points** (44.4% → 43.3%). Gradient scaling protects small float16 gradients from underflow.

### ✅ SimpleBPE Verdict

| Variant | Precision | Batch | GradScaler | Acc | Warm time | Verdict |
| :--- | :--- | ---: | :--- | ---: | ---: | :--- |
| Original baseline | float32 | 1536 | No | 42.6% | 133.5s | Control |
| Float32 + cosine | float32 | 2048 | No | 42.9% | 187.6s | ❌ Over budget |
| AMP, no scaler | float16 | 2048 | No | 43.3% | **70.7s** | ⚠️ Fastest, lower accuracy |
| **AMP + GradScaler** | **float16** | **2048** | **Yes** | **44.4%** | 72.9s | 🏆 Best SimpleBPE |

**New canonical [SimpleBPE.py](SimpleBPE.py) config:** `float16 AMP, GradScaler, batch=2048, lr=2e-3, n_steps=1801` → **44.4% in 72.9s warm-compiled.**

---

## 📊 The Leaderboard: Model Comparison

*Best configuration for each architecture tested.*

> ⚠️ **Accuracy is not comparable across tokenizer types.** BPE rows (marked †) predict from a vocabulary of 4,000 or 50,257 tokens — not 65 characters. Do not compare † rows to non-† rows. Judge BPE quality by the generated samples.

### Character-Level Models

| Model | Best Accuracy | Steps | Relative Speed (vs 2L Baseline) |
| :--- | ---: | ---: | ---: |
| [NameSLP.py](NameSLP.py) | 39.6% | 2000 | 1.8× |
| [TinyMLP.py](TinyMLP.py) | 59.4% | 2000 | 0.2× |
| [TorchMLP.py](TorchMLP.py) | 62.4% | 2000 | 0.2× |
| [SimpleTransformer.py](SimpleTransformer.py) | 67.2% | 2000 | 1.8× |
| **[TinyTransformer.py](TinyTransformer.py) (2 layers)** 🥇 | **68.4%** | **2000** | **1.0× (Control)** |
| [TinyTransformer.py](TinyTransformer.py) (context=64) | 68.5% | 1800 | 10.0× |
| [TinyTransformer.py](TinyTransformer.py) (Narrow-Deep 4L, 810K params) | 68.9% | 2400 | 3.5× |
| [TinyTransformer.py](TinyTransformer.py) (Efficient-Deep 4L, ffn=512) | 70.8% | 2000 | 2.3× |
| [TinyTransformer.py](TinyTransformer.py) (Balanced Narrow-Deep 4L, 192d) | 70.8% | 2400 | 2.9× |
| [TinyTransformer.py](TinyTransformer.py) (3 layers, Wider FFN=2048) | 71.8% | 2200 | 3.0× |
| [TinyTransformer.py](TinyTransformer.py) (3 layers, batch=1024, lr=2e-3) | 72.4% | 2200 | 2.5× |
| [TinyTransformer.py](TinyTransformer.py) (4 layers) | 73.1% | 3400 | 4.0× |
| [TinyTransformer.py](TinyTransformer.py) (3 layers) | 73.5% | 2200 | 1.5× |
| [TinyTransformer.py](TinyTransformer.py) (3L, ctx=16, 5000 stories) 🧠 | 71.7% | 2200 | ~2.5× |
| [TinyTransformer.py](TinyTransformer.py) (3L, ctx=32, 5000 stories, batch=1536) 👑 | 70.0% | 1600 | ~3.2× |
| [TinyTransformer.py](TinyTransformer.py) (3L, ctx=32, 5000 stories, warmup+clip) | 70.7% | 1600 | ~3.3× |
| [TinyTransformer.py](TinyTransformer.py) (3L, ctx=32, 5000 stories, batch=2048) | 70.5% | 1600 | ~4.8× |
| [TinyTransformer.py](TinyTransformer.py) (3L, ctx=32, 5000 stories, 8 heads) | 70.5% | 1800 | ~4.6× |
| [TinyTransformer.py](TinyTransformer.py) (3L, ctx=32, 5000 stories, embed=320) | 70.5% | 1600 | ~5.5× |
| **[TinyTransformer.py](TinyTransformer.py) (3L, batch=2048)** | **76.1%** | **2200** | **~3.5×** |

### BPE Models †

*Accuracy marked † is not comparable to character-level rows above. Judge quality by [Generated Samples](#-generated-samples-seeing-is-believing).*

| Model | Best Accuracy† | Steps | Timing |
| :--- | ---: | ---: | ---: |
| **[SimpleBPE.py](SimpleBPE.py) (AMP + GradScaler, batch=2048)** ⚡ | **44.4%** | **1801** | Warm: 72.9s; cold: 105.3s |
| [TinyTransformer.py](TinyTransformer.py) (3L, ctx=32 BPE tiktoken, 5000 stories) 🚀 | 50.9% | 1800 | ~5.3× |
| [TinyTransformer.py](TinyTransformer.py) (3L, ctx=32 BPE tiktoken, batch=2048) 🏆 | 50.0% | 1200 | ~5.0× |
| [TinyTransformer.py](TinyTransformer.py) (3L, custom BPE vocab=4000, batch=2048) ⚡ | 46.2% | 900 | ~2.6× |
| [TinyBPE.py](TinyBPE.py) (3L, custom BPE vocab=4000, n_steps=1001) 🏆 | ~47% | 1001 | ~2.7× |
| [TinyBPE.py](TinyBPE.py) (3L, custom BPE vocab=4000, 10k stories, n_steps=1201) 🏆 | ~45.9% | 1201 | ~3.2× |
| [TinyBPE.py](TinyBPE.py) (4L, custom BPE vocab=4000, 10k stories, n_steps=801) | ~44.3% | 801 | ~124s |
| [TinyBPE.py](TinyBPE.py) (3L, custom BPE vocab=4000, 10k stories, ctx=64, n_steps=401) | ~37.7% | 401 | ~121s |

---

## ⚠️ The Memorization Trap

Look at the leaderboard. Why does accuracy *drop* to ~70% after hitting 76.1%? Because we expanded the dataset from 1,000 to 5,000 stories. The 76.1% model was **cheating** — it memorized the evaluation set. The ~70% models stopped memorizing and actually learned English.

> **Lower accuracy score = higher real-world intelligence.**

This is the single most important insight in this document. On small datasets, high accuracy is an illusion. Always check your generated samples, not just the numbers.

---

## 🔬 Ablation & Experiment Summary

All tests below are single changes made to the 2-layer TinyTransformer baseline (~68% accuracy, 1.0× speed).

### 🏗️ Architecture (Shape & Size)

| Type | Change Tested | Accuracy Δ | Speed Δ | The Verdict |
| :--- | :--- | ---: | ---: | :--- |
| **Exp** | **Depth:** 2 → 3 layers | +5.1% | 1.5× slower | ✅ Best speed/accuracy tradeoff |
| **Exp** | **Depth:** 2 → 4 layers | +1.2% | 2.2× slower | ✅ Worth it if you have the time (73.1% at 3400 steps) |
| **Exp** | **Shape:** Wide/Short → Narrow/Deep | +1.0% | 20% slower | ✅ Depth beats width, even at half the parameters |
| **Exp** | **Narrow-Deep Alt.** (128d, 4L) | +0.5% | 3.4× slower | ⚠️ Half the params, competitive accuracy |
| **Exp** | **Efficient-Deep** (256d, ffn=512, 4L) | +2.4% | 2.3× slower | ⚠️ Strong mid-training but peaks early |
| **Exp** | **Balanced Narrow-Deep** (192d, 4L) | +2.4% | 2.8× slower | ⚠️ Ties Efficient-Deep but takes longer |
| **Exp** | **Wider FFN** (3L, ffn=2048) | +3.4% | 3.0× slower | ⚠️ Bigger MLP helps, but not enough to beat standard 3L |
| **Exp** | **Heads:** 4 → 8 | +0.5% | ~1.4× slower | ❌ Same ceiling, more overhead |
| **Exp** | **Width:** embed_dim 256 → 320 | +0.5% | 1.7× slower | ❌ 35% more params, zero gain — capacity is not the bottleneck |
| **Abl** | **Remove Positional Embeddings** | −7.7% | Negligible | ❌ Without this, the model reads sentences as "word soup" |

### ⚡ Training & Speed Hacks

| Type | Change Tested | Accuracy Δ | Speed Δ | The Verdict |
| :--- | :--- | ---: | ---: | :--- |
| **Exp** | **`torch.compile`** (Cold vs Warm) | Neutral | ~1.2× faster overall (one-time ~32s tax) | ✅ Always warm up before timing |
| **Exp** | **Precision:** float16 → bfloat16 | +0.2% | 4.2× slower | ❌ T4 has no native bfloat16 hardware |
| **Exp** | **Weight Tying** (sharing layers) | −3.0% | Neutral | ❌ Confused the model and hurt accuracy |
| **Exp** | **Activation:** ReLU → GELU | Neutral | 14% slower | ❌ Too math-heavy for no gain at this scale |
| **Exp** | **Loss:** Last-word vs Full-sequence | Neutral | 1.47× slower | ⚠️ Learns faster early on, but hits the same ceiling |
| **Exp** | **Flash Attention** + Context 32 | +0.2% | 3.2× slower | ⚠️ Works correctly, but model too small to benefit |
| **Exp** | **LR Warmup** (50 steps) + **Grad Clipping** (1.0) | +0.6% peak | Slightly slower | ⚠️ Smoother curve, not worth the extra code at this scale |

### 🧠 The "Real Intelligence" Push (Batch, Context & Data)

| Type | Change Tested | Accuracy Δ | Speed Δ | The Verdict |
| :--- | :--- | ---: | ---: | :--- |
| **Exp** | **High LR Fast Convergence** (batch=1024, lr=2e-3) | +4.0% | 2.5× slower | ⚠️ Faster, but high LR makes training unstable |
| **Exp** | **Middle Ground** (batch=1536) | +6.8% | 2.7× slower | ✅ Excellent compromise. ~1 min runtime |
| **Exp** | **Large Batch + High LR** (batch=2048) | +7.7% | ~3.5× slower | ✅ Huge raw accuracy win — but memorizes (see Memorization Trap) |
| **Exp** | **Dataset Size:** 1k → 5k stories | −4.7% | Negligible | ✅ Drops raw acc, but drastically improves grammar |
| **Exp** | **Context Size:** 8 → 16 (large dataset) | −1.5% | ~1.5× slower | ✅ Fixes pronoun/gender swapping |
| **Exp** | **Weight Decay:** 0 → 0.01 | Neutral | Negligible | ✅ Acts as a grammar regularizer — stops lazy repetition |
| **Exp** | **Context Size:** 16 → 32 (large dataset) | −1.6% | ~1.3× slower | ✅ Fixes 90% of pronoun swaps. The best 2-min tradeoff |
| **Exp** | **batch=2048 on 5k stories** (ctx=32) | +0.5% vs 1536 | 1.5× slower | ❌ Same ~70% ceiling — batch stops helping when genuinely learning |
| **Exp** | **Inference Temp:** 0.7 → 0.5 | N/A | N/A | ✅ Eliminates invented words (e.g., "throbe" → "robe") |
| **Exp** | **BPE Tokenization** (tiktoken gpt2, vocab=50257) | See Phase 5.1 | ~5.3× slower | ✅ **Breaks the 70% character ceiling** — full paragraphs & dialogue |
| **Exp** | **BPE + Larger Batch** (batch=2048, 1401 steps) | −0.9% vs Phase 5.1 | ~1.05× faster/step | ✅ Reaches ~50% in fewer steps. Best short-budget BPE variant |
| **Exp** | **BPE + Weight Tying** | N/A | Neutral | ❌ Loss=254 at step 0 — init instability at vocab=50k. Reverted |
| **Exp** | **Custom BPE** (vocab=4000, trained on TinyStories) | −3.8% vs tiktoken BPE | **~2× faster** | ✅ 4.43M params (vs 28M), fits in 103.6s. Best size/speed tradeoff |
| **Exp** | **Logit Softcapping** (±15, Gemma 2 style) | Neutral | Negligible | ✅ Stable training, no NaN issues |
| **Exp** | **Dataset Size:** 5k → 10k stories (BPE) | −0.9% raw | Negligible | ✅ Same memorization-trap pattern. Better quality, lower score |

---

## 📈 Step-by-Step Accuracy Data

*We split data into Phases to tell the story of our experiments. ⭐ marks peak accuracy. 📉 shows overfitting.*

**Legend:** **2L/3L/4L** = TinyTransformer with 2, 3, or 4 layers | **ND** = Narrow-Deep | **FFN** = Feed-Forward Network width

---

### Phase 3.1: Shape & Size Experiments
*Goal: Does adding layers, widening the model, or changing its shape beat the 2L Baseline?*

| Step | **3L** (Run 1) | **4L** | ND 4L (128d) | Eff. Deep 4L | Bal. ND 4L (192d) | Wider FFN 3L |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 19.3% | 19.3% | 10.6% | 5.2% | 19.3% | 20.2% |
| 800 | 64.8% | 64.6% | 63.0% | 63.9% | 65.6% | 64.7% |
| 1600 | 67.6% | 68.0% | 67.8% | 68.4% | 70.0% | 70.2% |
| 2200 | **73.5%** ⭐ | - | 68.1% 📉 | 69.7% 📉 | 70.4% | **71.8%** ⭐ |
| 2400 | 71.7% 📉 | - | 68.9% | - | **70.8%** ⭐ | - |
| 3400 | - | **73.1%** ⭐ | - | - | - | - |

> 💡 **Overfitting in action:** The 3-layer model hits 73.5% at step 2200, then drops to 71.7% at step 2400. The model memorized the training data so hard it got *worse* at new stories. Always stop at ⭐.

---

### Phase 3.2: Batch Size & Learning Rate
*Goal: Instead of changing the model's shape, what if we change HOW it learns? (3-layer model)*

| Step | High LR (batch=1024) | Mid Ground (batch=1536) | **Large Batch+LR** (batch=2048) |
| ---: | ---: | ---: | ---: |
| 0 | 19.3% | 19.3% | 19.3% |
| 800 | 62.9% | 65.5% | 66.9% |
| 1600 | 66.2% | 69.9% | 71.0% |
| 2200 | **72.4%** ⭐ | **75.2%** ⭐ | **76.1%** ⭐ |
| 2400 | 71.1% 📉 | 73.0% 📉 | - |

---

### Phase 3.3: The Real Intelligence Push (Dataset & Context)
*Goal: Stop chasing raw accuracy. Expand the dataset and context window to force the model to learn English rather than memorize 1,000 stories.*

| Step | **3L, 2048 batch, 3k stories** (ctx=8) | **3L, 1536 batch, 5k stories** (ctx=32, wd=0.01) |
| ---: | ---: | ---: |
| 0 | 18.5% | 19.2% |
| 800 | 66.0% | 64.4% |
| 1200 | 67.8% | 67.8% |
| 1600 | 69.3% | **70.0%** ⭐ |
| 2000 | 71.4% | - |

> 💡 Scores here are *lower* than Phase 3.2's 76.1%, but the generated text is dramatically better. See [The Memorization Trap](#️-the-memorization-trap).

---

### Phase 3.4: Optimizer Stability (Warmup & Gradient Clipping)
*Goal: Do standard stability techniques help enough at this scale to justify the extra code?*

*Changes added to the Phase 3.3 canonical config (3L, ctx=32, 5k stories, batch=1536): 50-step linear LR warmup + `clip_grad_norm_(params, 1.0)`.*

| Step | Loss | Acc | LR |
| ---: | ---: | ---: | ---: |
| 0 | 4.5791 | 5.6% | 5.96e-05 |
| 400 | 1.3176 | 60.5% | 1.82e-03 |
| 800 | 1.1465 | 64.8% | 1.26e-03 |
| 1200 | 1.0363 | 68.0% | 5.98e-04 |
| 1600 | 0.9784 | **70.7%** ⭐ | 1.60e-04 |
| 1800 | 0.9542 | 70.5% 📉 | 1.00e-04 |

**Training time: 133.6s**

### Phase 3.4 Verdict

| Technique | Peak Acc | Time | vs Baseline (70.0% / 127.7s) | Keep? |
|:---|---:|---:|:---|:---|
| Warmup (50 steps) + Grad Clipping | 70.7% | 133.6s | +0.7%, +5.9s, more code | ⚠️ Marginal |

> 💡 Valid techniques, but marginal at this scale. The simpler no-warmup baseline (Phase 3.3) reached 70.0% faster with less code.

---

### Phase 3.5: Larger Batch on Big Dataset
*Goal: Phase 3.2 showed batch=2048 was a huge win on 1k stories. Does it still work on 5k stories and ctx=32?*

*Single change from the Phase 3.3 canonical config:* `batch_size 1536 → 2048`

| Step | Loss | Acc | Time |
| ---: | ---: | ---: | ---: |
| 0 | 4.5770 | 19.2% | 0.2s |
| 400 | 1.3018 | 62.2% | 38.8s |
| 800 | 1.1103 | 65.7% | 77.4s |
| 1200 | 1.0087 | 68.0% | 115.0s |
| 1600 | 0.9760 | **70.5%** ⭐ | 153.3s |
| 2000 | 0.9301 | 70.4% | 191.2s |

**Training time: 191.2s**

> 💡 **Peak 70.5% — same ~70% ceiling, but taking 191s vs 128s (50% more time).** On 1k stories, bigger batches accelerated memorization. On 5k stories the model is genuinely *learning*, so the bottleneck has shifted. The ceiling is a **capacity ceiling**, not an optimisation ceiling.

---

### Phase 3.6: Attempting to Break the 70% Wall
*Goal: Three different approaches — bigger batch, more attention heads, wider model — all tried to break the ~70% ceiling.*

#### 3.6a. More Attention Heads (4 → 8)
*Single change from canonical:* `n_heads 4 → 8`

| Step | Loss | Acc | Time |
| ---: | ---: | ---: | ---: |
| 0 | 4.5728 | 19.2% | 29.1s |
| 800 | 1.1245 | 65.4% | 98.2s |
| 1200 | 1.0204 | 67.8% | 131.7s |
| 1800 | 0.9118 | **70.5%** ⭐ | 182.8s |

**Training time: 182.8s**

#### 3.6b. Wider Model (embed_dim 256 → 320)
*Single change:* `embed_dim 256 → 320` (3.26M params, +35%). Required `lr=1e-3` + grad clipping to prevent NaN divergence.

| Step | Loss | Acc | Time |
| ---: | ---: | ---: | ---: |
| 0 | 4.5175 | 19.2% | 0.2s |
| 800 | 1.1253 | 65.2% | 97.8s |
| 1200 | 1.0484 | 67.7% | 146.3s |
| 1600 | 0.9944 | **70.5%** ⭐ | 194.0s |
| 1800 | 0.9337 | 70.0% | 218.0s |

**Training time: 218.0s**

### Phase 3.6 Verdict

> 💡 **The Definitive Result:** Three completely different approaches — bigger batch (+33% data/step), more heads (+100% attention patterns), wider model (+35% parameters) — all converged on **exactly 70.5%**. This is the **information ceiling** of character-level tokenization at ctx=32 (~5–6 words). No amount of model capacity can extract more signal than exists in a 5-word window.

> 💡 **The NaN lesson:** The wider model (embed=320) immediately diverged to NaN at `lr=2e-3` with no clipping. Larger embeddings produce larger gradients that destabilize the optimizer before it can warm up. Gradient clipping became *necessary* here, not optional.

---

### Phase 3.7: `torch.compile` — Cold vs Warm Start
*Goal: Measure exactly how much `torch.compile` graph compilation costs, using the Phase 3.3 canonical config (3L, ctx=32, 5k stories, batch=1536). `params: 2,414,408`.*

**Cold Start** (compilation happens inside the timed run):

| Step | Loss | Acc | Time |
| ---: | ---: | ---: | ---: |
| 0 | 4.5738 | 19.2% | 32.5s |
| 800 | 1.1545 | 64.6% | 89.7s |
| 1200 | 1.0466 | 67.7% | 119.8s |
| 1600 | 0.9977 | **70.8%** ⭐ | 148.5s |
| 1800 | 0.9138 | 69.8% 📉 | 163.2s |

**Training time: 163.2s**

**Warm Start** (model already compiled before the timer starts):

| Step | Loss | Acc | Time |
| ---: | ---: | ---: | ---: |
| 0 | 4.5772 | 19.2% | 0.2s |
| 800 | 1.1552 | 64.2% | 61.3s |
| 1200 | 1.0432 | 67.5% | 90.0s |
| 1600 | 0.9828 | 70.4% | 120.1s |
| 1800 | 0.9244 | **70.5%** ⭐ | 134.9s |

**Training time: 134.9s**

> 💡 **Step 0 tells the whole story — 32.5s (cold) vs 0.2s (warm).** That's a one-time graph-compilation tax paid only on the very first call. After that, both runs advance at nearly identical per-step speed. Total difference: **163.2s (cold) vs 134.9s (warm) — ~1.2× (17%).** The longer you train, the smaller that tax looks. Always benchmark *after* warmup.

**Generated samples:**
- **Cold start (69.8%):** `Once there was a little boy named Tim. He was scared and said, "Thank you, Mom. I want to find they inside. They did not have the park with the temple home.`
- **Warm start (70.5%):** `Once there was a little boy named Tim. He was so happy. The dog was scared and said, "I will give you so much fun. It was a sorry, but I will said, "You should not stopped...`

---

## 🚀 Phase 5: BPE Transformer Experiments

These experiments build on Phase 4 by scaling BPE models from a GPT-2 tokenizer to a custom TinyStories vocabulary.

### Phase 5.1: BPE Tokenization — Breaking the Character Ceiling
*Goal: The Phase 3.6 experiments proved the ~70% ceiling is an **information bottleneck** from character-level tokenization, not a capacity problem. The fix: swap to subword BPE tokens so ctx=32 covers ~20–25 words instead of ~5–6.*

*Single change from the Phase 3.3 canonical config:* Replace character tokenizer with `tiktoken` GPT-2 BPE (`vocab_size=50,257`). Architecture and all hyperparameters unchanged. `params: 28,159,313` (dominated by the embedding table: 50257×256 = 12.9M, plus output head: 12.9M).

> ⚠️ **Accuracy is not comparable to earlier phases.** The model now predicts 1 of 50,257 tokens instead of 1 of 65 characters. A "50.9%" here is a much harder task than "70.0%" in Phase 3.3. Judge quality by the generated sample.

| Step | Loss | Acc | Time |
| ---: | ---: | ---: | ---: |
| 0 | 11.2628 | 7.6% | 0.2s |
| 200 | 3.5206 | 35.6% | 23.3s |
| 600 | 2.7770 | 41.4% | 71.5s |
| 1000 | 2.5899 | 45.9% | 118.0s |
| 1400 | 2.2842 | 49.0% | 164.9s |
| 1800 | 2.2064 | **50.9%** ⭐ | 211.3s |

**Training time: 211.3s** *(still climbing at step 1800 — model has not peaked)*

> 💡 **The ceiling is broken.** The loss curve shows no plateau at step 1800, unlike every character-level run which flattened by step 1600. ctx=32 BPE tokens ≈ 100+ characters — roughly 20–25 words. The model can now track subject–verb agreement, character names, and dialogue turns.

> 💡 **Why 28M params?** The transformer itself is the same ~2.4M as Phase 3.3. The extra 26M comes entirely from embedding tables. Weight tying (`linear.weight = tok_embed.weight`) would halve this to ~15M — a natural next experiment.

---

### Phase 5.2: BPE Short-Budget Run & Weight-Tying Ablation
*Goal: Fit the BPE model closer to a 2-minute Colab budget.*

*Single changes from Phase 5.1:* `batch_size 1536 → 2048`, `n_steps 1801 → 1401`. `params: 28,159,313`.

| Step | Loss | Acc | Time |
| ---: | ---: | ---: | ---: |
| 0 | 11.2768 | 7.6% | 33.4s |
| 400 | 2.9462 | 40.1% | 79.5s |
| 800 | 2.5360 | 44.5% | 127.6s |
| 1200 | 2.2626 | **50.0%** ⭐ | 175.4s |
| 1400 | 2.2288 | 49.9% 📉 | 199.4s |

**Training time: 199.4s** *(compile tax alone is 33.4s — warm start would land near 166s)*

> 💡 **Weight tying fails at BPE scale.** Three separate attempts at `linear.weight = tok_embed.weight` — with and without LR warmup — all produced **Loss ≈ 254 at step 0** and never recovered. At vocab=50,257 the shared matrix receives contradictory gradients from the embedding lookup and output projection simultaneously, destabilizing initialization before training can start.

---

### Phase 5.3: Custom Small-Vocab BPE (2-Minute Champion)
*Goal: GPT-2's 50,257-token vocab is oversized for TinyStories. Train a custom BPE tokenizer on the TinyStories corpus with a much smaller vocab, cutting embedding-table size dramatically.*

*Config:* `3L, ctx=32 custom-BPE tokens, 5k stories, batch=2048, n_steps=901, lr=2e-3, cosine LR, logit softcapping (±15), warm-compiled`. Tokenizer: HuggingFace BPE trained from scratch, `vocab_size=4000`. `params: 4,429,472` (84% reduction vs GPT-2 BPE).

| Step | Loss | Acc | Time |
| ---: | ---: | ---: | ---: |
| 0 | 8.6952 | 7.4% | 0.2s |
| 300 | 3.0530 | 38.2% | 34.8s |
| 600 | 2.5971 | 43.7% | 70.2s |
| 900 | 2.4410 | **46.2%** ⭐ | 103.6s |

**Training time: 103.6s** *(well under the 2-minute budget)*

> 💡 **6.4× smaller model, comparable learning curve.** 4.43M params vs 28M for GPT-2 BPE, reaching 46.2% in roughly half the wall-clock time.

> 💡 **Logit softcapping (Gemma 2 trick) added for free.** `logits = 15.0 * torch.tanh(logits / 15.0)` bounds extreme logit values — no accuracy cost, acts as a safety net against divergence.

> 💡 **Trade-off is explicit:** smaller vocab means lower raw accuracy vs GPT-2 BPE runs, but the model is 6.4× smaller and trains 2× faster with comparably fluent output.

---

### Phase 5.4: TinyBPE Tuning (Steps, LR, Vocab)

*Goal: Three single-variable experiments on the Phase 5.3 canonical config to find the optimal 2-minute TinyBPE config.*

*Baseline (Phase 5.3):* `46.2% at step 900, 103.6s`

#### 5.4a. More Steps (901 → 1201)

| Step | Loss | Acc | Time |
| ---: | ---: | ---: | ---: |
| 900 | 2.3761 | 46.4% | 104.4s |
| 1000 | 2.3650 | 47.4% | 116.1s |
| 1200 | 2.3813 | **48.1%** ⭐ | 139.3s |

**Training time: 139.3s** *(over 2-min budget — but proves the curve hasn't peaked)*

> 💡 **Still climbing at step 901.** More steps is the only lever that meaningfully moves accuracy. Since 1201 steps overshoots the 2-minute wall, the optimal within-budget config is `n_steps=1001` (~116s).

#### 5.4b. Slower LR Tail (eta_min 1e-4 → 3e-4)

| Step | Loss | Acc | Time |
| ---: | ---: | ---: | ---: |
| 900 | 2.4091 | **46.3%** ⭐ | 104.3s |

> 💡 **No effect** (+0.1% — within noise). The cosine tail difference is only ~0.2e-3 LR at step 900, too small to influence learning at this budget.

#### 5.4c. Larger Vocab (4000 → 6000 tokens)

| Step | Loss | Acc | Time |
| ---: | ---: | ---: | ---: |
| 900 | 2.4887 | **45.6%** ⭐ | 105.8s |

> 💡 **Marginal loss (−0.6%).** The larger vocab adds 1M params but the model doesn't have enough steps to learn the extra tokens at this budget. Generated text quality was subjectively better — richer tokenisation needs more steps to pay off.

#### ✅ Phase 5.4 Verdict

| Experiment | Change | Acc | Time | Verdict |
| :--- | :--- | ---: | ---: | :--- |
| Phase 5.3 baseline | — | 46.2% | 103.6s | Control |
| **5.4a. More steps** | 901 → 1201 | **48.1%** | 139.3s | ⚠️ Over budget — sets `n_steps=1001` as new optimal |
| 5.4b. Slower LR tail | eta_min → 3e-4 | 46.3% | 104.3s | ❌ Noise |
| 5.4c. Larger vocab | 4000 → 6000 | 45.6% | 105.8s | ❌ Needs more steps |

**New canonical [TinyBPE.py](TinyBPE.py) config:** `vocab=4000, n_steps=1001, eta_min=1e-4` → Expected accuracy: **~47% at ~116s**.

> ⚠️ **Superseded by Phase 5.5.** The canonical config was updated in Phase 5.5 to use `num_stories=10000, n_steps=1201`. See [Phase 5.5 Verdict](#-phase-55-verdict) for the current default.

---

### Phase 5.5: TinyBPE Scale-Up (10k Stories, Depth, Context)

*Baseline: TinyBPE canonical config (3L, custom BPE vocab=4000, 5k stories, batch=2048, n_steps=1001, ~116s) → 46.8%†*

#### 5.5a. More Data (5k → 10k stories)
*Single change:* `num_stories 5000 → 10000`

| Step | Loss | Acc | Time |
| ---: | ---: | ---: | ---: |
| 500 | 2.8900 | 41.3% | 59.4s |
| 1000 | 2.6245 | **44.8%** ⭐ | 117.0s |

> 💡 **Raw accuracy dropped ~2%** — identical pattern to the 1k→5k char-level transition. See [The Memorization Trap](#️-the-memorization-trap). Generated text shows genuine improvement: multi-character interactions, subordinate clauses with motives, and resolved story arcs.

#### 5.5b. More Steps on 10k (1001 → 1201)
*Single change from 5.5a:* `n_steps 1001 → 1201`

| Step | Loss | Acc | Time |
| ---: | ---: | ---: | ---: |
| 1000 | 2.5719 | 45.3% | 116.9s |
| 1200 | 2.5743 | **45.9%** ⭐ | 139.8s |

**Training time: 139.8s** *(new best for 10k config)*

#### 5.5c. Slower LR Tail on 10k (eta_min 1e-4 → 3e-4)
**Peak: 45.7% at step 1200, 139.6s — no effect (−0.2% vs 5.5b, within noise)**

> 💡 The tail oscillation is **batch sampling variance**, not LR decay. With batch=2048 from 10k stories, each batch covers ~20% of the dataset — late-training loss bounces are inherent stochasticity. ❌ Don't bother.

#### 5.5d. Deeper Model (3 → 4 layers, n_steps=801)
*Single change from 5.5b:* `n_layers 3 → 4, n_steps 1201 → 801`

| Step | Loss | Acc | Time |
| ---: | ---: | ---: | ---: |
| 400 | 2.8991 | 40.3% | 63.0s |
| 800 | 2.6297 | **44.3%** ⭐ | 124.4s |

> 💡 **4L/801 steps (44.3%) does not beat 3L/1201 steps (45.9%) within the 2-minute budget.** 4L needs ~1100 steps to reach its peak but that costs 241s. Within 2 minutes, **3L is optimal.**

#### 5.5e. Longer Context (ctx 32 → 64 BPE tokens, n_steps=401)
*Single change from 5.5b:* `context_size 32 → 64, n_steps 1201 → 401`

| Step | Loss | Acc | Time |
| ---: | ---: | ---: | ---: |
| 200 | 3.3762 | 35.1% | 61.9s |
| 400 | 3.1592 | **37.7%** ⭐ | 120.7s |

> 💡 **Lower accuracy, higher story quality.** ctx=64 BPE ≈ 40–50 words — enough for story-level conclusions and scene-setting. But ~30s/100 steps means only ~400 steps fit in 2 minutes — not enough to converge. Best framed as a "what if you had 5 minutes" variant.

#### 5.5f. Larger Vocab on 10k (vocab 4000 → 6000, n_steps=1201)
*Single change from 5.5b:* `vocab_size 4000 → 6000` → `params: 5,455,472`

| Step | Loss | Acc | Time |
| ---: | ---: | ---: | ---: |
| 500 | 2.9268 | 41.4% | 80.5s |
| 900 | 2.6630 | **45.7%** ⭐ | 143.1s |
| 1200 | 2.4774 | 45.0% | 188.8s |

> 💡 **vocab=6000 underperforms vocab=4000** (45.0% vs 45.9%) and takes 35% more time (188s vs 140s). The larger vocab needs ~1600+ steps to beat vocab=4000, which costs ~250s — not budget-viable.

#### ✅ Phase 5.5 Verdict

| Experiment | Change | Acc | Time | Verdict |
| :--- | :--- | ---: | ---: | :--- |
| 5.5a. More data | 5k → 10k stories | 44.8% | 117s | ✅ Better quality, lower raw score |
| **5.5b. More steps** | 1001 → 1201 | **45.9%** ⭐ | 140s | ✅ New best — sets new canonical |
| 5.5c. Slower LR tail | eta_min → 3e-4 | 45.7% | 140s | ❌ Sampling variance, not LR |
| 5.5d. 4 layers | n_layers=4, 801 steps | 44.3% | 124s | ❌ Needs 241s to beat 3L |
| 5.5e. ctx=64 | context_size 32→64 | 37.7% | 121s | ⚠️ Better quality, too slow for full run |
| 5.5f. Larger vocab | vocab 4000→6000 | 45.0% | 189s | ❌ Needs 250s to converge |

**New canonical [TinyBPE.py](TinyBPE.py) config:** `vocab=4000, num_stories=10000, n_steps=1201, eta_min=1e-4` → Expected accuracy: **~45.9%† at ~140s**.

*† Not comparable to character-level rows — BPE predicts 1 of 4,000 tokens vs 1 of 65 characters.*

---

## 📝 Experiment & Ablation Details

### 🏗️ Theme 1: Architecture Choices (Shape, Size & Encoding)

**1. Layer Depth (2 vs 3 vs 4 layers)**
- **Result:** 3 layers hit 73.5% in 2200 steps. 4 layers hit 73.1% but took 3400 steps.
- **Takeaway:** 3 layers is the **sweet spot**. Too deep and training becomes too slow for the accuracy gain.

**2. Shape: Narrow/Deep vs. Wide/Short**
- **Change:** Halved width (`embed_dim` 256→128, `ffn_dim` 1024→512) and doubled depth (2→4 layers), cutting params from 1.6M to 0.8M.
- **Result:** Despite 50% fewer parameters, the taller model beat the baseline by +1.0%.
- **Takeaway:** Depth beats width. A deeper network has more sequential steps to refine features and generalise better.

**3. More Attention Heads (4 → 8)**
- **Result:** Peak 70.5% at step 1800, 182.8s.
- **Takeaway:** Doubling attention heads added overhead without unlocking new capacity — the ~70% ceiling is an information limit, not an attention-head bottleneck.

**4. Wider Embeddings (embed_dim 256 → 320)**
- **Result:** Peak 70.5% at step 1600, 218.0s. Required lowering `lr` to 1e-3 and adding gradient clipping.
- **Takeaway:** 35% more parameters, zero improvement. Combined with the batch and heads experiments, this definitively proves the ~70% ceiling is an **information bottleneck** (ctx=32 ≈ 5 words), not a capacity bottleneck.

> 💡 **Three approaches, one ceiling:** Bigger batch, more heads, and wider model all converged on **exactly 70.5%**. The signal ceiling of a 5-word character window cannot be overcome by scaling alone.

**5. Positional Embeddings (Ablation)**
- **Result:** Accuracy crashed by 7.7%.
- **Takeaway:** Without positional embeddings, the transformer sees all letters simultaneously with no left-to-right concept — pure "word soup." Order matters.

---

### ⚡ Theme 2: Speed, Training & Optimization Hacks

**1. Weight Tying (Parameter Sharing)**
- **Result:** Accuracy dropped 3.0% at char-level (vocab=65); starting loss exploded above 250 at BPE scale (vocab=50,257).
- **Takeaway:** Weight tying saves millions of parameters in large-vocab models — but requires careful joint initialization. In this architecture it fails at both vocab sizes.

**2. Context Size (8 vs 64 characters)**
- **Result:** Accuracy barely moved (+1.1%), but training time exploded from 25s to 197s.
- **Takeaway:** Attention math scales quadratically — 8× the context = 7.8× the time. This is exactly why Flash Attention was invented.

**3. Precision: Float16 vs Bfloat16**
- **Result:** bfloat16 was 4.2× slower on the T4.
- **Takeaway:** The T4 has no physical bfloat16 circuits — it falls back to float32 emulation. Hardware matters.

**4. Activation: GELU vs ReLU**
- **Result:** Identical accuracy, 14% slower.
- **Takeaway:** Don't use complex math if simple math works just as well at this scale.

**5. Loss: Last-Word vs. Full-Sequence Causal Loss**
- **Result:** Much faster early learning (+3.7% at step 200), but the same final ceiling (~67.6%) at 1.47× the training cost.
- **Takeaway:** Full-sequence loss (the standard GPT approach) improves sample efficiency but doesn't raise the accuracy ceiling for a small architecture.

**6. Flash/SDPA Attention + Context Scaling**
- **Result:** 2.6× faster than naive long-context, but final accuracy improved only +0.2% and training was still 3.2× slower than the 8-char baseline.
- **Takeaway:** Memory-efficient attention works, but context windows are only as useful as the model's capacity to encode them.

**7. LR Warmup + Gradient Clipping**
- **Result:** Peak improved from 70.0% → 70.7%, but the simpler baseline reached 70.0% faster with less code.
- **Takeaway:** Valid techniques — marginal benefit at this scale.

---

### 🧠 Theme 3: The "Real Intelligence" Push (Batch, Context & Data)

**1. Large Batch + High LR (batch=2048)**
- **Result:** New best raw score: **76.1% at step 2200**.
- **Takeaway:** More data per step mattered more than more parameters — at this dataset size.

**2. The Memorization Trap (1k → 5k stories)**
- **Result:** Raw accuracy dropped from 76.1% to 71.4%, but generated text improved dramatically.
- **Takeaway:** With only 1,000 stories the model sees the same eval stories so often it memorizes the answers. 5k stories forces it to learn the actual rules of English grammar.

**3. Context is King (8 → 16 → 32 characters)**
- **Result:** The model stopped swapping pronouns mid-sentence. It could finally remember "named Lily" long enough to correctly use "She" a few words later.
- **Takeaway:** 8 characters is barely 1.5 words — the model can't see the subject by the time it writes the verb. 32 characters fixes the "amnesia" while still fitting inside a 2-minute Colab run.

**4. BPE Tokenization — Breaking the Information Ceiling**
- **Result:** Loss still falling at step 1800 (no plateau), 28M params, 211s training time. Generated text shows full multi-paragraph coherence, proper dialogue, and correct pronoun tracking.
- **Takeaway:** ctx=32 BPE tokens ≈ 100+ characters, giving the model a ~20-word memory vs the previous ~5 words. The architecture didn't change — the information did.

**5. Custom Small-Vocab BPE (vocab=4000 vs 50,257)**
- **Result:** 4,429,472 params (vs 28M), 46.2% in 103.6s.
- **Takeaway:** A right-sized vocabulary cuts embedding parameters by 84% and roughly halves training time, at a modest accuracy cost. Best speed/size tradeoff in the BPE family.

---

## 🦙 [TinyLlama.py](TinyLlama.py) — Modern Architecture (Benchmarks Pending)

[TinyLlama.py](TinyLlama.py) rebuilds the transformer using Llama-style components: **RoPE** positional encoding, **RMSNorm** layer normalisation, and **SiLU** activation, plus `torch.compile`.

> 📋 **Benchmark runs are in progress.** Results will appear here once complete. Expected experiments: RoPE vs learned positional embeddings ablation, RMSNorm vs LayerNorm, SiLU vs ReLU, and a head-to-head vs [TinyTransformer.py](TinyTransformer.py) on the Phase 3.3 canonical config.

---

## 📖 Generated Samples (Seeing is Believing)

**[TinyMLP.py](TinyMLP.py) (59.4% — Letters work, words are broken)**
> `Once tichec. Ther. She said outned. Sker to. Hif even very the box. It. I mesis momors.`

**[SimpleTransformer.py](SimpleTransformer.py) (67.2% — Almost real sentences)**
> `Once there was a faster. They learned the pusiade of the yell socked up and played together.`

**[TinyTransformer.py](TinyTransformer.py) — 3L, batch=2048 (76.1% — Highest raw score, but cheats)**
> `Once there was a little girl named Lily. She saw the new toy. He liked to play with her mom smiled and said, "Hello, Spot saw Tom was very happy with the cake was so smaller saw a big that she was happy`
*(Starts well, turns into word salad — it memorized patterns, not grammar.)*

**[TinyTransformer.py](TinyTransformer.py) — 3L, 2048 batch, 3k stories (71.4% — Generalisation Win)**
> `Once there was a great time and she was green and strong. Tim and Sue were so happy that the box opened the bug friends. She was sad and looked for them.`
*(Clauses flow much better — structure learned, not just words memorized.)*

**[TinyTransformer.py](TinyTransformer.py) — 3L, 1536 batch, 5k stories, ctx=32 (70.0% — Simplicity Champion 👑)**
> `Once there was a little boy named Tim. He was so happy. The dog was scared and happy. They saw a little girl who liver seen the ball. She lived in a big branch with the ball and went to the park.`
*(Cleanest code, fastest training at 127.7s, same ~70% ceiling as more complex variants.)*

**[TinyTransformer.py](TinyTransformer.py) — 3L, BPE tiktoken, ctx=32 tokens (50.9%† — Ceiling Broken 🚀)**
> `Once there was a little boy named Jack. He was only three years old and had lots of things he wanted to do. One day he saw something very special and he wanted to take it home.`
> `His mom said, "Mom, can you have some cookies?"`
> `Sam smiled and nodded. He said, "Yes, please. I will be careful."`
*(Full multi-paragraph structure, working dialogue, consistent pronouns — the ceiling is gone.)*

**[TinyTransformer.py](TinyTransformer.py) — 3L, BPE tiktoken, batch=2048, 1401 steps (50.0%† — Short-Budget BPE Champion 🏆)**
> `Once there was a little boy named Jack. He was only three years old and had lots of things he wanted to do. One day he saw something very special and he couldn't wait to find out he was very excited.`
> `The little boy was so excited! He decided to take the track home, and soon enough he got to his mom and said, "Let's go home now!"`
*(Full paragraphs and dialogue, reached in fewer steps — best speed/quality tradeoff for BPE.)*

**[TinyTransformer.py](TinyTransformer.py) — 3L, custom BPE vocab=4000, batch=2048, 901 steps (46.2%† — 2-Min Champion ⚡)**
> `Once there was a little boy named Jack. He was only three years old and had lots of things he wanted to do. One day he saw something very special.`
> `The boy's parents had an idea. He felt so happy, and he wore a nice story together.`
> `Once upon a time, there was a little dog named Max. Max was a very good friend.`
*(Clean output, multiple story arcs, only 4.43M params — best size-to-quality ratio tested.)*

*† Raw accuracy not comparable to character-level models. See [Scientific Controls](#-scientific-controls).*