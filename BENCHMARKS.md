# 🧪 AI Lab Notebook: Training Tiny Language Models

This file tracks our training experiments on character-level language models trained on the **TinyStories** dataset.

Our baseline model is **TinyTransformer.py** — a 2-layer transformer with float16 precision, ReLU activation, learned positional embeddings, and a context size of 8 characters. All runs use a standard Google Colab T4 GPU.

> **How to read this doc:** An **experiment** adds or upgrades something to see if the model improves. An **ablation** removes something to prove it was necessary. All tests change only one thing at a time.

---

## 📌 Contents

- [🔬 The Scientific Method: How We Trust Our Data](#-the-scientific-method-how-we-trust-our-data)
- [🧬 Lineage: From MLP-Digits to TinyTransformer](#-lineage-from-mlp-digits-to-tinytransformer)
- [🔧 The Default Stack: SimpleTransformer → TinyTransformer](#-the-default-stack-simpletransformer--tinytransformer)
- [🔧 SimpleTransformer Optimisation (Character-Level Baseline)](#-simpletransformer-optimisation-character-level-baseline)
- [📊 The Leaderboard: Model Comparison](#-the-leaderboard-model-comparison)
- [⚠️ The Memorization Trap](#️-the-memorization-trap)
- [🔬 Ablation & Experiment Summary](#-ablation--experiment-summary)
- [📈 Step-by-Step Accuracy Data](#-step-by-step-accuracy-data)
- [� Phase 12: TinyBPE Optimisation](#-phase-12-tinybpe-optimisation-steps-lr-tail-vocab-size)
- [🚀 Phase 13: TinyBPE Scale-Up (10k Stories, Depth, Context)](#phase-13-tinybpe-scale-up-10k-stories-depth-context)
- [�📝 Experiment & Ablation Details](#-experiment--ablation-details)
- [📖 Generated Samples](#-generated-samples-seeing-is-believing)

---

## 🔬 The Scientific Method: How We Trust Our Data

In AI, it is very easy to fool yourself. Here are the three rules we use to make sure our experiments are scientifically valid:

- **🎲 The Starting Seed (`torch.manual_seed`):** Neural networks start with random guesses. We hardcode the seed so our experiments are **reproducible**.
- **🎯 The Eval Seed:** When we test the model every 200 steps, we don't evaluate on the whole dataset (it would run out of GPU memory). Instead, we use a dedicated `eval_rng` so the model is *always* tested on the exact same 4,096 stories — eliminating accuracy wobble from random sampling.
- **✂️ The Golden Rule:** Change **only one thing at a time**. If we add a layer AND double the batch size and the model improves, we won't know which caused it.

---

> ⚠️ **The Colab Lottery & Scientific Controls:**
> Google Colab assigns T4 GPUs from a shared pool — sometimes fast, sometimes slow. If we only look at "Total Seconds," our data is ruined by hardware luck!
>
> To fix this, we use **Relative Speed Ratios**. We run the 2-Layer Baseline as our "Control" (1.0×). If an experiment takes twice as long, its speed is **2.0×**. This ratio holds true whether you run on a slow Colab GPU or a supercomputer.
>
> *Note: This is a **speed** effect, not an **accuracy** one — a controlled study of 65 runs across 6 GPU/TPU types found only ~0.05%–0.3% accuracy/loss variance between runs (T4 stdev ≈ 0.05%). So the Colab Lottery jitters the clock, not the score.*

> ⚠️ **Accuracy is not comparable across tokenizers.**
> A character-level model predicts 1 of 65 tokens. A BPE model predicts 1 of 50,257. Raw accuracy numbers from Phase 9 onwards **cannot be directly compared** to earlier phases — lower BPE accuracy does not mean a worse model. Always judge quality by the generated samples.

---

## 🧬 Lineage: From MLP-Digits to TinyTransformer

`TinyTransformer.py` evolved through three generations — each building directly on the last. The story starts with [MLP-Digits-Classifier](https://github.com/eniompw/MLP-Digits-Classifier), a minimal scikit-learn MLP trained on handwritten digits (MNIST). That project proved the concept of a layered neural network on a toy classification task. `TorchMLP` then reimplemented the same idea in pure PyTorch — adding `nn.Module`, `nn.Embedding`, and a proper training loop. `TinyTransformer` layered attention on top of that PyTorch foundation. Understanding this lineage explains why the baseline hyperparameters look the way they do (see [TinyTransformer-explained.md](TinyTransformer-explained.md) for the full walkthrough).

| | `MLP-Digits-Classifier` | `TorchMLP` | `TinyTransformer` |
| :--- | :--- | :--- | :--- |
| **Inspiration** | — | ← Inspired by MLP-Digits | ← Built on TorchMLP |
| **Framework** | scikit-learn | PyTorch | PyTorch |
| **Architecture** | 2-layer MLP on MNIST digits | 3-layer MLP + embeddings | 2-layer transformer + attention |
| **Optimizer** | LBFGS (sklearn default) | SGD | AdamW + cosine LR + GradScaler |
| **Custom forward** | No | Yes (embed + flatten) | Yes (full transformer loop) |
| **`torch.compile`** | No | No | Yes |

### What TinyTransformer Inherited (Unchanged)

- `embed_dim = 256`
- `torch.manual_seed(42)`
- `batch_size = 1024`
- 2001 training steps, evaluated every 200 steps
- Automatic device selection via `torch.set_default_device(...)`
- The same `load_tinystories(...)` data pipeline and sliding-window generation loop

### What Actually Changed

Only two hyperparameters were adjusted in the transition from TorchMLP to TinyTransformer:

| Setting | TorchMLP | TinyTransformer |
| :--- | :--- | :--- |
| `context_size` | 4 | 8 |
| `num_stories` | 200 | 1000 |

Everything else that's new — the 2-layer encoder (4 heads, `ffn_dim=1024`), `torch.compile`, float16 autocast + `GradScaler`, fused AdamW, `zero_grad(set_to_none=True)`, cosine LR schedule (`eta_min=1e-4`), gradient clipping (`1.0`), and inference temperature (`0.7`) — was adopted directly from [Keller Jordan's modded-nanogpt speedrun](https://github.com/KellerJordan/modded-nanogpt).

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

> 🔬 **Inspired ≠ Assumed:** A technique that wins at GPT-2 scale isn't guaranteed to help a 2M-parameter model trained in two minutes on a T4. Every Keller-lineage feature above was re-tested here via a dedicated ablation. That local testing is what most of this document is about.

---

## 🔧 The Default Stack: SimpleTransformer → TinyTransformer

`TinyTransformer.py`'s canonical config adds five code-level optimizations absent from `SimpleTransformer.py`. Each costs 1–2 lines:

| Component | `SimpleTransformer.py` | `TinyTransformer.py` | Accuracy Impact | Speed Impact | Proven By |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`torch.compile`** | ✅ Present | ✅ Same | Neutral | ~1.2× faster (after ~32s one-time compile tax) | Phase 8: Cold vs Warm ablation |
| **float16 autocast** | ❌ float32 | ✅ `torch.autocast` on forward + eval | Neutral | Major — halves memory bandwidth; enables batch=1536 + ctx=32 in <2 min | bfloat16 ablation: 4.2× slower for +0.2% |
| **`CosineAnnealingLR`** | ❌ Flat LR | ✅ `CosineAnnealingLR(T_max=n_steps, eta_min=1e-4)` | Smooths final convergence | Negligible | Phase 5: warmup on top gained only +0.6% |
| **AdamW** | ❌ `Adam(params, lr)` | ✅ `AdamW(..., betas=(0.9, 0.95), weight_decay=0.01, fused=True)` | Neutral; `weight_decay` stops repetitive output | `fused=True` speeds up GPU optimizer kernel | Experiment #9: `weight_decay` acts as grammar regularizer |
| **Fixed `eval_rng`** | ❌ Full dataset eval | ✅ Dedicated `eval_rng`, 4096-sample subset | Eliminates accuracy wobble | Faster per-eval | Scientific Method section |
| **Inference temperature** | 0.7 (hardcoded) | 0.5 (parameterized) | N/A | N/A | Eliminates invented words ("throbe" → "robe") |

> 💡 **The takeaway:** `torch.compile` + `float16` are the **speed engine**. `CosineAnnealingLR` + `AdamW`/`weight_decay` are the **quality polish**. `eval_rng` is the **scientific control**. All five originate from [Keller Jordan's modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt), but each earned its place only after local ablations confirmed it at this tiny scale.

---

## 🔧 SimpleTransformer Optimisation (Character-Level Baseline)

*Goal: Squeeze maximum accuracy from `SimpleTransformer.py` within a 2-minute Colab budget, changing only hyperparameters (zero new lines of code).*

*Baseline: Original config (2L, ctx=8, 200 stories, batch=1024, lr=1e-3) → 67.2% at step 2000, 36.6s*

| Change | Acc | Time | Verdict |
| :--- | ---: | ---: | :--- |
| Baseline (original) | 67.2% | 36.6s | Control |
| 3L + batch=1536 + lr=2e-3 + 1000 stories | 67.3% | 36.6s | ✅ Matches baseline faster |
| + ctx=32 + 5000 stories + eval subsample | 66.6% | 151.1s | ✅ Much cleaner text, no memorisation |
| + weight_decay=0.01 | 46.0% | 200.8s | ❌ Crushes 128d model — too small for regularisation |
| + embed_dim=256, ffn=512 (1.6M params) | ~63% | >240s | ❌ Over budget — compile tax ~28s alone |

**New canonical `SimpleTransformer.py` config:** `3L, ctx=32, 5000 stories, batch=1536, lr=2e-3, n_steps=1801, temp=0.5` → **~66.6% in ~151s**

> 💡 **`weight_decay` doesn't transfer.** The benchmark's `weight_decay=0.01` result was measured on `embed_dim=256`. At `embed_dim=128` with only 420K params, the same value collapses accuracy to ~46% — the regularisation overwhelms a model this small.

> 💡 **Eval OOM fix.** With `ctx=32` and 5000 stories, embedding the full dataset at eval time tries to allocate ~62GB. Fix: subsample 4096 rows with a fixed generator seed (`manual_seed(0)`) — eliminates OOM and stabilises the accuracy curve by always measuring the same rows.

> 💡 **Capacity ceiling confirmed at 420K params.** Accuracy plateaued at ~66-67% from step 1600 onward regardless of further steps. Breaking it requires graduating to `TinyTransformer.py`'s optimizer stack (AdamW + cosine LR + float16).

---

## 🔧 SimpleBPE.py — Minimal BPE Baseline

*Goal: Prove the tokenizer swap alone (character → BPE) improves text quality, using `SimpleTransformer.py`'s minimal code style with zero architectural changes.*

*Config: 3L, ctx=32 BPE tokens, vocab=4000, batch=1536, lr=2e-3, n_steps=1801, Adam flat LR. `params: 1,429,536`*

| Step | Loss | Acc | Time |
| ---: | ---: | ---: | ---: |
| 0 | 8.5443 | 2.1% | 0.2s |
| 200 | 3.6637 | 30.7% | 18.5s |
| 400 | 3.3484 | 34.4% | 36.5s |
| 600 | 3.1490 | 36.5% | 53.7s |
| 800 | 2.9328 | 38.1% | 70.7s |
| 1000 | 2.8415 | 38.9% | 88.1s |
| 1200 | 2.6802 | 39.6% | 105.6s |
| 1400 | 2.6610 | 40.9% | 123.3s |
| 1600 | 2.7252 | 41.7% | 140.7s |
| 1800 | 2.5975 | **42.2%** ⭐ | 157.9s |

**Training time: 157.9s**

**Generated sample:**
> ` fun. It was a good friend, a little girl named Lily. Lily loved to play with her ball and share it with her friends. One day,`

### Optimisation attempt: batch=2048 + cosine LR, n_steps=1601

| Step | Loss | Acc | Time |
| ---: | ---: | ---: | ---: |
| 0 | 8.5487 | 2.2% | 0.2s |
| 200 | 3.5808 | 31.0% | 24.5s |
| 400 | 3.2577 | 34.9% | 48.3s |
| 600 | 3.0478 | 37.3% | 71.1s |
| 800 | 2.8120 | 39.3% | 94.2s |
| 1000 | 2.7187 | 41.4% | 117.9s |
| 1200 | 2.6165 | 42.4% | 141.2s |
| 1400 | 2.5863 | 42.9% | 164.3s |
| 1600 | 2.6645 | **42.9%** ⭐ | 187.6s |

**Training time: 187.6s** *(over budget)*

> 💡 **batch=2048 is over budget on SimpleBPE.** Unlike `TinyBPE.py` which uses float16 to absorb the larger batch, `SimpleBPE.py` runs float32 — so batch=2048 adds ~50% wall-clock time per step for only +0.7% accuracy gain. Not worth it without float16.

> 💡 **Cosine LR smoothed the tail.** Loss stopped bouncing in the final 400 steps (compare step 1600 loss: 2.66 flat LR vs 2.58 cosine), but the gain was absorbed by the budget overrun. Valid improvement, marginal at this scale without more steps.

> 💡 **The tokenizer swap alone is the story.** `SimpleBPE.py` vs `SimpleTransformer.py` — identical architecture, identical training loop, one import changed — yet the generated text jumps from broken clauses to coherent multi-sentence paragraphs. The ~42% vs ~67% raw accuracy gap is meaningless; BPE predicts 1 of 4,000 tokens vs 1 of 65 characters.

> 💡 **`SimpleTransformer.py` → `SimpleBPE.py` is the clearest demonstration in the repo** that the ~67% character-level ceiling is an information bottleneck, not a model capacity problem. No new code required.

---

## 📊 The Leaderboard: Model Comparison

*Best configuration for each architecture we tested.*

| Model | Best Accuracy | Steps Taken | Relative Speed (vs 2L Baseline) |
| :--- | ---: | ---: | ---: |
| NameSLP.py | 39.6% | 2000 | 1.8× |
| TinyMLP.py | 59.4% | 2000 | 0.2× |
| TorchMLP.py | 62.4% | 2000 | 0.2× |
| SimpleTransformer.py | 67.2% | 2000 | 1.8× |
| **TinyTransformer.py (2 layers)** 🥇 | **68.4%** | **2000** | **1.0× (Control)** |
| TinyTransformer.py (context=64) | 68.5% | 1800 | 10.0× |
| TinyTransformer.py (Narrow-Deep 4L, 810K params) | 68.9% | 2400 | 3.5× |
| TinyTransformer.py (Efficient-Deep 4L, ffn=512) | 70.8% | 2000 | 2.3× |
| TinyTransformer.py (Balanced Narrow-Deep 4L, 192d) | 70.8% | 2400 | 2.9× |
| TinyTransformer.py (3 layers, Wider FFN=2048) | 71.8% | 2200 | 3.0× |
| TinyTransformer.py (3 layers, batch=1024, lr=2e-3) | 72.4% | 2200 | 2.5× |
| TinyTransformer.py (4 layers) | 73.1% | 3400 | 4.0× |
| TinyTransformer.py (3 layers) | 73.5% | 2200 | 1.5× |
| TinyTransformer.py (3 layers, batch=1536) ✨ | 73.0%* | 2200 | 2.7× |
| **TinyTransformer.py (3 layers, batch=2048)** | **76.1%** | **2200** | **~3.5×** |
| **TinyTransformer.py (3L, ctx=16, 5000 stories)** 🧠 | **71.7%** | **2200** | **~2.5×** |
| **TinyTransformer.py (3L, ctx=32, 5000 stories, 1536 batch)** 👑 | **70.0%** | **1600** | **~3.2×** |
| TinyTransformer.py (3L, ctx=32, 5000 stories, warmup+clip) | 70.7% | 1600 | ~3.3× |
| TinyTransformer.py (3L, ctx=32, 5000 stories, 2048 batch) | 70.5% | 1600 | ~4.8× |
| TinyTransformer.py (3L, ctx=32, 5000 stories, 8 heads) | 70.5% | 1800 | ~4.6× |
| TinyTransformer.py (3L, ctx=32, 5000 stories, embed=320) | 70.5% | 1600 | ~5.5× |
| **TinyTransformer.py (3L, ctx=32 BPE tokens, 5000 stories)** 🚀 | **50.9%†** | **1800** | **~5.3×** |
| **TinyTransformer.py (3L, ctx=32 BPE, 5000 stories, batch=2048)** 🏆 | **50.0%†** | **1200** | **~5.0×** |
| **TinyTransformer.py (3L, custom BPE vocab=4000, 5000 stories, batch=2048)** ⚡ | **46.2%†** | **900** | **~2.6×** |
| **TinyBPE.py (3L, custom BPE vocab=4000, n_steps=1001)** 🏆 | **~47%†** | **1001** | **~2.7×** |
| **TinyBPE.py (3L, custom BPE vocab=4000, 10k stories, n_steps=1201)** 🏆 | **~45.9%†** | **1201** | **~3.2×** |

*† Accuracy not comparable to character-level rows — BPE predicts 1 of 50,257 tokens vs 1 of 65 characters. See generated sample for true quality assessment.*

---

## ⚠️ The Memorization Trap

Look at the bottom rows of the leaderboard. Why did accuracy drop to ~70% after hitting 76.1%? Because we expanded the dataset from 1,000 to 5,000 stories. The 76.1% model was **cheating** — it memorized the test set. The ~70% models stopped memorizing and actually learned English.

> **Lower accuracy score = higher real-world intelligence.**

This is the single most important insight in this document. On small datasets, high accuracy is an illusion. Always check your generated samples, not just the numbers.

---

## 🔬 Ablation & Experiment Summary

All tests below are single changes made to our baseline 2-layer TinyTransformer (~68% accuracy, 1.0× speed).

### 🏗️ Architecture (Shape & Size)

| Type | Change Tested | Accuracy Δ | Speed Δ | The Verdict |
| :--- | :--- | ---: | ---: | :--- |
| **Exp** | **Depth:** 2 → 3 layers | +5.1% | 1.5× slower | ✅ Best speed/accuracy tradeoff. |
| **Exp** | **Depth:** 2 → 4 layers | +1.2% | 2.2× slower | ✅ Worth it if you have time (73.1% at 3400 steps). |
| **Exp** | **Shape:** Wide/Short → Narrow/Deep | +1.0% | 20% slower | ✅ Depth beats width, even with half the parameters! |
| **Exp** | **Narrow-Deep Alt. HPs** (128d, 4L) | +0.5% | 3.4× slower | ⚠️ Half the params, competitive accuracy. |
| **Exp** | **Efficient-Deep** (256d, ffn=512, 4L) | +2.4% | 2.3× slower | ⚠️ Strong mid-training but peaks early. |
| **Exp** | **Balanced Narrow-Deep** (192d, 4L) | +2.4% | 2.8× slower | ⚠️ Ties Efficient-Deep but takes longer. |
| **Exp** | **Wider FFN** (3L, ffn=2048) | +3.4% | 3.0× slower | ⚠️ Bigger MLP helps, but not enough to beat standard 3L. |
| **Exp** | **Heads:** 4 → 8 | +0.5% | ~1.4× slower | ❌ Same ceiling, more overhead. |
| **Exp** | **Width:** embed_dim 256 → 320 | +0.5% | 1.7× slower | ❌ 35% more params, zero gain. Capacity isn't the bottleneck. |
| **Abl** | **Remove Positional Embeddings** | −7.7% | Negligible | ❌ Without this, the AI reads sentences as "word soup." |

### ⚡ Training & Speed Hacks

| Type | Change Tested | Accuracy Δ | Speed Δ | The Verdict |
| :--- | :--- | ---: | ---: | :--- |
| **Exp** | **`torch.compile`** (Cold vs Warm) | Neutral | ~1.2× faster overall (one-time ~32s compile tax) | ✅ Always "warm up" your model before timing it! |
| **Exp** | **Precision:** float16 → bfloat16 | +0.2% | 4.2× slower | ❌ The T4 GPU doesn't have native bfloat16 hardware. |
| **Exp** | **Weight Tying** (sharing layers) | −3.0% | Neutral | ❌ Confused the model and hurt accuracy. |
| **Exp** | **Activation:** ReLU → GELU | Neutral | 14% slower | ❌ GELU is too math-heavy for this small model. |
| **Exp** | **Loss:** Last-word vs Full-sequence | Neutral | 1.47× slower | ⚠️ Learns faster early on, but hits the same ceiling. |
| **Exp** | **Flash Attention** + Context 32 | +0.2% | 3.2× slower | ⚠️ Memory-efficient math works, but model is too small to benefit. |
| **Exp** | **LR Warmup** (50 steps) + **Grad Clipping** (1.0) | +0.6% peak | Slightly slower | ⚠️ Smoother curve, but not worth the extra code at this scale. |

### 🧠 The "Real Intelligence" Push (Batch, Context & Data)

| Type | Change Tested | Accuracy Δ | Speed Δ | The Verdict |
| :--- | :--- | ---: | ---: | :--- |
| **Exp** | **High LR Fast Convergence** (batch=1024) | +4.0% | 2.5× slower | ⚠️ Faster, but high LR makes training unstable. |
| **Exp** | **Middle Ground** (batch=1536) | +6.8% | 2.7× slower | ✅ Excellent compromise. ~1 min runtime. |
| **Exp** | **Large Batch + High LR** (batch=2048) | +7.7% | ~3.5× slower | ✅ Huge accuracy win — but memorizes (see Memorization Trap). |
| **Exp** | **Dataset Size:** 1k → 3k/5k stories | −4.7% | Negligible | ✅ Drops raw acc, but drastically improves grammar. |
| **Exp** | **Context Size:** 8 → 16 (on large dataset) | −1.5% | ~1.5× slower | ✅ Fixes pronoun/gender swapping. Model can track subjects! |
| **Exp** | **Weight Decay:** 0 → 0.01 | Neutral | Negligible | ✅ Acts as a "grammar regularizer." Stops lazy repetition. |
| **Exp** | **Context Size:** 16 → 32 (on large dataset) | −1.6% | ~1.3× slower | ✅ Fixes 90% of pronoun swaps. The ultimate 2-min tradeoff. |
| **Exp** | **batch=2048 on 5k stories** (ctx=32) | +0.5% vs 1536 | 1.5× slower | ❌ Same ~70% ceiling. Batch size stops helping when genuinely learning. |
| **Exp** | **Inference Temp:** 0.7 → 0.5 | N/A (Inference) | N/A | ✅ Eliminates fake words (e.g., "throbe" → "robe"). |
| **Exp** | **BPE Tokenization** (tiktoken gpt2, vocab=50257) | See Phase 9 | ~5.3× slower | ✅ **Breaks the 70% character ceiling** — full paragraphs & dialogue. |
| **Exp** | **BPE + Larger Batch** (batch=2048, 1401 steps) | −0.9% vs P9 | ~1.05× faster/step | ✅ Reaches ~50% in fewer steps. Best short-budget BPE variant. |
| **Exp** | **BPE + Weight Tying** (`linear.weight = tok_embed.weight`) | N/A | Neutral | ❌ Loss=254 at step 0 across all attempts. Init instability at vocab=50k. Reverted. |
| **Exp** | **BPE + LR Warmup** (100-step linear + cosine) | N/A | Negligible | ❌ Did not fix tied-weight instability; unnecessary without tying. |
| **Exp** | **Custom BPE** (vocab=4000, trained on TinyStories) | −3.8% vs P9/P10 BPE | **~2× faster** | ✅ 4.43M params (vs 28M), fits in 103.6s. Best size/speed tradeoff. |
| **Exp** | **Logit Softcapping** (±15, Gemma 2 style) | Neutral | Negligible | ✅ Stable training, no NaN issues at custom vocab scale. |
| **Exp** | **Dataset Size:** 5k → 10k stories (BPE) | −0.9% raw | Negligible | ✅ Same memorisation-trap pattern as char-level. Better quality, lower score. |

---

## 📈 Step-by-Step Accuracy Data

*We split data into Phases to tell the story of our experiments. ⭐ marks peak accuracy. 📉 shows overfitting.*

**Legend:**
- **2L/3L/4L** = TinyTransformer with 2, 3, or 4 layers
- **ND** = Narrow-Deep (skinnier model, more layers)
- **FFN** = Feed-Forward Network width

### Phase 1: The Baselines
*Goal: Does a basic Transformer beat the older, simpler models?*

| Step | NameSLP | TinyMLP | SimpleTrans | **2L (Baseline)** |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 3.5% | 4.7% | 4.0% | 19.3% |
| 200 | 37.1% | 44.8% | 53.5% | 54.8% |
| 800 | 38.9% | 55.0% | 62.4% | 63.2% |
| 1600 | 39.5% | 58.3% | 66.2% | 67.0% |
| 2000 | **39.6%** ⭐ | **59.4%** ⭐ | **67.2%** ⭐ | 67.4% |

### Phase 2: Shape & Size Tests
*Goal: Does adding layers, widening the model, or changing its shape beat the 2L Baseline?*

| Step | **3L** (Run 1) | **4L** | ND 4L (128d) | Eff. Deep 4L | Bal. ND 4L (192d) | Wider FFN 3L |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 19.3% | 19.3% | 10.6% | 5.2% | 19.3% | 20.2% |
| 800 | 64.8% | 64.6% | 63.0% | 63.9% | 65.6% | 64.7% |
| 1600 | 67.6% | 68.0% | 67.8% | 68.4% | 70.0% | 70.2% |
| 2000 | 70.2% | 68.9% | 69.4% | **70.8%** ⭐ | 70.5% | 71.1% |
| 2200 | **73.5%** ⭐ | - | 68.1% 📉 | 69.7% 📉 | 70.4% | **71.8%** ⭐ |
| 2400 | 71.7% 📉 | - | 68.9% | - | **70.8%** ⭐ | - |
| 3400 | - | **73.1%** ⭐ | - | - | - | - |

> 💡 **Overfitting in action:** The 3-layer model hits 73.5% at step 2200, then drops to 71.7% at step 2400. The model memorized the training data so hard it got *worse* at writing new stories. Always stop at ⭐!

### Phase 3: The "Raw Score" Champions
*Goal: Instead of changing the model's shape, what if we change HOW it learns? (Using the 3-layer model)*

| Step | High LR (batch=1024) | Mid Ground (batch=1536) | **Large Batch+LR** (batch=2048) |
| ---: | ---: | ---: | ---: |
| 0 | 19.3% | 19.3% | 19.3% |
| 800 | 62.9% | 65.5% | 66.9% |
| 1600 | 66.2% | 69.9% | 71.0% |
| 2000 | 68.8% | 70.8% | 72.3% |
| 2200 | **72.4%** ⭐ | **75.2%** ⭐ | **76.1%** ⭐ |
| 2400 | 71.1% 📉 | 73.0% 📉 | - |

### Phase 4: The Real Intelligence Push
*Goal: Stop chasing raw accuracy. Expand the dataset and context window to force the model to learn English rather than memorize 1,000 stories.*

| Step | **3L, 2048 batch, 3k stories** (ctx=8) | **3L, 1536 batch, 5k stories** (ctx=32, wd=0.01) |
| ---: | ---: | ---: |
| 0 | 18.5% | 19.2% |
| 400 | 62.1% | 61.4% |
| 800 | 66.0% | 64.4% |
| 1200 | 67.8% | 67.8% |
| 1600 | 69.3% | **70.0%** ⭐ |
| 2000 | 71.4% | - |

> 💡 Scores here are *lower* than Phase 3's 76.1%, but the generated text is dramatically better. Lower eval score, higher real-world intelligence.

### Phase 5: Optimizer Stability (Warmup & Gradient Clipping)
*Goal: Do standard optimizer-stability tricks help enough at this scale to justify the extra code?*

*Changes added to the Phase 4 canonical config (3L, ctx=32, 5k stories, batch=1536):*
- LR Warmup: 50-step linear warmup before cosine decay
- Gradient Clipping: `clip_grad_norm_(params, 1.0)` after every backward pass

| Step | Loss | Acc | LR |
| ---: | ---: | ---: | ---: |
| 0 | 4.5791 | 5.6% | 5.96e-05 |
| 200 | 1.4316 | 55.9% | 1.97e-03 |
| 400 | 1.3176 | 60.5% | 1.82e-03 |
| 600 | 1.1867 | 64.2% | 1.57e-03 |
| 800 | 1.1465 | 64.8% | 1.26e-03 |
| 1000 | 1.1120 | 65.0% | 9.21e-04 |
| 1200 | 1.0363 | 68.0% | 5.98e-04 |
| 1400 | 0.9979 | 68.6% | 3.33e-04 |
| 1600 | 0.9784 | **70.7%** ⭐ | 1.60e-04 |
| 1800 | 0.9542 | 70.5% 📉 | 1.00e-04 |

**Training time: 133.6s**

> 💡 Warmup + clipping gave a smoother curve and slightly higher peak (**70.7%**), but the simpler no-warmup baseline reached **70.0%** in less time (**127.7s**) with less code. Valid technique — marginal benefit at this scale.

### Phase 6: Bigger Batch on Large Dataset
*Goal: Phase 3 showed batch=2048 was a huge win on 1k stories. Does the same trick work on 5k stories and ctx=32?*

*Single change from Phase 4 canonical config:* `batch_size 1536 → 2048`

| Step | Loss | Acc | Time |
| ---: | ---: | ---: | ---: |
| 0 | 4.5770 | 19.2% | 0.2s |
| 200 | 1.4332 | 56.6% | 18.7s |
| 400 | 1.3018 | 62.2% | 38.8s |
| 600 | 1.1745 | 64.9% | 58.6s |
| 800 | 1.1103 | 65.7% | 77.4s |
| 1000 | 1.0814 | 65.6% | 96.0s |
| 1200 | 1.0087 | 68.0% | 115.0s |
| 1400 | 0.9984 | 69.2% | 134.3s |
| 1600 | 0.9760 | **70.5%** ⭐ | 153.3s |
| 1800 | 0.8941 | 70.3% | 172.3s |
| 2000 | 0.9301 | 70.4% | 191.2s |

**Training time: 191.2s**

> 💡 Peak accuracy **70.5%** — the same ~70% ceiling, but taking **191s vs 128s** (50% more time). The batch size increase bought nothing.
>
> **Why batch size stops working:** On 1k stories, a bigger batch helped the model memorize faster. On 5k stories, the model is genuinely *learning*, so the bottleneck has shifted. The ceiling is a **capacity ceiling**, not an optimisation ceiling. Breaking it requires more parameters or architectural changes — not a bigger batch.

### Phase 7: Breaking the Ceiling — More Heads, More Params (Both Fail)
*Goal: Three experiments tried to break the ~70% wall: bigger batch, more heads, wider model. All three landed on exactly 70.5%.*

#### 7a. More Attention Heads (4 → 8)
*Single change from canonical:* `n_heads 4 → 8`

| Step | Loss | Acc | Time |
| ---: | ---: | ---: | ---: |
| 0 | 4.5728 | 19.2% | 29.1s |
| 200 | 1.4091 | 57.3% | 45.8s |
| 400 | 1.2916 | 61.7% | 63.8s |
| 600 | 1.1752 | 64.8% | 81.4s |
| 800 | 1.1245 | 65.4% | 98.2s |
| 1000 | 1.0994 | 65.6% | 114.8s |
| 1200 | 1.0204 | 67.8% | 131.7s |
| 1400 | 0.9919 | 68.6% | 148.8s |
| 1600 | 0.9828 | 70.4% | 165.8s |
| 1800 | 0.9118 | **70.5%** ⭐ | 182.8s |

**Training time: 182.8s**

#### 7b. Wider Model (embed_dim 256 → 320)
*Single change from canonical:* `embed_dim 256 → 320` (3.26M params, +35%). Required `lr=1e-3` + grad clipping to prevent NaN divergence.

| Step | Loss | Acc | Time |
| ---: | ---: | ---: | ---: |
| 0 | 4.5175 | 19.2% | 0.2s |
| 200 | 1.4214 | 56.8% | 25.9s |
| 400 | 1.3129 | 62.1% | 50.6s |
| 600 | 1.1738 | 65.0% | 74.0s |
| 800 | 1.1253 | 65.2% | 97.8s |
| 1000 | 1.1088 | 65.5% | 122.2s |
| 1200 | 1.0484 | 67.7% | 146.3s |
| 1400 | 1.0073 | 69.0% | 170.1s |
| 1600 | 0.9944 | **70.5%** ⭐ | 194.0s |
| 1800 | 0.9337 | 70.0% | 218.0s |

**Training time: 218.0s**

> 💡 **The Definitive Result:** Three completely different approaches — bigger batch (+33% data/step), more heads (+100% attention patterns), wider model (+35% parameters) — all converged on **exactly 70.5%**. This is not coincidence; it is the **information ceiling** of character-level tokenization at ctx=32 (~5–6 words). No amount of model capacity can extract more signal than exists in a 5-word window.
>
> 💡 **The NaN Lesson:** The wider model (embed=320) immediately diverged to NaN at `lr=2e-3` with no clipping. This is the first experiment where gradient clipping became *necessary*, not optional — larger embeddings produce larger gradients that destabilize the optimizer before it can warm up.
>
> 💡 **The torch.compile Warning:** The embed=320 run triggered `torch._dynamo` recompilation warnings (hit config limit of 8). This is caused by the eval loop toggling `autocast` on/off, forcing graph recompilation.

### Phase 8: `torch.compile` — Cold vs Warm Start
*Goal: Measure exactly how much `torch.compile` graph compilation costs, using the Phase 4 canonical config (3L, ctx=32, 5k stories, batch=1536). `params: 2,414,408`.*

**Cold Start** (compilation happens inside the timed run):

| Step | Loss | Acc | Time |
| ---: | ---: | ---: | ---: |
| 0 | 4.5738 | 19.2% | 32.5s |
| 200 | 1.4513 | 55.4% | 46.0s |
| 400 | 1.3140 | 61.3% | 59.8s |
| 600 | 1.2122 | 63.4% | 74.2s |
| 800 | 1.1545 | 64.6% | 89.7s |
| 1000 | 1.0998 | 64.6% | 105.3s |
| 1200 | 1.0466 | 67.7% | 119.8s |
| 1400 | 0.9982 | 68.7% | 134.1s |
| 1600 | 0.9977 | **70.8%** ⭐ | 148.5s |
| 1800 | 0.9138 | 69.8% 📉 | 163.2s |

**Training time: 163.2s**

**Warm Start** (model already compiled before the timer starts):

| Step | Loss | Acc | Time |
| ---: | ---: | ---: | ---: |
| 0 | 4.5772 | 19.2% | 0.2s |
| 200 | 1.4477 | 55.7% | 15.3s |
| 400 | 1.3053 | 62.0% | 31.9s |
| 600 | 1.2024 | 63.1% | 47.0s |
| 800 | 1.1552 | 64.2% | 61.3s |
| 1000 | 1.1250 | 64.9% | 75.5s |
| 1200 | 1.0432 | 67.5% | 90.0s |
| 1400 | 1.0152 | 68.8% | 105.1s |
| 1600 | 0.9828 | 70.4% | 120.1s |
| 1800 | 0.9244 | **70.5%** ⭐ | 134.9s |

**Training time: 134.9s**

> 💡 Step 0 tells the whole story — **32.5s (cold) vs 0.2s (warm)** — a one-time graph-compilation tax of ~32 seconds, paid only on the very first call. After that, both runs advance at nearly identical per-step speed. Total time: **163.2s (cold) vs 134.9s (warm)** — a **~1.2× (17%) difference**.
>
> **The takeaway:** `torch.compile`'s cost is almost entirely a **fixed upfront tax**, not a per-step penalty. The longer you train, the smaller that tax looks as a percentage of total time. Always benchmark *after* warmup.

**Generated samples:**
- **Cold start (69.8%):** `Once there was a little boy named Tim. He was scared and said, "Thank you, Mom. I want to find they inside. They did not have the park with the temple home. They were happy to have a new friend and said, "I will help you find a big`
- **Warm start (70.5%):** `Once there was a little boy named Tim. He was so happy. The dog was scared and said, "I will give you so much fun. It was a sorry, but I will said, "You should not stopped and said, "I like that you do the kitchen. It was a dark and`

### Phase 9: Breaking the Ceiling — BPE Tokenization (tiktoken)
*Goal: All Phase 7 experiments proved the ~70% ceiling is an **information bottleneck** from character-level tokenization, not a capacity problem. The fix: swap to subword BPE tokens so ctx=32 covers ~20–25 words instead of ~5–6.*

*Single change from Phase 4 canonical config:* Replace character tokenizer with `tiktoken` GPT-2 BPE (`vocab_size=50,257`). Architecture and all hyperparameters unchanged. `params: 28,159,313` (dominated by the larger embedding table: 50257×256 vs 65×256).

> ⚠️ **Accuracy is not comparable to earlier phases.** The model now predicts 1 of 50,257 tokens instead of 1 of 65 characters. A "50.9%" here is a much harder task than "70.0%" in Phase 4. Judge quality by the generated sample.

| Step | Loss | Acc | Time |
| ---: | ---: | ---: | ---: |
| 0 | 11.2628 | 7.6% | 0.2s |
| 200 | 3.5206 | 35.6% | 23.3s |
| 400 | 3.0509 | 39.8% | 48.2s |
| 600 | 2.7770 | 41.4% | 71.5s |
| 800 | 2.7235 | 42.7% | 94.3s |
| 1000 | 2.5899 | 45.9% | 118.0s |
| 1200 | 2.3443 | 47.5% | 141.5s |
| 1400 | 2.2842 | 49.0% | 164.9s |
| 1600 | 2.2427 | 49.2% | 188.1s |
| 1800 | 2.2064 | **50.9%** ⭐ | 211.3s |

**Training time: 211.3s** *(still climbing at step 1800 — model has not peaked)*

> 💡 **The ceiling is broken.** The loss curve shows no plateau at step 1800, unlike every character-level run which flattened by step 1600. The model is still learning — running to `n_steps=3601` would likely push accuracy higher.
>
> 💡 **Why 28M params?** The transformer itself is the same ~2.4M as Phase 4. The extra ~26M comes entirely from the embedding table (`tok_embed`: 50257×256 = 12.9M) and output head (`linear`: 50257×256 = 12.9M). Weight tying (`linear.weight = tok_embed.weight`) would halve this to ~15M with no accuracy cost — a natural next experiment.
>
> 💡 **ctx=32 BPE tokens ≈ 100+ characters** — roughly 20–25 words. The model now has enough context to track subject–verb agreement, character names, and dialogue turns across a full sentence, which is reflected directly in the generated sample quality.

### Phase 10: BPE Short-Budget Run & Weight-Tying Ablation
*Goal: Fit the BPE model closer to a 2-minute Colab budget. Single change from Phase 9: `batch_size` 1536 → 2048, `n_steps` 1801 → 1401.*

*Config:* `3L, ctx=32 BPE tokens, 5k stories, batch=2048, n_steps=1401, lr=2e-3, cosine LR, no weight tying`. `params: 28,159,313`.

| Step | Loss | Acc | Time |
| ---: | ---: | ---: | ---: |
| 0 | 11.2768 | 7.6% | 33.4s |
| 200 | 3.3718 | 36.2% | 56.2s |
| 400 | 2.9462 | 40.1% | 79.5s |
| 600 | 2.6206 | 43.5% | 103.3s |
| 800 | 2.5360 | 44.5% | 127.6s |
| 1000 | 2.4468 | 48.0% | 151.5s |
| 1200 | 2.2626 | **50.0%** ⭐ | 175.4s |
| 1400 | 2.2288 | 49.9% 📉 | 199.4s |

**Training time: 199.4s** *(compile tax alone is 33.4s — a warm start would land near 166s)*

> 💡 **Approaching the ceiling.** Loss drops only 0.034 from step 1200→1400, versus 0.18 from 1000→1200 — the batch=2048 BPE model is starting to plateau near 50%, much sooner than Phase 9's still-climbing curve at step 1800.
>
> 💡 **Weight tying fails at BPE scale.** Three separate attempts at `linear.weight = tok_embed.weight` — with and without a 100-step `SequentialLR` warmup — all produced **Loss ≈ 254 at step 0** and never recovered. At vocab=65 (Theme 2, Item 1) tying hurt accuracy by 3.0% due to mismatched init scale; at vocab=50,257 the failure mode is worse — the shared matrix receives contradictory gradients from the embedding lookup and output projection simultaneously, destabilizing initialization before training can even start.
>
> 💡 **LR warmup doesn't help.** Testing `LinearLR(start_factor=0.01, total_iters=100)` + `CosineAnnealingLR` on top of the broken tied-weight run showed no recovery. Removing tying (not adding warmup) was the actual fix.

### Phase 11: Custom Small-Vocab BPE (2-Minute Champion)
*Goal: GPT-2's 50,257-token vocab is oversized for TinyStories. Train a custom BPE tokenizer directly on the TinyStories corpus with a much smaller vocab, cutting embedding-table size dramatically.*

*Config:* `3L, ctx=32 custom-BPE tokens, 5k stories, batch=2048, n_steps=901, lr=2e-3, cosine LR, logit softcapping (±15), warm-compiled`. Tokenizer: HuggingFace `tokenizers` BPE trained from scratch on the TinyStories text, `vocab_size=4000`. `params: 4,429,472` (vs 28,159,313 for GPT-2 BPE — an 84% reduction).

| Step | Loss | Acc | Time |
| ---: | ---: | ---: | ---: |
| 0 | 8.6952 | 7.4% | 0.2s |
| 100 | 3.7445 | 31.8% | 11.1s |
| 200 | 3.2566 | 35.4% | 22.5s |
| 300 | 3.0530 | 38.2% | 34.8s |
| 400 | 2.9823 | 40.2% | 47.3s |
| 500 | 2.7751 | 41.7% | 58.9s |
| 600 | 2.5971 | 43.7% | 70.2s |
| 700 | 2.6493 | 43.7% | 81.2s |
| 800 | 2.5199 | 43.8% | 92.2s |
| 900 | 2.4410 | **46.2%** ⭐ | 103.6s |

**Training time: 103.6s** *(well under the 2-minute budget, including warm compile)*

> 💡 **6.4× smaller model, comparable learning curve.** At 4.43M params (vs 28M for GPT-2 BPE), the custom-vocab model reaches 46.2% in 900 steps — lower than Phase 9/10's ~50%, but in roughly half the wall-clock time, with an embedding table 92% smaller.
>
> 💡 **Custom vocab trains fast.** Training the 4000-token BPE tokenizer directly on the 5k-story corpus took only a few seconds and required no external dependency beyond HuggingFace `tokenizers`.
>
> 💡 **Logit softcapping (Gemma 2 trick) added for free.** `logits = 15.0 * torch.tanh(logits / 15.0)` before the loss bounds extreme logit values — no accuracy cost, and it's a safety net against divergence at any vocab size.
>
> 💡 **Trade-off is explicit:** smaller vocab (4000 vs 50257) means each token covers less linguistic ground, so raw accuracy is lower than GPT-2 BPE runs. But the model is 6.4× smaller and trains 2× faster, generating comparably fluent text — a strong choice when compute is the binding constraint.

## Phase 12: TinyBPE Optimisation (Steps, LR Tail, Vocab Size)
*Goal: Three single-variable experiments on the Phase 11 canonical config (3L, custom BPE vocab=4000, batch=2048, 901 steps) to find the optimal 2-minute TinyBPE config.*

*Baseline (Phase 11):* `46.2% at step 900, 103.6s`

### 12a. More Steps (901 → 1201)
*Single change:* `n_steps 901 → 1201`

| Step | Loss | Acc | Time |
| ---: | ---: | ---: | ---: |
| 0 | 8.6961 | 7.4% | 0.2s |
| 100 | 3.7353 | 31.6% | 11.5s |
| 200 | 3.2515 | 35.9% | 23.5s |
| 300 | 3.0439 | 38.3% | 36.1s |
| 400 | 3.0003 | 39.8% | 48.1s |
| 500 | 2.7944 | 41.8% | 59.5s |
| 600 | 2.6135 | 44.0% | 70.5s |
| 700 | 2.6437 | 43.7% | 81.6s |
| 800 | 2.4895 | 45.1% | 92.9s |
| 900 | 2.3761 | 46.4% | 104.4s |
| 1000 | 2.3650 | 47.4% | 116.1s |
| 1100 | 2.3311 | 47.4% | 127.7s |
| 1200 | 2.3813 | **48.1%** ⭐ | 139.3s |

**Training time: 139.3s** *(over 2-min budget — but proves the curve hasn't peaked)*

> 💡 **Still climbing at step 901.** Loss dropped from 2.45 → 2.38 between steps 900–1200, confirming the Phase 11 run was cut off early. Peak accuracy improved by **+2.1%** (46.2% → 48.1%). The step-1200 loss wobble (+0.05) mirrors the step-700 blip — stochastic noise, not a plateau.
>
> 💡 **Takeaway:** More steps is the only lever that meaningfully moves accuracy. Since 1201 steps overshoots the 2-minute wall, the optimal within-budget config is `n_steps=1001` (~116s).

### 12b. Slower LR Tail (eta_min 1e-4 → 3e-4)
*Single change:* `eta_min 1e-4 → 3e-4`

| Step | Loss | Acc | Time |
| ---: | ---: | ---: | ---: |
| 0 | 9.7056 | 7.4% | 0.2s |
| 300 | 3.0556 | 38.2% | 35.6s |
| 600 | 2.5919 | 43.3% | 70.5s |
| 900 | 2.4091 | **46.3%** ⭐ | 104.3s |

**Training time: 104.3s**

> 💡 **No effect.** +0.1% vs baseline — within noise. The cosine tail difference between `1e-4` and `3e-4` is only ~0.2e-3 LR at step 900, too small to influence learning at this budget. Would only matter at 2000+ steps where the tail dominates.

### 12c. Larger Vocab (4000 → 6000 tokens)
*Single change:* `vocab_size 4000 → 6000` → `params: 5,455,472`

| Step | Loss | Acc | Time |
| ---: | ---: | ---: | ---: |
| 0 | 9.0211 | 8.4% | 0.2s |
| 300 | 2.9156 | 39.6% | 36.8s |
| 600 | 2.6876 | 42.5% | 71.1s |
| 900 | 2.4887 | **45.6%** ⭐ | 105.8s |

**Training time: 105.8s**

> 💡 **Marginal loss (−0.6%).** The larger vocab adds 1M params to the embedding tables but the model doesn't have enough steps to learn the extra tokens at this budget. Generated text quality was subjectively better despite the lower score — richer tokenisation needs more steps to pay off.
>
> 💡 **Natural follow-up:** vocab=6000 + n_steps=1101 (~120s) — tests whether richer tokenisation beats vocab=4000 given equal wall-clock time.

### ✅ Phase 12 Verdict & New Canonical Config

| Experiment | Change | Acc | Time | Verdict |
| :--- | :--- | ---: | ---: | :--- |
| Phase 11 baseline | — | 46.2% | 103.6s | Control |
| **12a. More steps** | 901 → 1201 | **48.1%** | 139.3s | ⚠️ Over budget — sets `n_steps=1001` as new optimal |
| 12b. Slower LR tail | eta_min → 3e-4 | 46.3% | 104.3s | ❌ Noise |
| 12c. Larger vocab | 4000 → 6000 | 45.6% | 105.8s | ❌ Needs more steps |

**New canonical `TinyBPE.py` config:** `vocab=4000, n_steps=1001, eta_min=1e-4` — all other hyperparameters unchanged. Expected accuracy: **~47%** at **~116s**.

---

## Phase 13: TinyBPE Scale-Up (10k Stories, Depth, Context)

*Baseline: TinyBPE canonical config (3L, custom BPE vocab=4000, 5k stories, batch=2048, n_steps=1001, ~116s) → 46.8%†*

### 13a. More Data (5k → 10k stories, n_steps=1001)
*Single change:* `num_stories 5000 → 10000`

| Step | Loss | Acc | Time |
| ---: | ---: | ---: | ---: |
| 0 | 8.6104 | 7.9% | 0.2s |
| 500 | 2.8900 | 41.3% | 59.4s |
| 1000 | 2.6245 | **44.8%** ⭐ | 117.0s |

**Training time: 117.0s**

> 💡 **Raw accuracy dropped ~2%** (46.8% → 44.8%) — identical pattern to the 1k→5k char-level transition. The model is generalising rather than memorising. Generated text shows genuine improvement: multi-character interactions, subordinate clauses with motives ("Sue was sad *because* she wanted to take the car home"), and resolved story arcs. Lower score, higher real-world intelligence.
>
> 💡 **Loss oscillates in the tail.** The cosine LR decays too aggressively for the harder 10k distribution, causing ±0.1 loss bounce rather than clean convergence — more steps needed.

### 13b. More Steps on 10k (n_steps 1001 → 1201)
*Single change from 13a:* `n_steps 1001 → 1201`

| Step | Loss | Acc | Time |
| ---: | ---: | ---: | ---: |
| 1000 | 2.5719 | 45.3% | 116.9s |
| 1100 | 2.5445 | 45.6% | 128.4s |
| 1200 | 2.5743 | **45.9%** ⭐ | 139.8s |

**Training time: 139.8s** *(new best for 10k config)*

> 💡 **+1.1% over 1001-step run** (44.8% → 45.9%). Loss still oscillating at step 1200 — sampling noise from batch=2048 covering only ~20% of 10k stories per step. This is a capacity ceiling, not an LR problem.

### 13c. Slower LR Tail on 10k (eta_min 1e-4 → 3e-4)
*Single change from 13b:* `eta_min 1e-4 → 3e-4`

**Peak: 45.7% at step 1200, 139.6s — no effect (−0.2% vs 13b, within noise)**

> 💡 The tail oscillation is **batch sampling variance**, not LR decay. With batch=2048 from 10k stories, each batch covers ~20% of the dataset — late-training loss bounces are inherent stochasticity. `eta_min` tuning cannot fix this. ❌ Don't bother.

### 13d. Deeper Model (n_layers 3 → 4, n_steps=801, warm compile)
*Single change from 13b:* `n_layers 3 → 4, n_steps 1201 → 801`

| Step | Loss | Acc | Time |
| ---: | ---: | ---: | ---: |
| 0 | 8.6876 | 7.9% | 0.3s |
| 400 | 2.8991 | 40.3% | 63.0s |
| 800 | 2.6297 | **44.3%** ⭐ | 124.4s |

**Training time: 124.4s**

> 💡 **4L/801 steps (44.3%) does not beat 3L/1201 steps (45.9%) within the 2-minute budget.** 4L needs ~1100 steps to reach its peak but that costs 241s — confirmed by earlier 4L/1401-step cold run peaking at 47.5% at step 1100. Within 2 minutes, **3L is optimal.**

### 13e. Longer Context (context_size 32 → 64 BPE tokens, n_steps=401)
*Single change from 13b:* `context_size 32 → 64, n_steps 1201 → 401`

| Step | Loss | Acc | Time |
| ---: | ---: | ---: | ---: |
| 0 | 8.7146 | 7.6% | 0.5s |
| 100 | 3.7437 | 29.5% | 32.5s |
| 200 | 3.3762 | 35.1% | 61.9s |
| 300 | 3.1576 | 37.3% | 90.0s |
| 400 | 3.1592 | **37.7%** ⭐ | 120.7s |

**Training time: 120.7s** *(~30s/100 steps — 2.5× slower than ctx=32)*

> 💡 **Lower accuracy, higher story quality.** ctx=64 BPE ≈ 40–50 words gives the model enough context to produce story-level conclusions ("The moral of the story is...") and descriptive scene-setting absent from all ctx=32 runs. Character swap mid-paragraph persists — 401 steps is too few for the model to learn long-range coreference with the wider window.
>
> 💡 **ctx=64 is an out-of-budget qualitative experiment.** ~30s/100 steps means only ~400 steps fit in 2 minutes — not enough to converge. Best framed as a "what if you had 5 minutes" variant. Within the 2-minute constraint, ctx=32 remains optimal.

### 13f. Larger Vocab on 10k (vocab=4000 → 6000, n_steps=1201)
*Single change from 13b:* `vocab_size 4000 → 6000` → `params: 5,455,472`

| Step | Loss | Acc | Time |
| ---: | ---: | ---: | ---: |
| 0 | 9.0729 | 8.0% | 0.3s |
| 500 | 2.9268 | 41.4% | 80.5s |
| 900 | 2.6630 | **45.7%** ⭐ | 143.1s |
| 1200 | 2.4774 | 45.0% | 188.8s |

**Training time: 188.8s**

> 💡 **vocab=6000 underperforms vocab=4000** (45.0% vs 45.9%) and takes 35% more time (188s vs 140s). Loss still declining at step 1200 — the larger vocab needs ~1600+ steps to beat vocab=4000, which costs ~250s. Not budget-viable.
>
> 💡 **vocab=4000 is definitively optimal for the 2-minute budget.** Tested at both 5k and 10k stories, at 901 and 1201 steps — it consistently outperforms vocab=6000 within the time constraint.

### ✅ Phase 13 Verdict & New Canonical Config

| Experiment | Change | Acc | Time | Verdict |
| :--- | :--- | ---: | ---: | :--- |
| 13a. More data | 5k → 10k stories | 44.8% | 117s | ✅ Better quality, lower raw score |
| **13b. More steps** | 1001 → 1201 | **45.9%** ⭐ | 140s | ✅ New best — sets new canonical |
| 13c. Slower LR tail | eta_min → 3e-4 | 45.7% | 140s | ❌ Noise — sampling variance, not LR |
| 13d. 4 layers | n_layers=4, 801 steps | 44.3% | 124s | ❌ Needs 241s to beat 3L — not budget-viable |
| 13e. ctx=64 | context_size 32→64 | 37.7% | 121s | ⚠️ Better quality, too slow for full run |
| 13f. Larger vocab | vocab 4000→6000 | 45.0% | 189s | ❌ Needs 250s to converge — not budget-viable |

**New canonical `TinyBPE.py` config:** `vocab=4000, num_stories=10000, n_steps=1201, eta_min=1e-4` — all other hyperparameters unchanged. Expected accuracy: **~45.9%†** at **~140s**.

*† Accuracy not comparable to character-level rows — BPE predicts 1 of 4,000 tokens vs 1 of 65 characters.*

---

## 📝 Experiment & Ablation Details

### 🏗️ Theme 1: Architecture Choices (Shape, Size, & Encoding)

**1. Layer Depth (2 vs 3 vs 4 layers)**
- **Result:** 3 layers hit 73.5% in 2200 steps. 4 layers hit 73.1% but took 3400 steps.
- **Takeaway:** 3 layers is the **sweet spot**. Going deeper gives the model more "steps" to process logic, but too deep and training becomes too slow.

**2. Shape: Narrow/Deep vs. Wide/Short**
- **Change:** Halved width (`embed_dim` 256→128, `ffn_dim` 1024→512) and doubled depth (2→4 layers), cutting parameters from 1.6M to 0.8M.
- **Result:** Despite 50% fewer parameters, the taller model beat the baseline by +1.0%.
- **Takeaway:** Depth beats width. A deeper network has more sequential steps to refine features and generalize better.

**3. More Attention Heads (4 → 8)**
- **Change:** `n_heads` 4→8, all else fixed (3L, ctx=32, batch=1536, 5k stories).
- **Result:** Peak **70.5%** at step 1800, training time **182.8s**.
- **Takeaway:** Doubling attention heads added overhead without unlocking new capacity — the ~70% ceiling is an architecture-wide information limit, not an attention-head bottleneck.

**4. Wider Embeddings (embed_dim 256 → 320)**
- **Change:** `embed_dim` 256→320, growing params from 2.41M to 3.26M (+35%). Required lowering `lr` to 1e-3 and adding gradient clipping.
- **Result:** Peak **70.5%** at step 1600, training time **218.0s**.
- **Takeaway:** 35% more parameters, zero improvement. Combined with the batch and heads experiments, this definitively proves the ~70% ceiling is an **information bottleneck** (ctx=32 ≈ 5 words), not a capacity bottleneck. To break it, you need subword tokenization — not a bigger model.

> 💡 **Three approaches, one ceiling:** Bigger batch (+33% data/step), more heads (+100% attention patterns), and wider model (+35% parameters) all converged on **exactly 70.5%**. The signal ceiling of a 5-word character window cannot be overcome by scaling alone.

**5. Positional Embeddings (Ablation)**
- **Change:** Removed the code that tells the model the order of letters.
- **Result:** Accuracy crashed by 7.7%.
- **Takeaway:** Without positional embeddings, the transformer sees all letters simultaneously with no left-to-right concept — pure "word soup." Order matters.

---

### ⚡ Theme 2: Speed, Training, & Optimization Hacks

**1. Weight Tying (Parameter Sharing)**
- **Change:** Forced the input embedding and output linear head to share weights (`linear.weight = tok_embed.weight`).
- **Result:** Accuracy dropped 3.0%, and starting loss exploded above 250 at step 0.
- **Takeaway:** Weight tying saves millions of parameters in large-vocabulary models (50k+ tokens). On our 65-character alphabet, it just confuses the model — the two layers have mismatched initialization needs.

**2. Context Size (8 vs 64 characters)**
- **Change:** Expanded the context window from 8 to 64 characters.
- **Result:** Accuracy barely moved (+1.1%), but training time exploded from 25s to 197s.
- **Takeaway:** Attention math scales quadratically — 8× the context = 7.8× the time. This is exactly why Flash Attention was invented.

**3. Precision: Float16 vs Bfloat16**
- **Change:** Swapped float16 for bfloat16.
- **Result:** bfloat16 was 4.2× slower on the T4 GPU.
- **Takeaway:** The T4 has no physical bfloat16 circuits — it falls back to float32 emulation. Hardware matters.

**4. Activation: GELU vs ReLU**
- **Change:** Swapped ReLU for GELU (used in GPT models).
- **Result:** Identical accuracy, 14% slower.
- **Takeaway:** Don't use complex math if simple math works just as well at this scale.

**5. Loss: Last-Word vs. Full-Sequence Causal Loss**
- **Change:** Applied causal masking to calculate loss across all 8 sequence positions instead of just the last one — producing 8× more training signal per batch.
- **Result:** Much faster early learning (+3.7% at step 200), but the same final accuracy ceiling (~67.6%) at 1.47× the training cost.
- **Takeaway:** Full-sequence loss (the standard GPT approach) improves sample efficiency but doesn't raise the accuracy ceiling for a small architecture.

**6. Flash/SDPA Attention + Context Scaling**
- **Change:** Activated PyTorch's native SDPA with memory-efficient Flash attention kernels; expanded context from 8 to 32 characters on the Narrow-Deep model.
- **Result:** 2.6× faster than naive long-context (confirming O(T²) bottleneck bypassed), but final accuracy improved only +0.2% and training was still 3.2× slower than the 8-char baseline.
- **Takeaway:** Memory-efficient attention works, but context windows are only as useful as the model's capacity to encode them.

**7. LR Warmup + Gradient Clipping**
- **Change:** Added 50-step linear LR warmup and `clip_grad_norm_(params, 1.0)` on every backward pass. Applied to the Phase 4 config (3L, ctx=32, 5k stories, batch=1536).
- **Result:** Peak improved from **70.0% → 70.7%**, but the simpler baseline reached 70.0% faster and with less code.
- **Takeaway:** Valid techniques, but **marginal** at this scale. Useful as a teaching example, not as a necessary addition.

**8. Weight Tying at BPE Scale (vocab=50,257)**
- **Change:** `linear.weight = tok_embed.weight` on the Phase 9 BPE config, tested with and without LR warmup.
- **Result:** Loss=254 at step 0 in every attempt; the model never matched the untied baseline within budget.
- **Takeaway:** Weight tying is the standard GPT-2/LLaMA memory-saving trick, but it requires careful joint initialization. In this architecture it destabilizes training at large vocab sizes (50k+) just as it hurt accuracy at small vocab (65). Verdict: ❌ don't tie weights here, for either vocab size.

---

### 🧠 Theme 3: The "Real Intelligence" Push (Batch, Context & Data)

**1. Large Batch + High LR (3 Layers, 2048 batch)**
- **Change:** Doubled batch size (1024→2048) and learning rate (1e-3→2e-3).
- **Result:** New best raw score: **76.1% at step 2200**.
- **Takeaway:** More data per step mattered more than more parameters — on this dataset size.

**2. The Memorization Trap (Dataset Size: 1k → 3k/5k)**
- **Change:** Expanded `num_stories` from 1,000 to 5,000.
- **Result:** Raw accuracy dropped from 76.1% to 71.4%. But the generated text improved dramatically — the 76.1% model produced word salad; the 5k-story model produced clean clauses.
- **Takeaway:** With only 1,000 stories, the model sees the same evaluation stories so often it memorizes the answers. Expanding the dataset forces it to learn the actual rules of English grammar.

**3. Context is King for Semantics (8 → 16 → 32 characters)**
- **Change:** Doubled context size twice, giving the model a ~5-6 word short-term memory.
- **Result:** The model stopped swapping pronouns mid-sentence. It could finally remember "named Lily" long enough to correctly use "She" a few words later.
- **Takeaway:** 8 characters is barely 1.5 words — the model can't see the subject by the time it writes the verb. 32 characters fixes the "amnesia" while still fitting inside a 2-minute Colab run.

**4. Mild Weight Decay & Inference Temperature**
- **Change:** Added `weight_decay=0.01` to the optimizer; lowered generation temperature from 0.7 to 0.5.
- **Result:** Weight decay stopped the model from repeating phrases. Lower temperature eliminated fake words like "throbe" (→ "robe").
- **Takeaway:** Training is only half the battle. A little regularization during training and conservative sampling during generation polish the final output.

**5. Large Batch (2048) on 5k Stories (Capacity Ceiling)**
- **Change:** `batch_size` 1536→2048 on the Phase 4 canonical config.
- **Result:** Peak **70.5%**, training time **191.2s** — vs 70.0% / 127.7s for the 1536-batch baseline. Same ceiling, 50% more time.
- **Takeaway:** On 1k stories, bigger batches accelerated memorization. On 5k stories the model is genuinely learning, so optimisation speed is no longer the bottleneck. The ~70% ceiling requires architectural changes to break. *(See Theme 1, items 3–4 for the matching head-count and embedding experiments that hit the same ceiling.)*

**6. BPE Tokenization — Breaking the Information Ceiling**
- **Change:** Replaced the character-level tokenizer with `tiktoken` GPT-2 BPE (`vocab_size=50,257`). Zero architecture changes — only the data pipeline and embedding sizes change.
- **Result:** Loss still falling at step 1800 (no plateau), 28M params (26M in embeddings), training time 211s. Generated text shows full multi-paragraph coherence, proper dialogue, and correct pronoun tracking.
- **Takeaway:** The ~70% character-level ceiling was always an **information ceiling**, not a capacity ceiling. ctx=32 BPE tokens ≈ 100+ characters, giving the model a ~20-word memory vs the previous ~5 words. The architecture didn't change — the information did.

**7. Custom Small-Vocab BPE (vocab=4000 vs 50,257)**
- **Change:** Trained a BPE tokenizer from scratch on the TinyStories corpus (HuggingFace `tokenizers`, `vocab_size=4000`) instead of using GPT-2's pretrained 50,257-token vocab. Added logit softcapping (±15) for stability.
- **Result:** 4,429,472 params (down from 28,159,313), 46.2% accuracy at step 900, 103.6s total training time.
- **Takeaway:** GPT-2's vocab is built for general English text, not a 1,500-word children's-story corpus. A right-sized vocabulary cuts embedding parameters by 84% and roughly halves training time, at a modest accuracy cost versus the oversized vocab. This is the best speed/size tradeoff in the BPE family so far.

---

## 📖 Generated Samples (Seeing is Believing)

**TinyMLP.py (59.4% — Letters work, words are broken)**
> `Once tichec. Ther. She said outned. Sker to. Hif even very the box. It. I mesis momors.`

**SimpleTransformer.py (67.2% — Almost real sentences)**
> `Once there was a faster. They learned the pusiade of the yell socked up and played together.`

**TinyTransformer.py — 3L, batch=2048 (76.1% — Highest raw score, but cheats)**
> `Once there was a little girl named Lily. She saw the new toy. He liked to play with her mom smiled and said, "Hello, Spot saw Tom was very happy with the cake was so smaller saw a big that she was happy`
*(Starts well, turns into word salad — it memorized patterns, not grammar.)*

**TinyTransformer.py — 3L, 2048 batch, 3k stories (71.4% — Generalization Win)**
> `Once there was a great time and she was green and strong. Tim and Sue were so happy that the box opened the bug friends. She was sad and looked for them. He grabbed the box of the went to help his mom came in`
*(Clauses flow much better — structure learned, not just words memorized.)*

**TinyTransformer.py — 3L, 1536 batch, 5k stories, ctx=32 (70.0% — Simplicity Champion 👑)**
> `Once there was a little boy named Tim. He was so happy. The dog was scared and happy. They saw a little girl who liver seen the ball. She lived in a big branch with the ball and went to the park. They are happy and started to share`
*(Cleanest code, fastest training at 127.7s, same ~70% ceiling as more complex variants.)*

**TinyTransformer.py — 3L, 2048 batch, 5k stories, ctx=32 (70.5% — Capacity Ceiling Confirmed)**
> `Once there was a little boy named Tim. Tim was very happy to have fun! The dog said, "I will help you find her friends. The boy and her friends with the road. She saw a big ball. They saw a big back to the tree. They played together`
*(Indistinguishable from the 1536-batch run despite 50% more training time.)*

**TinyTransformer.py — 3L, 1536 batch, 5k stories, ctx=32, warmup+clip (70.7% — Stability Variant)**
> `Once there was a little boy named Tim. Tim was so happy to ho excited to show the water. The bird said, "Thank you sad and wanted to be kind out the little girl became good friends. They liked to play with the park.`
*(Smoother training curve, but output quality not clearly better than the simpler version.)*

**TinyTransformer.py — 3L, 1536 batch, 5k stories, ctx=32, 8 heads (70.5% — Head Count Test)**
> `Once there was a little boy named Tim. He thought about the dog because he was happy. They all lived happily ever after.Once upon a time, there was a little boy named Tim. Tim had a big bug and always be friends and they were so sad`
*(Loops and repeats story openings — accuracy held, coherence did not improve.)*

**TinyTransformer.py — 3L, 1536 batch, 5k stories, ctx=32, embed=320 (70.5% — Capacity Ceiling Proof)**
> `Once there was a little boy named Tim. He was so happy to share best climbed the pictures. She was time, they went to the park. It said, "Thank you, they go to the park. They are happy to have a new friends. The bird was sad and sai`
*(35% more parameters, same 70.5% ceiling, 70% more training time.)*

**TinyTransformer.py — 3L, BPE tiktoken, ctx=32 tokens, 5k stories (50.9%† — Ceiling Broken 🚀)**
> `Once there was a little boy named Jack. He was only three years old and had lots of things he wanted to do. One day he saw something very special and he wanted to take it home.`
> `His mom said to his mom, "Mom, can you have some cookies?"`
> `Sam smiled and nodded. He said, "Yes, please. I will be careful."`
> `Lily and Ben smiled. They had fun. They were happy. They had a fun day at the park.`
*(Full multi-paragraph structure, working dialogue, consistent pronouns — the ceiling is gone.)*

*† Raw accuracy not comparable to character-level models. See [Scientific Method](#-the-scientific-method-how-we-trust-our-data) warning above.*

**TinyTransformer.py — 3L, BPE tiktoken, batch=2048, 1401 steps (50.0%† — Short-Budget BPE Champion 🏆)**
> `Once there was a little boy named Jack. He was only three years old and had lots of things he wanted to do. One day he saw something very special and he couldn't wait to find out he was very excited.`
> `The little boy was so excited! He decided to take the track home, and soon enough he got to his mom and said, "Let's go home now!"`
> `They held the basket until they found out it was too late. Tim's mom tried to use the other side of the bush, but it was too late. And they both went home, and they played together in the park. The sun was shining,`
*(Full paragraphs and dialogue, reached in fewer steps than Phase 9 — best speed/quality tradeoff for BPE so far.)*

**TinyTransformer.py — 3L, custom BPE vocab=4000, batch=2048, 901 steps (46.2%† — 2-Min Champion ⚡)**
> `Once there was a little boy named Jack. He was only three years old and had lots of things he wanted to do. One day he saw something very special.`
> `The boy's parents had an idea. He felt so happy, and he wore a nice story together.`
> `The elderly thanked his mom and dad for help. They learned to be careful, when they were ready to eat.`
> `Once upon a time, there was a little dog named Max. Max was a very good friend. He loved to play with his friends. One day, Max found a big, shiny thing. He thought it was very good at it.`
*(Clean decoded output, multiple story arcs, only 4.43M params — best size-to-quality ratio tested.)*
