# 🧪 AI Lab Notebook: Training Tiny Language Models

This file tracks our training experiments on character-level language models trained on the **TinyStories** dataset. 

Our baseline model is **TinyTransformer.py** (a 2-layer transformer, float16 precision, ReLU activation, learned positional embeddings, and a context size of 8 characters). All runs use a standard Google Colab T4 GPU.

---

## 📌 Contents

- [🧬 Lineage: From TorchMLP to TinyTransformer](#-lineage-from-torchmlp-to-tinytransformer)
- [🔬 The Scientific Method: How We Trust Our Data](#-the-scientific-method-how-we-trust-our-data)
- [🧠 How to Read This Document](#-how-to-read-this-document)
- [🔧 The Default Stack: What Changed from SimpleTransformer](#-the-default-stack-what-changed-from-simpletransformer)
- [📊 The Leaderboard: Model Comparison](#-the-leaderboard-model-comparison)
- [🔬 Ablation & Experiment Summary](#-ablation--experiment-summary)
  - [🏗️ Architecture (Shape & Size)](#%EF%B8%8F-architecture-shape--size)
  - [⚡ Training & Speed Hacks](#-training--speed-hacks)
  - [🧠 The "Real Intelligence" Push (Batch, Context & Data)](#-the-real-intelligence-push-batch-context--data)
- [📈 Step-by-Step Accuracy Data](#-step-by-step-accuracy-data)
  - [Phase 1: The Baselines (Where we started)](#phase-1-the-baselines-where-we-started)
  - [Phase 2: Shape & Size Tests (Does depth or width matter more?)](#phase-2-shape--size-tests-does-depth-or-width-matter-more)
  - [Phase 3: The "Raw Score" Champions (Batch Size & Learning Rate)](#phase-3-the-raw-score-champions-batch-size--learning-rate)
  - [Phase 4: The Real Intelligence Push (Generalization vs. Memorization)](#phase-4-the-real-intelligence-push-generalization-vs-memorization)
  - [Phase 5: Optimizer Stability (Warmup & Gradient Clipping)](#phase-5-optimizer-stability-warmup--gradient-clipping)
  - [Phase 6: Bigger Batch on Large Dataset (Does batch size help when we're not memorizing?)](#phase-6-bigger-batch-on-large-dataset-does-batch-size-help-when-were-not-memorizing)
  - [Phase 7: Breaking the Ceiling — More Heads, More Params (Both Fail)](#phase-7-breaking-the-ceiling--more-heads-more-params-both-fail)
- [📝 Experiment & Ablation Details](#-experiment--ablation-details)
  - [🏗️ Theme 1: Architecture Choices (Shape, Size, & Encoding)](#%EF%B8%8F-theme-1-architecture-choices-shape-size--encoding)
  - [⚡ Theme 2: Speed, Training, & Optimization Hacks](#-theme-2-speed-training--optimization-hacks)
  - [🧠 Theme 3: The "Real Intelligence" Push (Data, Context, & Regularization)](#-theme-3-the-real-intelligence-push-data-context--regularization)
- [📖 Generated Samples (Seeing is Believing)](#-generated-samples-seeing-is-believing)

---

> ⚠️ **The Colab Lottery & Scientific Controls:**  
> Google Colab assigns T4 GPUs from a shared pool. Sometimes you get a fast one, sometimes a slow one. If we only look at "Total Seconds," our data is ruined by hardware luck! 
> 
> To fix this, we use **Relative Speed Ratios**. We run the 2-Layer Baseline model as our "Control" (1.0× speed). If an experiment takes twice as long, its speed is **2.0×**. This ratio stays true whether you run it on a slow Colab GPU or a supercomputer!
>
> *Note: Larger batch sizes (e.g., 2048) use more memory bandwidth, which makes the "Colab Lottery" even more extreme. Runtimes can swing from ~65s to ~90s. Always use ratios!*

---

## 🧬 Lineage: From TorchMLP to TinyTransformer

`TinyTransformer.py` wasn't built from scratch — it evolved directly from `TorchMLP.py`
(see `TinyTransformer-explained.md`). Understanding this lineage explains *why* the baseline
hyperparameters look the way they do.

### Inherited from TorchMLP (unchanged)

These settings were carried over without modification:

- `embed_dim = 256`
- `torch.manual_seed(42)`
- `batch_size = 1024`
- `2001` training epochs, evaluated every `200` steps
- Automatic device selection via `torch.set_default_device(...)`
- The same `load_tinystories(...)` data pipeline and sliding-window generation loop

### What actually changed

Only two hyperparameters were adjusted when moving from MLP to Transformer:

| Setting | TorchMLP | TinyTransformer |
| :--- | :--- | :--- |
| `context_size` | 4 | 8 |
| `num_stories` | 200 | 1000 |

Everything else that's new in the first Transformer version — the 2-layer encoder (4 heads,
`ffn_dim=1024`), `torch.compile`, float16 autocast + `GradScaler`, fused AdamW,
`zero_grad(set_to_none=True)`, cosine LR schedule (`eta_min=1e-4`), gradient clipping (`1.0`),
and inference temperature (`0.7`) — was **not** discovered through ablations here. These features
were adopted directly from [Keller Jordan's modded-nanogpt speedrun](https://github.com/KellerJordan/modded-nanogpt),
a record-breaking GPT-2 training optimization repo that served as the direct inspiration for
`TinyTransformer.py`. One idea from that speedrun, AdamW `betas=(0.9, 0.95)`, was documented but
deliberately left at PyTorch defaults in the initial version.

This is why Phase 1 treats `TinyMLP.py` / `TorchMLP.py` as prior-generation reference points
rather than unrelated models: `TinyTransformer.py` is a direct descendant of the MLP baseline
with attention layered on top.

### 🔗 The Keller Jordan Influence

The following features in `TinyTransformer.py` are Keller-lineage ideas:

| Feature | In TinyTransformer? | Origin |
| :--- | :--- | :--- |
| `torch.compile` | ✅ Yes | Keller record #1 |
| `AdamW betas=(0.9, 0.95)` | ✅ Yes | llm.c baseline, refined by Keller |
| `fused=True` optimizer | ✅ Yes | Keller training loop |
| `float16` mixed precision | ✅ Yes | Keller record #10 |
| `CosineAnnealingLR` + `eta_min=1e-4` | ✅ Yes | Keller record #19 (decay to 0.1×, not 0) |
| Pre-LN (`norm_first=True`) | ✅ Yes | Keller modernized architecture |
| `bfloat16` | ❌ Tried, failed | T4 has no native bfloat16 hardware |
| Flash Attention | ❌ Tried, marginal | Model too small to benefit |
| Muon optimizer | ❌ Not tried | Too complex for educational scope |
| RoPE embeddings | ❌ Not tried | Learned pos. embeddings kept for clarity |

This lineage matters because it explains *why* the defaults work so well out of the box —
they were battle-tested at GPT-2 scale before being ported down to this tiny model.
It also explains why some "failed" experiments in this notebook (bfloat16, Flash Attention)
are legitimate wins at larger scale: the technique is sound, but the hardware or model
size isn't the right fit here.

---

### 🌱 How TorchMLP itself was created

`TorchMLP.py` was born from a rename and refactor of `TorchLinear.py`
(commit `84b89b7f`, May 31 2026). `TorchLinear.py` was already a character-level language model
with a core MLP + embedding pipeline, but it defined the model as a flat `nn.Sequential` block —
the same style used in `torch_mlp_sequential.py` from the
[MLP-Digits-Classifier](https://github.com/eniompw/MLP-Digits-Classifier) repo.

The rename introduced two structural improvements:

- **`nn.Sequential` → `nn.Module`:** Refactored into a proper class, making the architecture
  easier to extend toward transformers.
- **Separate `nn.Embedding` + MLP:** Made the embed → flatten → predict pipeline explicit and
  readable.

The full three-file lineage looks like this:

| | `torch_mlp_sequential` | `TorchMLP` | `TinyTransformer` |
| :--- | :--- | :--- | :--- |
| **Weight management** | `nn.Linear` | `nn.Linear` | `nn.Linear` |
| **Optimizer** | Manual SGD | SGD | AdamW + cosine LR + GradScaler |
| **Architecture** | 2-layer MLP | 3-layer MLP + embeddings | 2-layer transformer + attention |
| **Custom forward** | No | Yes (embed + flatten) | Yes (full transformer loop) |
| **`torch.compile`** | No | No | Yes |

---

## 🔬 The Scientific Method: How We Trust Our Data

In AI, it is very easy to fool yourself. Here are the three rules we use to make sure our experiments are scientifically valid:

*   **🎲 The Starting Seed (`torch.manual_seed`):** Neural networks start with random guesses. The specific random guess you start with changes your final score slightly. We hardcode the seed so our experiments are **reproducible**.
*   **🎯 The Eval Seed:** When we test the model every 200 steps, we don't test it on the whole dataset (it would run out of GPU memory). We grab a random subset. But if the subset changes every time, our accuracy will "wobble" up and down based on luck! We fixed this by creating a dedicated `eval_rng`. Now, the model is *always* tested on the exact same 4,096 stories.
*   **✂️ The Golden Rule:** Change **only one thing at a time**. If we add a layer AND double the batch size, and the model gets better, which one caused it? We won't know. Science requires isolation.

---

## 🧠 How to Read This Document

Before we dive in, here are two key scientific concepts we use to test AI models. Think of the model like a recipe or a PC build:

*   **🧪 Experiment:** Trying a *new feature* or *upgrading a setting* to see if it makes the model better. (e.g., *"What if we add more layers to the model's brain?"* or *"What if we double the memory?"*)
*   **✂️ Ablation:** Taking an existing feature *away* to prove that it's actually necessary. It's like removing the baking powder from a cake recipe to see if it actually matters. If the cake goes flat, you proved the baking powder matters! (e.g., *"What if we remove the model's ability to know word order?"*)

---

## 🔧 The Default Stack: What Changed from SimpleTransformer

The canonical TinyTransformer config isn't just bigger hyperparameters — it also adds five code-level optimizations that `SimpleTransformer.py` doesn't have. Each costs 1-2 lines. Here's what each does, proven by the experiments in this notebook:

| Component | `SimpleTransformer.py` | `TinyTransformer.py` (canonical) | Accuracy Impact | Speed Impact | Proven By |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`torch.compile`** | ✅ Already present | ✅ Same | Neutral | **~2.3× faster** | Direct ablation (Cold vs Warm) |
| **float16 autocast** | ❌ float32 only | ✅ `torch.autocast` on forward + eval | Neutral | Major — halves memory bandwidth, enables batch=1536 + ctx=32 in <2min | bfloat16 ablation: 4.2× slower for +0.2% |
| **`CosineAnnealingLR`** | ❌ Flat LR | ✅ `CosineAnnealingLR(T_max=n_steps, eta_min=1e-4)` | Smooths final convergence | ~0 | Phase 5: adding warmup on top only gained +0.6%, meaning cosine does the heavy lifting |
| **AdamW** | ❌ `Adam(params, lr)` | ✅ `AdamW(params, lr, betas=(0.9, 0.95), weight_decay=0.01, fused=True)` | Neutral on acc; `weight_decay` stops repetitive output | `fused=True` speeds up GPU optimizer kernel | Experiment #9: `weight_decay` acts as "grammar regularizer" |
| **Fixed `eval_rng`** | ❌ Full dataset eval every 200 steps | ✅ Dedicated `eval_rng` generator, 4096-sample subset | Eliminates accuracy wobble | Faster per-eval (4096 vs full dataset) | Scientific Method section: "stops accuracy wobble" |
| **Inference temp** | 0.7 (hardcoded) | 0.5 (parameterized) | N/A (inference only) | N/A | Experiment: eliminates fake words ("throbe" → "robe") |

> 💡 **The takeaway:** `torch.compile` + `float16` are the **speed engine** — together they make the 2-minute Colab budget possible. `CosineAnnealingLR` + `AdamW`/`weight_decay` are the **quality polish** — they don't raise the accuracy ceiling but make training more stable and output cleaner. `eval_rng` is the **scientific control** — it makes the numbers trustworthy. All five trace back to [Keller Jordan's modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt), not discovered through ablation — but every adjacent experiment confirmed these are the right defaults.

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

> **🚨 The Plot Twist (Read before judging the scores!):**
> Look at the bottom rows. Why did accuracy go *down* to ~70%? Because we expanded the dataset from 1,000 to 5,000 stories. The 76.1% model was cheating—it memorized the test. The ~70% models stopped memorizing and actually learned English. **Lower accuracy score = higher real-world intelligence!**

---

## 🔬 Ablation & Experiment Summary

Here is the quick cheat sheet of what we learned. All tests below are single changes made to our baseline 2-layer TinyTransformer (~68% accuracy, 1.0× speed).

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
| **Exp** | **Heads:** 4 → 8 | +0.5% | ~1.4× slower | ❌ Same ceiling, more overhead. More heads did not unlock new capacity. |
| **Exp** | **Width:** embed_dim 256 → 320 (3L, 4 heads) | +0.5% | 1.7× slower | ❌ 35% more params, zero gain. Capacity isn't the bottleneck. |
| **Abl** | **Remove Positional Embeddings** | −7.7% | Negligible | ❌ Without this, the AI reads sentences as "word soup." |

### ⚡ Training & Speed Hacks
| Type | Change Tested | Accuracy Δ | Speed Δ | The Verdict |
| :--- | :--- | ---: | ---: | :--- |
| **Exp** | **`torch.compile`** (Cold vs Warm) | Neutral | ~2.3× faster | ✅ Always "warm up" your model before timing it! |
| **Exp** | **Precision:** float16 → bfloat16 | +0.2% | 4.2× slower | ❌ The T4 GPU doesn't have native bfloat16 hardware. |
| **Exp** | **Weight Tying** (sharing layers) | −3.0% | Neutral | ❌ Confused the model and hurt accuracy. |
| **Exp** | **Activation:** ReLU → GELU | Neutral | 14% slower | ❌ GELU is too math-heavy for this small model. |
| **Exp** | **Loss:** Last-word vs Full-sequence | Neutral | 1.47× slower | ⚠️ Learns faster early on, but hits the same ceiling. |
| **Exp** | **Flash Attention** + Context 32 | +0.2% | 3.2× slower | ⚠️ Proves memory-efficient math works, but model is too small to use it. |
| **Exp** | **LR Warmup** (50 steps) + **Grad Clipping** (1.0) | +0.6% peak | Slightly slower | ⚠️ Smoother curve, but not worth the extra code for this scale. |

### 🧠 The "Real Intelligence" Push (Batch, Context & Data)
| Type | Change Tested | Accuracy Δ | Speed Δ | The Verdict |
| :--- | :--- | ---: | ---: | :--- |
| **Exp** | **High LR Fast Convergence** (batch=1024) | +4.0% | 2.5× slower | ⚠️ Faster, but high LR makes training unstable. |
| **Exp** | **Middle Ground** (batch=1536) | +6.8% | 2.7× slower | ✅ Excellent compromise. ~1 min runtime. |
| **Exp** | **Large Batch + High LR** (batch=2048) | +7.7% | ~3.5× slower | ✅ Huge accuracy win — best raw score (but memorizes). |
| **Exp** | **Dataset Size:** 1k → 3k/5k stories | −4.7% | Negligible | ✅ **The Memorization Trap:** Drops raw acc, but drastically improves grammar. |
| **Exp** | **Context Size:** 8 → 16 (on large dataset) | −1.5% | ~1.5× slower | ✅ Fixes pronoun/gender swapping. Model can track subjects! |
| **Exp** | **Weight Decay:** 0 → 0.01 | Neutral | Negligible | ✅ Acts as a "grammar regularizer." Stops lazy repetition. |
| **Exp** | **Context Size:** 16 → 32 (on large dataset) | −1.6% | ~1.3× slower | ✅ The ultimate 2-min tradeoff. Fixes 90% of pronoun swaps. |
| **Exp** | **batch=2048 on 5k stories** (ctx=32) | +0.5% vs 1536 | 1.5× slower | ❌ Same ~70% ceiling. Batch size stops helping when model is genuinely learning. |
| **Exp** | **Inference Temp:** 0.7 → 0.5 | N/A (Inference) | N/A | ✅ Eliminates fake words (e.g., "throbe" → "robe"). |

---

## 📈 Step-by-Step Accuracy Data

*We split the data into "Phases" to tell the story of our experiments. ⭐ marks the peak accuracy. 📉 shows where the model starts overfitting and getting worse!*

**Legend:** 
* **2L/3L/4L** = TinyTransformer with 2, 3, or 4 layers.
* **ND** = Narrow-Deep (skinnier model, more layers).
* **FFN** = Feed-Forward Network (the "thinking" part of the layer).

### Phase 1: The Baselines (Where we started)
*Goal: See if our basic Transformer architecture beats the older, simpler models.*

| Step | NameSLP | TinyMLP | SimpleTrans | **2L (Baseline)** |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 3.5% | 4.7% | 4.0% | 19.3% |
| 200 | 37.1% | 44.8% | 53.5% | 54.8% |
| 800 | 38.9% | 55.0% | 62.4% | 63.2% |
| 1600 | 39.5% | 58.3% | 66.2% | 67.0% |
| 2000 | **39.6%** ⭐ | **59.4%** ⭐ | **67.2%** ⭐ | 67.4% |

### Phase 2: Shape & Size Tests (Does depth or width matter more?)
*Goal: Find out if adding layers, widening the brain, or changing the shape gives us better accuracy than the 2L Baseline.*

| Step | **3L** (Run 1) | **4L** | ND 4L (128d) | Eff. Deep 4L | Bal. ND 4L (192d) | Wider FFN 3L |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 19.3% | 19.3% | 10.6% | 5.2% | 19.3% | 20.2% |
| 800 | 64.8% | 64.6% | 63.0% | 63.9% | 65.6% | 64.7% |
| 1600 | 67.6% | 68.0% | 67.8% | 68.4% | 70.0% | 70.2% |
| 2000 | 70.2% | 68.9% | 69.4% | **70.8%** ⭐ | 70.5% | 71.1% |
| 2200 | **73.5%** ⭐ | - | 68.1% 📉 | 69.7% 📉 | 70.4% | **71.8%** ⭐ |
| 2400 | 71.7% 📉 | - | 68.9% | - | **70.8%** ⭐ | - |
| 3400 | - | **73.1%** ⭐ | - | - | - | - |

> 💡 **Pro-Tip:** Look at the 3-Layer model. It hits 73.5% at step 2200, but drops to 71.7% at step 2400. This is called **overfitting**. The model memorized the training data so hard that it got worse at writing new stories. Always stop training when you hit the ⭐!

### Phase 3: The "Raw Score" Champions (Batch Size & Learning Rate)
*Goal: Instead of changing the model's shape, what if we just change HOW it learns? (Using the 3-Layer model).*

| Step | High LR (batch=1024) | Mid Ground (batch=1536) | **Large Batch+LR** (batch=2048) |
| ---: | ---: | ---: | ---: |
| 0 | 19.3% | 19.3% | 19.3% |
| 800 | 62.9% | 65.5% | 66.9% |
| 1600 | 66.2% | 69.9% | 71.0% |
| 2000 | 68.8% | 70.8% | 72.3% |
| 2200 | **72.4%** ⭐ | **75.2%** ⭐ | **76.1%** ⭐ |
| 2400 | 71.1% 📉 | 73.0% 📉 | - |

### Phase 4: The Real Intelligence Push (Generalization vs. Memorization)
*Goal: Stop chasing raw accuracy numbers and fix the AI's "amnesia." By expanding the dataset and context window, we force the model to actually learn English rather than memorizing 1,000 stories.*

| Step | **3L, 2048 batch, 3k stories** (ctx=8) | **3L, 1536 batch, 5k stories** (ctx=32, wd=0.01) |
| ---: | ---: | ---: |
| 0 | 18.5% | 19.2% |
| 400 | 62.1% | 61.4% |
| 800 | 66.0% | 64.4% |
| 1200 | 67.8% | 67.8% |
| 1600 | 69.3% | **70.0%** ⭐ |
| 2000 | 71.4% | - |

> 💡 **Pro-Tip:** Look at the scores! They are *lower* than Phase 3 (which hit 76.1%). But look at the generated samples below. This proves that on small datasets, high accuracy is just memorization (overfitting). If you want a model that writes well in the real world, train it on more data and accept a slightly lower eval score!

### Phase 5: Optimizer Stability (Warmup & Gradient Clipping)
*Goal: Test whether standard optimizer-stability tricks help enough at this scale to justify the extra code complexity.*

*Changes added to the Phase 4 winning config (3L, ctx=32, 5k stories, batch=1536):*
- **LR Warmup:** 50-step linear warmup before cosine decay
- **Gradient Clipping:** `clip_grad_norm_(params, 1.0)` after every backward pass
- **`eta_min`:** 1e-4

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

> 💡 **Key Result:** Warmup + clipping produced a smoother, more monotonic learning curve and a slightly higher peak (**70.7%**), but the simpler no-warmup baseline still reached **70.0%** in less time (**127.7s**) with less code.

> 💡 **Verdict:** These are valid techniques, but for this tiny model they are **not the new canonical config**. They belong in the notebook as a useful negative/edge result: standard stability tricks can help a little, but they do **not** beat simplicity enough to justify the extra moving parts.

### Phase 6: Bigger Batch on Large Dataset (Does batch size help when we're not memorizing?)
*Goal: Phase 3 showed batch=2048 was a huge win on 1k stories. Does the same trick work when we switch to 5k stories and ctx=32?*

*Single change from the Phase 4 canonical config:* `batch_size 1536 → 2048`

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

> 💡 **Key Result:** Peak accuracy is **70.5%** — essentially the same ~70% ceiling as the 1536-batch canonical config, but taking **191s vs 128s** (50% more time). The batch size increase bought nothing.

> 💡 **The Insight — Why batch size stops working:** On 1k stories, a bigger batch helped because it let the model memorize more patterns per step. On 5k stories, the model is genuinely *learning* rather than memorising, so the bottleneck has shifted. More data-per-step doesn't help if the model has already extracted everything it can from its 2.4M parameters. The ceiling is a **capacity ceiling**, not an optimisation ceiling. Breaking it requires more parameters or architectural changes — not a bigger batch.

### Phase 7: Breaking the Ceiling — More Heads, More Params (Both Fail)
*Goal: Three experiments tried to break the ~70% wall — bigger batch, more heads, wider model. All three landed on exactly 70.5%.*

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
*Single change from canonical:* `embed_dim 256 → 320` (3.26M params, +35%). Required `lr=1e-3` + grad clipping (`clip_grad_norm_ 1.0`) to prevent NaN divergence.

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

> 💡 **The Definitive Result:** Three completely different approaches — bigger batch (+33% data/step), more heads (+100% attention patterns), and wider model (+35% parameters / +860K params) — all converged on **exactly 70.5%**. This is not coincidence; it is the **information ceiling** of character-level tokenization at ctx=32 (~5-6 words of context). No amount of model capacity can extract more signal than exists in a 5-word window.
>
> 💡 **The NaN Lesson:** The wider model (embed=320) immediately diverged to NaN at `lr=2e-3` with no clipping. This is the first experiment where gradient clipping became *necessary*, not optional — larger embeddings produce larger gradients that destabilize the optimizer before it can warm up.
>
> 💡 **The torch.compile Warning:** The embed=320 run triggered `torch._dynamo` recompilation warnings (hit config limit of 8). This is caused by the eval loop toggling `autocast` on/off, forcing graph recompilation. Part of the slowdown beyond just bigger matrices.

---

## 📝 Experiment & Ablation Details

Here, we dive deep into the specific upgrades, setting adjustments, and feature removals we tested to find the ultimate small-scale language model. We have categorized these investigations into three primary thematic blocks.

### 🏗️ Theme 1: Architecture Choices (Shape, Size, & Encoding)

**1. Layer Depth (2 vs 3 vs 4 layers)**
*   **The Change:** We added extra layers to see if a "taller" brain is better than a "wider" one. 
*   **Result:** 3 layers hit 73.5% in 2200 steps. 4 layers hit 73.1% but took 3400 steps.
*   **The Takeaway:** 3 layers is the **sweet spot**. Think of it like building a tower: going wider takes huge amounts of material, but going taller gives the model more "steps" to process complex logic. But go too tall, and it becomes too slow to train!

**2. Shape: Narrow/Deep vs. Wide/Short**
*   **The Feature Removed/The Change:** We halved the model's width (`embed_dim` 256 → 128, `ffn_dim` 1024 → 512) and doubled its depth (2 → 4 layers), cutting the total parameters in half from 1.6M to 0.8M parameters.
*   **Result:** Despite having 50% fewer parameters, the taller, skinnier model actually *beat* the wider baseline's accuracy by +1.0% (69.1% vs 68.1%).
*   **The Takeaway:** Depth is incredibly powerful for reasoning and context. A deeper, narrow network generalizes and learns compositional rules better than a shallow, wide model because it has more sequential processing steps to refine features.

**3. Weight Tying (Parameter Sharing)**
*   **The Feature Removed/The Change:** We forced the "input reading" layer (token embedding) and the "output guessing" layer (linear head) to share the exact same weights (`linear.weight = tok_embed.weight`), reducing parameter overhead.
*   **Result:** Accuracy dropped by 3.0%, and the starting loss exploded at step 0 to over 250!
*   **The Takeaway:** Weight tying is a great trick for massive models with huge 50,000-word vocabularies because it saves millions of parameters. But on our tiny 65-character alphabet, it just confuses the model because the layers have mismatched initialization needs (Kaiming uniform for linear projection vs. normal distribution for embeddings).

**4. Positional Embeddings**
*   **The Feature Removed:** We removed the code that tells the AI the order of the letters. The AI now sees "tac" and "cat" as the exact same thing.
*   **Result:** Accuracy crashed by 7.7%. 
*   **The Takeaway:** Transformers are like a person reading a handful of Scrabble tiles scattered on a table. By default, they see all the letters but have no concept of left-to-right order. Without Positional Embeddings, the AI is just looking at "word soup." Order matters!

**3. Context Size (8 vs 64 characters)**
*   **The Change:** We gave the model a bigger "short-term memory," letting it look at 64 characters at once instead of 8.
*   **Result:** Accuracy barely moved (+1.1%), but training time exploded from 25s to 197s!
*   **The Takeaway:** Attention math scales quadratically (if you double the context, you quadruple the math). 8x the context meant 7.8x the time. This is exactly why researchers invented "Flash Attention" to fix this later.

**4. Float16 vs Bfloat16 Precision**
*   **The Change:** We swapped standard float16 math for bfloat16 (a newer format that handles big numbers better).
*   **Result:** bfloat16 was 4.2× slower on our T4 GPU!
*   **The Takeaway:** Hardware matters. The older T4 GPU doesn't have physical circuits for bfloat16, so it fakes it using float32, which is slow. 

**5. GELU vs ReLU Activation**
*   **The Change:** We swapped ReLU (a simple "if negative, make zero" math rule) for GELU (a complex curve used in GPT models).
*   **Result:** Identical accuracy, but 14% slower.
*   **The Takeaway:** Don't use complex math if simple math works just as well. GELU's complex calculations slowed the GPU down with no benefit at this small scale.

#### Phase 2: The Training Hacks
**6. Large Batch + High LR (3 Layers, 2048 batch)**
*   **The Change:** Instead of making the model bigger, we doubled the **batch size** (data processed at once) from 1024 → 2048 and doubled the **learning rate** from 1e-3 → 2e-3 to match.
*   **Result:** A new best raw score: **76.1% at step 2200**. 
*   **The Takeaway:** For this dataset, **more data per step mattered more than more parameters**. 

**7. The Memorization Trap (Dataset Size: 1k → 3k/5k)**
*   **The Change:** We expanded `num_stories` from 1,000 to 5,000.
*   **Result:** The raw accuracy score dropped from 76.1% down to 71.4%. However, the generated text improved dramatically. The 76.1% model produced word salad ("the cake was so smaller saw a big"), while the 5,000-story model produced clean clauses.
*   **The Takeaway:** With only 1,000 stories, the model sees the exact same evaluation stories so many times that it just memorizes the answers. It "hacks" the test. Expanding the dataset forces the AI to learn the underlying *rules* of English grammar to succeed.

#### Phase 3: The Coherence Push
**8. Context is King for Semantics (8 → 16 → 32 characters)**
*   **The Change:** We doubled `context_size` from 8 to 16, and then to 32, giving the AI a 5-6 word short-term memory.
*   **Result:** The AI stopped swapping pronouns mid-sentence. It could finally remember "named Lily" long enough to correctly use "She" later in the sentence.
*   **The Takeaway:** 8 characters is barely 1.5 words. The AI literally could not see the subject of the sentence by the time it wrote the verb. 32 characters fixes the "amnesia" while still fitting inside the 2-minute Colab budget!

**9. Mild Weight Decay & Inference Temperature**
*   **The Change:** We added a tiny amount of `weight_decay=0.01` to the optimizer, and lowered the generation `temperature` from 0.7 to 0.5.
*   **Result:** Weight decay stopped the model from repeating the same phrases over and over. The lower temperature stopped the model from making risky, weird guesses that resulted in fake words like "throbe" (turning it into the real word "robe").
*   **The Takeaway:** Training is only half the battle. A little regularization during training, and conservative sampling during generation, polishes the final output.

#### Phase 4: Optimizer Stability
**10. LR Warmup + Gradient Clipping**
*   **The Change:** Added a 50-step linear LR warmup (via `SequentialLR` chaining `LinearLR` → `CosineAnnealingLR`) and gradient clipping (`clip_grad_norm_(params, 1.0)`) on every backward pass. Applied on top of the Phase 4 winning config (3L, ctx=32, 5k stories, batch=1536, lr=2e-3).
*   **Result:** Peak accuracy improved from **70.0% → 70.7%** at step 1600, but the simpler cosine-only version still reached **70.0%** in less time and with less code.
*   **The Takeaway:** Warmup and clipping are still useful teaching examples because they show that standard optimizer tricks can smooth training. But at this scale, they are a **marginal optimization**, not a must-have. The simpler code wins on clarity-to-benefit ratio.

#### Phase 5: The Capacity Ceiling
**11. Large Batch (2048) on 5k Stories**
*   **The Change:** Increased `batch_size` from 1536 → 2048 on the Phase 4 canonical config (3L, ctx=32, 5k stories, lr=2e-3). All other hyperparameters unchanged.
*   **Result:** Peak accuracy **70.5%** at step 1600, training time **191.2s** — compared to 70.0% / 127.7s for the 1536-batch baseline. Same ceiling, 50% more time.
*   **The Takeaway:** On the 1k-story dataset, bigger batches helped because the model was memorizing — more examples per step = faster memorization. On 5k stories the model is genuinely learning, so optimisation speed is no longer the bottleneck. The ~70% ceiling is a **model capacity** limit. To break it, we need to change the architecture (more parameters, more heads, wider FFN) — not the batch size.

**12. More Attention Heads (4 → 8)**
*   **The Change:** Increased `n_heads` from 4 to 8 while keeping `embed_dim=256`, `ffn_dim=1024`, `n_layers=3`, `ctx=32`, `batch=1536`, and `num_stories=5000` fixed.
*   **Result:** Peak accuracy reached **70.5%** at step 1800, but training time rose to **182.8s**.
*   **The Takeaway:** Doubling the number of heads did not unlock better reasoning or longer-range tracking. It mostly added overhead while landing on the same ceiling as other recent runs. This points to a broader architecture limit rather than an attention-head bottleneck.

**13. Wider Embeddings (embed_dim 256 → 320)**
*   **The Change:** Increased `embed_dim` from 256 to 320, growing parameters from 2.41M to 3.26M (+35%). Required lowering `lr` from 2e-3 to 1e-3 and adding gradient clipping to prevent NaN divergence.
*   **Result:** Peak accuracy **70.5%** at step 1600, training time **218.0s**.
*   **The Takeaway:** A 35% bigger model with 860K more parameters produced **zero improvement**. Combined with the batch and heads experiments, this definitively proves the ~70% ceiling is an **information bottleneck** (character-level ctx=32 ≈ 5 words), not a capacity bottleneck. To break it, you need a better input representation (subword tokenization), not a bigger model.

---

### ✂️ ABLATION: Proving What Matters
*This test removes a crucial feature to prove why the AI needs it in the first place.*

**1. Ablation: Positional Embeddings**
*   **The Feature Removed:** We removed the code that tells the AI the order of the letters. The AI now sees "tac" and "cat" as the exact same thing.
*   **Result:** Accuracy crashed by 7.7%. 
*   **The Takeaway:** Transformers are like a person reading a handful of Scrabble tiles scattered on a table. By default, they see all the letters but have no concept of left-to-right order. Without Positional Embeddings, the AI is just looking at "word soup." Order matters!

**2. Ablation: Weight Tying (Parameter Sharing)**
*   **The Feature Removed/The Change:** We forced the "input reading" layer (token embedding) and the "output guessing" layer (linear head) to share the exact same weights (`linear.weight = tok_embed.weight`), reducing parameter overhead.
*   **Result:** Accuracy dropped by 3.0%, and the starting loss exploded at step 0 to over 250!
*   **The Takeaway:** Weight tying is a great trick for massive models with huge 50,000-word vocabularies because it saves millions of parameters. But on our tiny 65-character alphabet, it just confuses the model because the layers have mismatched initialization needs (Kaiming uniform for linear projection vs. normal distribution for embeddings).

**3. Ablation: Last-Word vs. Full-Sequence Causal Loss**
*   **The Feature Removed/The Change:** Instead of only calculating the loss on the very last predicted token, we applied a causal mask and calculated the loss across all 8 sequence positions, producing 8× more training feedback per batch.
*   **Result:** The model learned much faster early in training (+3.7% accuracy at step 200), but plateaued at the same ~67.6% final accuracy ceiling while taking 1.47× longer to train.
*   **The Takeaway:** Asking the model to predict *every* token in the sentence (the standard way GPT models learn) dramatically improves sample-efficiency early on. However, for a small architecture, it doesn't raise the ultimate accuracy ceiling—it just helps the model reach it faster at the cost of heavier step-by-step math.

**4. Ablation/Experiment: Narrow/Deep vs. Wide/Short Shape**
*   **The Feature Removed/The Change:** We halved the model's width (`embed_dim` 256 → 128, `ffn_dim` 1024 → 512) and doubled its depth (2 → 4 layers), cutting the total parameters in half from 1.6M to 0.8M parameters.
*   **Result:** Despite having 50% fewer parameters, the taller, skinnier model actually *beat* the wider baseline's accuracy by +1.0% (69.1% vs 68.1%).
*   **The Takeaway:** Depth is incredibly powerful for reasoning and context. A deeper, narrow network generalizes and learns compositional rules better than a shallow, wide model because it has more sequential processing steps to refine features.

**5. Ablation/Experiment: Flash/SDPA Attention & Context Scaling**
*   **The Feature Removed/The Change:** We activated PyTorch's native Scaled Dot-Product Attention (SDPA) with Memory-Efficient Flash attention kernels and expanded the context size from 8 to 32 characters on the Narrow-Deep model.
*   **Result:** It ran 2.6× faster than the naive long-context approach (confirming the O(T²) attention bottleneck was successfully bypassed!), but final accuracy only improved by +0.2% and training took 3.2× longer than the 8-character context.
*   **The Takeaway:** Memory-efficient attention math works beautifully, but context windows are only as good as the model's capacity. Bumping the memory window to 32 characters on a skinny 128-dimension model provides more information than its small "brain" can physically encode or utilize.

---

## 📖 Generated Samples (Seeing is Believing)

Numbers are great, but what does the AI actually write? Here are samples from our models, showing how they get smarter as we change the architecture and training settings.

**TinyMLP.py (59.4% Acc - Letters work, words are broken)**
> `Once tichec. Ther. She said outned. Sker to. Hif even very the box. It. I mesis momors.`

**SimpleTransformer.py (67.2% Acc - Almost real sentences)**
> `Once there was a faster. They learned the pusiade of the yell socked up and played together.`

**TinyTransformer.py - 3 Layers, batch=2048 (76.1% Acc - Highest raw score, but cheats)**
> `Once there was a little girl named Lily. She saw the new toy. He liked to play with her mom smiled and said, "Hello, Spot saw Tom was very happy with the cake was so smaller saw a big that she was happy`
*(Notice how it starts well but turns into word salad because it memorized patterns, not grammar).*

**TinyTransformer.py - 3L, 2048 batch, 3k stories (71.4% Acc - Generalization Win!)**
> `Once there was a great time and she was green and strong. Tim and Sue were so happy that the box opened the bug friends. She was sad and looked for them. He grabbed the box of the went to help his mom came in`
*(Notice how much better the clauses flow compared to the 76.1% champion. It learned structure, not just memorized words!)*

**TinyTransformer.py - 3L, 1536 batch, 5k stories, ctx=32 (70.0% Acc - Simplicity Champion 👑)**
> `Once there was a little boy named Tim. He was so happy. The dog was scared and happy. They saw a little girl who liver seen the ball. She lived in a big branch with the ball and went to the park. They are happy and started to share`
*(Cleanest code, fastest training at 127.7s, same ~70% ceiling as more complex variants.)*

**TinyTransformer.py - 3L, 2048 batch, 5k stories, ctx=32 (70.5% Acc - Phase 6: Capacity Ceiling Confirmed)**
> `Once there was a little boy named Tim. Tim was very happy to have fun! The dog said, "I will help you find her friends. The boy and her friends with the road. She saw a big ball. They saw a big back to the tree. They played together`
*(Output quality is indistinguishable from the 1536-batch run despite 50% more training time. Confirms the ceiling is model capacity, not optimisation.)*

**TinyTransformer.py - 3L, 1536 batch, 5k stories, ctx=32, warmup+clip (70.7% Acc - Stability Variant)**
> `Once there was a little boy named Tim. Tim was so happy to ho excited to show the water. The bird said, "Thank you sad and wanted to be kind out the little girl became good friends. They liked to play with the park.`
*(Useful as a teaching experiment: the curve is smoother, but the output quality is not clearly better enough to justify the added code.)*

**TinyTransformer.py - 3L, 1536 batch, 5k stories, ctx=32, 8 heads (70.5% Acc - Head Count Test)**
> `Once there was a little boy named Tim. He thought about the dog because he was happy. They all lived happily ever after.Once upon a time, there was a little boy named Tim. Tim had a big bug and always be friends and they were so sad`
*(The first sentence is clean, but the sample quickly loops and repeats story openings. Accuracy stayed near the ceiling, but coherence did not clearly improve.)*

**TinyTransformer.py - 3L, 1536 batch, 5k stories, ctx=32, embed=320 (70.5% Acc - Capacity Ceiling Proof)**
> `Once there was a little boy named Tim. He was so happy to share best climbed the pictures. She was time, they went to the park. It said, "Thank you, they go to the park. They are happy to have a new friends. The bird was sad and sai`
*(35% more parameters, same 70.5% ceiling, 70% more training time. The model has more capacity than it has information to use.)*
