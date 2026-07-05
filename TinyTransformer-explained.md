# TinyTransformer Explained

Think of `TinyTransformer.py` as `TorchMLP.py` with its tunnel vision fixed — instead of squinting at 4 characters, it reads a 32-character window and uses attention to weigh which parts matter most.

This walkthrough covers what changed, why it was changed, and what each optimization actually does.

> **Lineage:** `TorchMLP.py` was itself inspired by [MLP-Digits-Classifier](https://github.com/eniompw/MLP-Digits-Classifier) — a minimal MLP trained on handwritten digits — adapted here for character-level language modeling. The optimization techniques in `TinyTransformer.py` were sourced from [MicroGPT](https://github.com/eniompw/MicroGPT), which in turn drew from [Keller Jordan's modded-nanogpt speedrun](https://github.com/KellerJordan/modded-nanogpt). See [BENCHMARKS.md](BENCHMARKS.md) for the full lineage and ablation evidence.

---

## Contents

1. [What changed from TorchMLP](#1-what-changed-from-torchmlp)
2. [Model architecture](#2-model-architecture)
3. [Training loop and optimization stack](#3-training-loop-and-optimization-stack)
4. [Speed optimizations](#4-speed-optimizations)
5. [Accuracy, quality, and scientific controls](#5-accuracy-quality-and-scientific-controls)
6. [Generation behavior](#6-generation-behavior)
7. [Practical scaling notes](#7-practical-scaling-notes)
8. [Summary: What TinyTransformer adds over TorchMLP](#8-summary-what-tinytransformer-adds-over-torchmlp)

---

## 1. What Changed from TorchMLP

The core task is identical: load TinyStories, convert text to character IDs, train a next-character predictor with cross-entropy, and generate text autoregressively. Everything below is what changed on top of that shared foundation.

| Setting / Component | `TorchMLP.py` | `TinyTransformer.py` | Why it changed |
| :--- | :--- | :--- | :--- |
| `context_size` | 4 | 32 | More context = fewer pronoun swaps, better sentence coherence |
| `num_stories` | 200 | 5000 | Larger dataset prevents memorization, forces grammar learning |
| Model block | 3-layer MLP | 3-layer Transformer Encoder (4 heads) | Attention lets the model relate any two positions in the window |
| Positional encoding | None | Learned `pos_embed` | Transformers are order-blind without it (−7.7% accuracy if removed) |
| Optimizer | SGD | AdamW `betas=(0.9, 0.95)`, `weight_decay=0.01`, `fused=True` | Faster convergence, gradient stability, grammar regularization |
| LR schedule | Flat | `CosineAnnealingLR(T_max=n_steps, eta_min=1e-4)` | Smooth late-stage convergence without a hard stop |
| Precision | float32 | float16 autocast | Halves memory bandwidth; enables batch=1536 + ctx=32 in <2 min |
| Graph compilation | None | `torch.compile(...)` | Fuses ops into optimized CUDA kernels (~1.2× faster after warmup) |
| Device setup | Manual `.to(device)` | `torch.set_default_device(device)` | Removes boilerplate and prevents hardware mismatch errors |
| Eval method | Full dataset every 200 steps | Fixed `eval_rng`, 4096-sample subset | Eliminates accuracy wobble from random subset variation |
| Inference temperature | — | `0.5` | Sharpens output distribution, eliminates invented words |

---

## 2. Model Architecture

### Token + Position Embeddings

```python
tok_embed = nn.Embedding(len(idx_to_char), embed_dim)  # shape: (vocab_size, 256)
pos_embed = nn.Embedding(context_size, embed_dim)       # shape: (32, 256)

# Inside forward:
x = tok_embed(batch_x) + pos_embed(torch.arange(context_size))
# x shape: (B, 32, 256)  — batch × sequence × embedding
```

`tok_embed` maps each character ID to a 256-dim vector (what the token *is*). `pos_embed` gives each position its own learned 256-dim vector (where the token *sits*). Their sum is the input to the transformer — the model now knows both identity and position simultaneously.

Without `pos_embed`, the transformer sees all characters as an unordered bag of letters. Our benchmarks show removing it drops accuracy by **−7.7%**.

### Transformer Encoder

```python
transformer = torch.compile(
    nn.TransformerEncoder(
        nn.TransformerEncoderLayer(
            d_model=embed_dim,   # 256 — width of each token's representation
            nhead=n_heads,       # 4 — parallel attention patterns
            dim_feedforward=ffn_dim,  # 1024 — inner width of the FFN sublayer
            batch_first=True,    # tensor shape is (B, T, C), not (T, B, C)
            dropout=0.,          # no dropout — model is already small
            norm_first=True      # Pre-LN: normalize before attention, more stable
        ),
        num_layers=n_layers      # 3 — sweet spot between depth and speed
    )
)
```

`batch_first=True` keeps tensor shapes as `(B, T, C)` throughout — batch size, sequence length, channel width. `norm_first=True` (Pre-LN) applies layer normalization *before* the attention and FFN sublayers, which tends to be more numerically stable during training.

After the encoder, only the final time step is used for prediction:

```python
logits = linear(transformer(x)[:, -1, :])
# transformer(x) shape: (B, 32, 256)
# [:, -1, :]     shape: (B, 256)      — take the last position only
# logits         shape: (B, vocab_size)
```

This mirrors the autoregressive setup: read the full 32-character context, predict only the next character.

---

## 3. Training Loop and Optimization Stack

The training loop is still compact and readable:

1. Sample random mini-batch indices
2. Run forward pass under mixed-precision autocast
3. Compute cross-entropy loss
4. Backprop → optimizer step → scheduler step
5. Every 200 steps: evaluate accuracy on a fixed 4096-sample subset

The key difference from `TorchMLP.py` is that speed and stability techniques are layered together in one loop rather than being added one at a time.

---

## 4. Speed Optimizations

### Global Device Default

```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_default_device(device)
```

Instead of manually calling `.to('cuda')` on every tensor and module, we tell PyTorch to place everything on the GPU by default. Keeps the code clean and eliminates a common source of device-mismatch errors.

### `torch.compile`

```python
transformer = torch.compile(nn.TransformerEncoder(...))
```

Traces the model graph and fuses operations into optimized CUDA kernels, reducing Python overhead and kernel launch fragmentation. Gives roughly **1.2× faster** throughput after an initial ~32-second one-time compilation tax on the first forward call.

> ⚠️ Always run a warmup pass before timing. Benchmarking a cold-start `torch.compile` model makes it look far slower than it actually is (32.5s step 0 vs 0.2s after warmup).

### `float16` Autocast

```python
with torch.autocast('cuda', dtype=torch.float16):
    logits = linear(transformer(x)[:, -1, :])
    loss = F.cross_entropy(logits, batch_y)
```

Runs the forward pass and loss in 16-bit floats, halving memory bandwidth and accelerating matrix math on the T4. This is what makes `batch=1536, ctx=32` fit inside a 2-minute Colab run.

### Fused AdamW

```python
optimizer = torch.optim.AdamW(
    params, lr=lr,
    betas=(0.9, 0.95),
    weight_decay=0.01,
    fused=True           # groups parameter updates into a single CUDA kernel
)
```

`fused=True` combines all parameter update operations into one optimized kernel launch instead of many small ones. The lower second beta `(0.95 vs default 0.999)` makes the optimizer react faster to recent gradient patterns — important for transformer workloads.

---

## 5. Accuracy, Quality, and Scientific Controls

### Cosine LR Schedule with Floor

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=n_steps, eta_min=1e-4
)
```

Starts with large learning rate steps for fast early progress, then smoothly decays to `1e-4` — a floor that prevents the optimizer from grinding to a complete halt late in training. Adding a warmup on top (Phase 5 experiment) only improved peak accuracy by +0.7%, so cosine alone does most of the work.

### Fixed Eval Seed

```python
eval_rng = torch.Generator(device=device).manual_seed(0)

# Inside the eval loop:
eval_idx = torch.randint(0, len(input_ids), (4096,), generator=eval_rng)
```

Every 200 steps the model is evaluated on the same 4,096 stories — not a new random sample each time. Without this, the accuracy curve wobbles up and down based on which stories happened to be sampled, making it impossible to trust small differences between runs.

### `weight_decay=0.01`

Adds a small regularization penalty that discourages any single weight from growing too large. In practice this acts as a **grammar regularizer** — it stops the model from lazily repeating the same phrase over and over in generated text.

### Inference Temperature

```python
next_token_probs = torch.softmax(logits / temp, dim=1)  # temp = 0.5
```

Dividing logits by `0.5` before softmax sharpens the probability distribution: confident predictions become more likely, low-confidence guesses are suppressed. This eliminates invented words like "throbe" (→ "robe") without any retraining.

---

## 6. Generation Behavior

Generation follows the same sliding-window loop as `TorchMLP.py`:

1. Start from the first real context window in the dataset
2. Run forward pass → get next-token probabilities
3. Sample one token using temperature-scaled softmax
4. Append token, drop the oldest, repeat

The differences from `TorchMLP.py`:

| | `TorchMLP.py` | `TinyTransformer.py` |
| :--- | :--- | :--- |
| Context length | 4 chars | 32 chars |
| Model forward | MLP | Transformer encoder |
| Temperature | None | 0.5 (sharpens output) |

---

## 7. Practical Scaling Notes

- **Context size 64** makes training ~7.8× slower for only +1.1% accuracy. Attention scales quadratically — this is exactly the problem Flash Attention was designed to solve.
- **The Colab Lottery** means GPU speeds vary run to run. Use Relative Speed Ratios (comparing against a fixed baseline) rather than absolute seconds to judge whether a change is actually faster.
- **The ~70% ceiling** on the 5k-story dataset is an information bottleneck, not a capacity bottleneck. Three experiments — bigger batch, more heads, wider embeddings — all converged on exactly 70.5%. Breaking it requires subword tokenization, not a bigger model.

Observed run (3-layer, `batch=1536`, 5000 stories, `context_size=32`):
- **Best accuracy:** ~70.1% at step 1600
- **Training time:** ~143.8s (varies by GPU)

---

## 8. Summary: What TinyTransformer Adds over TorchMLP

`TorchMLP.py` is ideal for understanding baseline language-model mechanics. `TinyTransformer.py` keeps that readability and layers on practical modern training techniques:

| Category | What was added | Effect |
| :--- | :--- | :--- |
| **Architecture** | 3-layer Transformer Encoder + positional embeddings | Attention-based context; word-order awareness |
| **Speed** | `torch.compile`, float16 autocast, fused AdamW, global device default | ~2× faster GPU utilization; fits in 2-min Colab budget |
| **Stability** | Cosine LR schedule, `betas=(0.9, 0.95)`, `weight_decay=0.01` | Smoother convergence; stops repetitive output |
| **Reproducibility** | Fixed `eval_rng`, `torch.manual_seed(42)` | Eliminates accuracy wobble; results are trustworthy |
| **Output quality** | Temperature scaling (`0.5`) | Cleaner text; no invented words |

The result is both a modeling upgrade (better context handling via attention) and a systems upgrade (better speed and stability on GPU) — while keeping the code compact enough to read in one sitting.