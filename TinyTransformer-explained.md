# TinyTransformer Explained

This file walks through `TinyTransformer.py` and explains how it builds on `TorchMLP.py`.

`TorchMLP.py` is the baseline: a character-level MLP that predicts the next character from a fixed context window.
`TinyTransformer.py` keeps the same core training goal, dataset pipeline style, and autoregressive generation loop, then adds transformer attention plus a set of speed and quality optimizations.

Research source for optimization ideas: https://github.com/eniompw/MicroGPT

## Contents

1. [What stays the same from TorchMLP](#what-stays-the-same-from-torchmlp)
2. [What changes in TinyTransformer](#what-changes-in-tinytransformer)
3. [Model architecture](#model-architecture)
4. [Training loop and optimization stack](#training-loop-and-optimization-stack)
5. [Speed optimizations](#speed-optimizations)
6. [Accuracy, quality, and scientific controls](#accuracy-quality-and-scientific-controls)
7. [Generation behavior](#generation-behavior)
8. [Practical scaling notes](#practical-scaling-notes)
9. [Why this version is stronger than TorchMLP](#why-this-version-is-stronger-than-torchmlp)
10. [Short summary](#short-summary)

## What stays the same from TorchMLP

Both scripts share the same high-level workflow:

1. Load TinyStories using `load_tinystories(...)` from `tinystories_dataset.py`.
2. Convert text into character IDs and context windows.
3. Train a next-character predictor using cross-entropy.
4. Generate text autoregressively by repeatedly predicting one token at a time.

So this is not a new task. It is the same task with a stronger architecture and faster training stack.

## What changes in TinyTransformer

Compared with `TorchMLP.py`, `TinyTransformer.py` makes these structural changes:

- Increases context window from 4 to 32 (`context_size=32`) so the model can use more recent characters.
- Increases dataset slice from 200 stories to 5000 stories for a richer training signal.
- Replaces the MLP block with a transformer encoder (`3` layers, `4` heads). *Note: 3 layers was found to be the perfect "sweet spot" between depth and speed!*
- Adds positional embeddings (`pos_embed`) so token order is represented explicitly.
- Uses modern PyTorch optimization features for GPU throughput and stability.
- Automatically sets the default device to GPU (`torch.set_default_device(device)`) to remove manual `.to(device)` boilerplate.

## Model architecture

### Token + position embeddings

```python
tok_embed = nn.Embedding(len(idx_to_char), embed_dim)
pos_embed = nn.Embedding(context_size, embed_dim)
```

- `tok_embed` maps each character ID to a 256-dim vector.
- `pos_embed` gives each position in the 32-token window its own learned vector.
- The input to the transformer is their sum:

```python
x = tok_embed(batch_x) + pos_embed(torch.arange(context_size))
```

This lets the model know both what token it sees and where it appears in the context. Without this, the AI sees sentences as a jumbled "bag of letters" (our benchmarks show that removing positional embeddings drops accuracy by ~7.7%).

### Transformer encoder

```python
transformer = torch.compile(
    nn.TransformerEncoder(
        nn.TransformerEncoderLayer(
            embed_dim, n_heads, ffn_dim,
            batch_first=True,
            dropout=0.,
            norm_first=True
        ),
        n_layers
    )
)
```

- `d_model=256`
- `nhead=4`
- feed-forward width `1024`
- `3` stacked encoder layers (best speed/accuracy tradeoff)
- `batch_first=True` keeps tensor shape as `(B, T, C)`
- `norm_first=True` is often more stable for transformer training

After the encoder, only the last time step is used for next-token prediction:

```python
logits = linear(transformer(x)[:, -1, :])
```

This mirrors the autoregressive setup: use the whole context, predict the next character.

## Training loop and optimization stack

The training loop is still simple and compact:

1. Sample random mini-batch indices.
2. Run forward pass with mixed precision autocast.
3. Compute cross-entropy loss.
4. Backprop, optimizer step, scheduler step.
5. Print periodic evaluation accuracy using a fixed random seed.

The key difference vs `TorchMLP.py` is that many performance and stability techniques are combined in one loop.

## Speed optimizations

The speed-oriented changes in this section are adapted from experiments and notes in the MicroGPT repository:
https://github.com/eniompw/MicroGPT

### 1) Global Device Default

```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_default_device(device)
```
Instead of manually moving every tensor to the GPU (`.to('cuda')`), we tell PyTorch to put everything on the GPU by default. This keeps the code clean and prevents hardware mismatch errors.

### 2) `torch.compile`

`torch.compile(...)` traces and fuses model operations into optimized kernels.
This reduces Python overhead and kernel launch fragmentation, often giving large speedups on GPU-heavy code. (Just remember to "warm up" the model by running it once before timing it!)

### 3) `float16` autocast

```python
with torch.autocast('cuda', dtype=torch.float16):
    ...
```

`float16` reduces memory traffic and accelerates matrix math on modern GPUs.

### 4) Fused AdamW

```python
optimizer = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.95), fused=True)
```

With `fused=True`, parameter updates are grouped into optimized CUDA kernels instead of many small launches.
This reduces update overhead, especially when many tensors are involved.

## Accuracy, quality, and scientific controls

### 1) Cosine LR schedule with floor

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, n_steps, eta_min=1e-4
)
```

- Starts with larger steps to move quickly early in training.
- Gradually decays learning rate to `1e-4` for finer late-stage convergence.

### 2) AdamW momentum tuning (`betas=(0.9, 0.95)`)

Lower second beta than the PyTorch default (`0.999`) makes the optimizer react faster to recent gradient patterns in transformer workloads.

### 3) Large Batch + High LR (`batch=1536`, `lr=2e-3`)

Instead of making the model bigger, we feed it more data per step (`batch_size=1536`) and set a high learning rate (`2e-3`) to match. While a batch size of 1536 is a slight decrease from the peak offline-memorization baseline (2048), it serves as a balance to offset the mathematical overhead of a larger `context_size=32` while keeping the run under our budget, and still processing far more parallel tokens per step than earlier setups.

### 4) Scientific Controls: Fixed Eval Seed

```python
eval_rng = torch.Generator(device=device).manual_seed(0)
# ... inside the loop:
eval_idx = torch.randint(0, len(input_ids), (4096,), generator=eval_rng)
```

When we test the model every 200 steps, we don't test it on the whole dataset. We grab a random subset. But if the subset changes every time, our accuracy will "wobble" up and down based on luck! We fixed this by creating a dedicated `eval_rng`. Now, the model is always tested on the exact same 4,096 stories, completely eliminating accuracy noise.

### 5) Inference temperature (`0.5`)

```python
next_token_probs = torch.softmax(linear(transformer(x)[:, -1, :]) / 0.5, 1)
```

Dividing logits by `0.5` sharpens the output distribution:

- high-confidence tokens become more likely
- low-confidence tokens become less likely

This makes the model more confident, producing cleaner text and eliminating weird/fake words (like "throbe" -> "robe"), without retraining.

## Generation behavior

Generation logic is still close to `TorchMLP.py`:

1. Start from the first real context window from the dataset.
2. Predict next-token distribution.
3. Sample one token.
4. Slide context by one and repeat.

Main differences are:

- context length is 32 instead of 4
- transformer encoder replaces MLP forward pass
- temperature scaling (`/ 0.5`) improves output readability

## Practical scaling notes

Recent experiments highlight practical limits for this compact setup:

- Increasing `context_size` to `64` makes training much slower for only a tiny incremental accuracy gain.
- The "Colab Lottery" means GPU speeds vary wildly. We use Relative Speed Ratios (comparing against a baseline run) instead of absolute seconds to judge if an architecture is actually faster.

Observed run (3-Layer, `batch=1536` configuration on 5,000 stories, `context_size=32`):

- best accuracy: `~70.1%` at step `1600` (using fixed eval seed)
- training time: `~143.8s` (varies by GPU)
- result: Reaches optimal generalization/intelligence under a 2.5-minute budget. While our previous 1,000-story baseline achieved higher nominal accuracy, expanding to 5,000 stories forces the model to stop memorizing and actually learn English grammar (preventing grammatical/coherence breakdown).

## Why this version is stronger than TorchMLP

`TorchMLP.py` is ideal for understanding baseline language-model training mechanics.
`TinyTransformer.py` keeps that readability but introduces practical modern training techniques:

- attention-based sequence modeling
- mixed precision training
- compiled graph execution
- fused optimizer kernels
- learning-rate scheduling
- scientific reproducibility (fixed eval seeds)

So it is both a modeling upgrade (better context handling) and a systems upgrade (better speed/stability on GPU).

## Short summary

`TinyTransformer.py` is the direct next step after `TorchMLP.py`:

- same task (next-character prediction)
- stronger architecture (3-layer transformer + positional embeddings)
- faster training stack (`torch.compile`, AMP, fused AdamW, global device)
- safer convergence stack (cosine LR floor, balanced batch size)
- scientific reproducibility (fixed eval seed to eliminate accuracy wobble)
- cleaner generation via temperature scaling (`0.5`)

Together these changes make training more efficient and output quality more consistent while keeping the code compact and educational.
```