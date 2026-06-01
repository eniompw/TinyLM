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
6. [Accuracy and quality optimizations](#accuracy-and-quality-optimizations)
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

- Increases context window from 4 to 8 (`context_size=8`) so the model can use more recent characters.
- Increases dataset slice from 200 stories to 1000 stories for a richer training signal.
- Replaces the MLP block with a transformer encoder (`2` layers, `4` heads).
- Adds positional embeddings (`pos_embed`) so token order is represented explicitly.
- Uses modern PyTorch optimization features for GPU throughput and stability.

## Model architecture

### Token + position embeddings

```python
tok_embed = nn.Embedding(len(vocab), 256)
pos_embed = nn.Embedding(8, 256)
```

- `tok_embed` maps each character ID to a 256-dim vector.
- `pos_embed` gives each position in the 8-token window its own learned vector.
- The input to the transformer is their sum:

```python
input_embed = tok_embed(batch_inputs) + pos_embed(torch.arange(8))
```

This lets the model know both what token it sees and where it appears in the context.

### Transformer encoder

```python
transformer = torch.compile(
    nn.TransformerEncoder(
        nn.TransformerEncoderLayer(
            256, 4, 1024,
            batch_first=True,
            dropout=0.,
            norm_first=True
        ),
        2
    )
)
```

- `d_model=256`
- `nhead=4`
- feed-forward width `1024`
- `2` stacked encoder layers
- `batch_first=True` keeps tensor shape as `(B, T, C)`
- `norm_first=True` is often more stable for transformer training

After the encoder, only the last time step is used for next-token prediction:

```python
logits = model(transformer(input_embed)[:, -1, :])
```

This mirrors the autoregressive setup: use the whole context, predict the next character.

## Training loop and optimization stack

The training loop is still simple and compact:

1. Sample random mini-batch indices.
2. Run forward pass with mixed precision autocast.
3. Compute cross-entropy loss.
4. Backprop with gradient scaling.
5. Unscale, clip gradients, optimizer step, scheduler step.
6. Print periodic evaluation accuracy.

The key difference vs `TorchMLP.py` is that many performance and stability techniques are combined in one loop.

## Speed optimizations

The speed-oriented changes in this section are adapted from experiments and notes in the MicroGPT repository:
https://github.com/eniompw/MicroGPT

### 1) `torch.compile`

`torch.compile(...)` traces and fuses model operations into optimized kernels.
This reduces Python overhead and kernel launch fragmentation, often giving large speedups on GPU-heavy code.

### 2) `float16` autocast + `GradScaler`

```python
with torch.autocast('cuda', dtype=torch.float16):
    ...
scaler = torch.amp.GradScaler('cuda')
```

- `float16` reduces memory traffic and accelerates matrix math on modern GPUs.
- `GradScaler` prevents tiny gradients from underflowing to zero during backprop.

This gives most of the AMP speed benefit while keeping training numerically safe.

### 3) Fused AdamW

```python
optimizer = torch.optim.AdamW(params, lr=1e-3, fused=True)
```

With `fused=True`, parameter updates are grouped into optimized CUDA kernels instead of many small launches.
This reduces update overhead, especially when many tensors are involved.

### 4) `zero_grad(set_to_none=True)`

```python
optimizer.zero_grad(set_to_none=True)
```

Setting grads to `None` avoids writing zeros into every gradient buffer each step.
That cuts memory bandwidth work during gradient reset.

## Accuracy and quality optimizations

### 1) Cosine LR schedule with floor

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, 2000, eta_min=1e-4
)
```

- Starts with larger steps to move quickly early in training.
- Gradually decays learning rate to `1e-4` for finer late-stage convergence.

### 2) Gradient clipping

```python
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(params, 1.0)
```

Clipping at `1.0` prevents rare extreme batches from causing unstable jumps.
This is especially helpful in transformer training where occasional spikes can derail progress.

### 3) AdamW momentum tuning (`betas=(0.9, 0.95)`)

The condensed notes include `betas=(0.9, 0.95)` as an additional quality tweak.
In this file, optimizer betas are currently left at PyTorch defaults. If you want exact parity with the note, use:

```python
optimizer = torch.optim.AdamW(
    params,
    lr=1e-3,
    betas=(0.9, 0.95),
    fused=True
)
```

Lower second beta than default can make the optimizer react faster to recent gradient patterns in transformer workloads.

### 4) Inference temperature (`0.7`)

```python
probabilities = torch.softmax(
    model(transformer(input_embed)[:, -1, :]) / 0.7,
    1
)
```

Dividing logits by `0.7` sharpens the output distribution:

- high-confidence tokens become more likely
- low-confidence tokens become less likely

This usually produces cleaner text with less gibberish, without retraining.

## Generation behavior

Generation logic is still close to `TorchMLP.py`:

1. Start from the first real context window from the dataset.
2. Predict next-token distribution.
3. Sample one token.
4. Slide context by one and repeat.

Main differences are:

- context length is 8 instead of 4
- transformer encoder replaces MLP forward pass
- temperature scaling (`/ 0.7`) improves output readability

## Practical scaling notes

Recent experiments highlight practical limits for this compact setup:

- Using `5000` TinyStories can sometimes crash CUDA (typically memory pressure).
- Increasing `context_size` to `64` makes training much slower.
- Accuracy gain versus `context_size=8` is modest relative to the runtime cost in these runs.

Observed run (`context_size=8`):

- best accuracy: `67.4%` at epoch `1600`
- final accuracy: `67.0%` at epoch `2000`
- training time: `25.4s`

Observed run (`context_size=64`):

- best accuracy: `68.5%` at epoch `1800`
- final accuracy: `68.4%` at epoch `2000`
- training time: `197.5s`

Direct comparison:

- best accuracy delta (`64 - 8`): `+1.1` points
- training time ratio (`64 / 8`): about `7.8x` slower

## Why this version is stronger than TorchMLP

`TorchMLP.py` is ideal for understanding baseline language-model training mechanics.
`TinyTransformer.py` keeps that readability but introduces practical modern training techniques:

- attention-based sequence modeling
- mixed precision training
- compiled graph execution
- fused optimizer kernels
- learning-rate scheduling and gradient clipping

So it is both a modeling upgrade (better context handling) and a systems upgrade (better speed/stability on GPU).

## Short summary

`TinyTransformer.py` is the direct next step after `TorchMLP.py`:

- same task (next-character prediction)
- stronger architecture (transformer + positional embeddings)
- faster training stack (`torch.compile`, AMP, fused AdamW, `set_to_none=True`)
- safer convergence stack (cosine LR floor, gradient clipping)
- cleaner generation via temperature scaling (`0.7`)

Together these changes make training more efficient and output quality more consistent while keeping the code compact and educational.
