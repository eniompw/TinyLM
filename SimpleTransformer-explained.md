# SimpleTransformer Explained

This file walks through `SimpleTransformer.py` and explains how it builds on `TorchMLP.py`.

`TorchMLP.py` is the baseline: a character-level MLP that predicts the next character from a fixed context window.
`SimpleTransformer.py` keeps the same training loop style, dataset pipeline, and autoregressive generation, then swaps the MLP block for a transformer encoder with positional embeddings — and nothing else changes.

It is designed as the minimal step between `TorchMLP.py` and `TinyTransformer.py`.

## Contents

1. [What stays the same from TorchMLP](#what-stays-the-same-from-torchmlp)
2. [What changes in SimpleTransformer](#what-changes-in-simpletransformer)
3. [Model architecture](#model-architecture)
4. [Training loop](#training-loop)
5. [Generation behavior](#generation-behavior)
6. [How SimpleTransformer differs from TinyTransformer](#how-simpletransformer-differs-from-tinytransformer)
7. [Short summary](#short-summary)

## What stays the same from TorchMLP

Both scripts share the same high-level workflow:

1. Load TinyStories using `load_tinystories(...)` from `tinystories_dataset.py`.
2. Convert text into character IDs and context windows.
3. Train a next-character predictor using cross-entropy.
4. Generate text autoregressively by repeatedly predicting one token at a time.

The training loop structure is also identical — random batch sampling, forward pass, cross-entropy loss, `zero_grad / backward / step` — and the evaluation and generation sections follow the same pattern.

## What changes in SimpleTransformer

Compared with `TorchMLP.py`, `SimpleTransformer.py` makes these structural changes:

- Increases context window from 4 to 8 (`context_size=8`) so the model can use more recent characters.
- Replaces the MLP block with a transformer encoder (`2` layers, `4` heads).
- Adds positional embeddings (`pos_embed`) so token order is represented explicitly.
- Adds a `# --- Hyperparameters ---` block at the top, matching `TinyTransformer.py` style.
- Keeps all other code — optimizer, eval, generate — as close to `TorchMLP.py` as possible.

## Model architecture

### Token + position embeddings

```python
tok_embed = nn.Embedding(len(idx_to_char), embed_dim)
pos_embed = nn.Embedding(context_size, embed_dim)
```

- `tok_embed` maps each character ID to a 128-dim vector.
- `pos_embed` gives each of the 8 positions in the context window its own learned vector.
- The input to the transformer is their sum:

```python
x = tok_embed(input_ids[batch_idx]) + pos_embed(torch.arange(context_size))
```

This is the key difference from `TorchMLP.py`, which concatenates embeddings into a flat vector with `.view()`. The transformer needs a sequence as input, not a flat vector, so embeddings are summed instead.

### Transformer encoder

```python
transformer = torch.compile(
    nn.TransformerEncoder(
        nn.TransformerEncoderLayer(embed_dim, n_heads, ffn_dim, batch_first=True, dropout=0.),
        n_layers
    )
)
```

- `embed_dim=128`, `n_heads=4`, `ffn_dim=256`, `n_layers=2`

**`batch_first=True`** — `nn.TransformerEncoderLayer` defaults to `batch_first=False`,
expecting `(seq_len, batch, embed_dim)`. Our embeddings come out as
`(batch_size, context_size, embed_dim)`, so this flag matches PyTorch's expectations
to our natural data layout instead of requiring manual transposes. It can't be
replaced by reordering arguments — it governs how tensor *axes* are interpreted,
not call syntax. Skipping it wouldn't crash training (since `batch_size=1024` and
`context_size=8` differ), it would silently scramble batch and sequence dimensions.

**`dropout=0.`** — the layer's real default is `0.1`, not `0`, so this is
an active override. Omitting it would quietly reintroduce 10% dropout, making runs
non-deterministic and breaking parity with the dropout-free `TorchMLP.py` baseline.

After the encoder, only the last time step is used for next-token prediction:

```python
loss = F.cross_entropy(linear(transformer(x)[:, -1, :]), target_ids[batch_idx])
```

`[:, -1, :]` extracts the final position's hidden state — representing the model's summary of the full context — and passes it to the linear output layer.

### Output projection

```python
linear = nn.Linear(embed_dim, len(idx_to_char))
```

Same role as in `TorchMLP.py`: maps the hidden state to logits over the vocabulary. `F.cross_entropy` applies softmax internally.

## Training loop

The training loop is identical in structure to `TorchMLP.py`:

```python
for step in range(n_steps):
    batch_idx = torch.randint(0, len(input_ids), (batch_size,))
    x = tok_embed(input_ids[batch_idx]) + pos_embed(torch.arange(context_size))
    loss = F.cross_entropy(linear(transformer(x)[:, -1, :]), target_ids[batch_idx])
    optimizer.zero_grad(); loss.backward(); optimizer.step()
```

The only difference from `TorchMLP.py` is the forward pass line — the embedding and model call — everything else is unchanged.

Evaluation uses the full dataset (safe at `num_stories=200`):

```python
x_eval = tok_embed(input_ids) + pos_embed(torch.arange(context_size))
pred_ids = linear(transformer(x_eval)[:, -1, :]).argmax(1)
```

> **Note:** If increasing `num_stories` beyond ~200, switch to a 4096-sample subset to avoid GPU OOM:
> ```python
> eval_idx = torch.randint(0, len(input_ids), (4096,))
> x_eval = tok_embed(input_ids[eval_idx]) + pos_embed(torch.arange(context_size))
> pred_ids = linear(transformer(x_eval)[:, -1, :]).argmax(1)
> # use target_ids[eval_idx] in accuracy calculation
> ```

## Generation behavior

Generation logic is the same as `TorchMLP.py` with two differences:

1. Context length is 8 instead of 4.
2. The forward pass uses transformer + positional embeddings instead of MLP + `.view()`.

```python
x = tok_embed(torch.tensor([context_ids])) + pos_embed(torch.arange(context_size))
next_token_probs = torch.softmax(linear(transformer(x)[:, -1, :]) / 0.7, 1)
next_token = torch.multinomial(next_token_probs, 1).item()
```

Temperature `0.7` sharpens the output distribution so high-confidence tokens are sampled more often, producing cleaner text without retraining.

## How SimpleTransformer differs from TinyTransformer

`SimpleTransformer.py` intentionally removes several features present in `TinyTransformer.py` to keep the code minimal:

| Feature | TinyTransformer.py | SimpleTransformer.py |
|---|---|---|
| Optimizer | AdamW (`betas=(0.9, 0.95)`, fused) | Adam (defaults) |
| LR scheduler | CosineAnnealingLR | None (fixed `lr=1e-3`) |
| Mixed precision | `torch.autocast` float16 | None |
| `embed_dim` | 256 | 128 |
| `ffn_dim` | 1024 | 256 |
| `num_stories` | 1000 | 200 |
| Eval strategy | 4096-sample subset | Full dataset |
| Final accuracy | ~67.7% | ~67.2% |
| Training time | ~20.5s | ~35.6s |

The accuracy gap is small (~0.5%). The speed difference comes from full-dataset eval rather than the model itself.

## Short summary

`SimpleTransformer.py` is the minimal step from `TorchMLP.py` to a transformer:

- same task (next-character prediction)
- same training loop structure
- same dataset and generation logic
- new: transformer encoder replaces MLP block
- new: positional embeddings added alongside token embeddings
- new: hyperparameters extracted to a named block at the top

No scheduler, no mixed precision, no advanced optimizer settings — just the transformer itself.
