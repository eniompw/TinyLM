# TorchMLP Explained

This file walks through [TorchMLP.py](TorchMLP.py) from top to bottom. The script is a small character-level language model built with PyTorch, trained on TinyStories text, and used to generate new text one character at a time.

## Contents

1. [What the script does](#what-the-script-does)
2. [Imports and setup](#imports-and-setup)
3. [Hyperparameters](#hyperparameters)
4. [Data preparation](#data-preparation)
5. [Model architecture](#model-architecture)
6. [Training loop](#training-loop)
7. [Generation](#generation)
8. [Embedding dimension experiment summary](#embedding-dimension-edim-experiment-summary)
9. [Context size and hidden size experiment summary](#context-size-ctx-and-hidden-size-h-experiment-summary)
10. [Why this model is useful](#why-this-model-is-useful)
11. [Things to notice](#things-to-notice)

## What the script does

At a high level, the script:

1. Loads TinyStories through a shared data loader (`load_tinystories`) in `tinystories_dataset.py`.
2. Builds a character vocabulary from the first 200 stories.
3. Converts the text into fixed-size context windows of 4 characters.
4. Trains a tiny MLP to predict the next character from that context.
5. Samples from the trained model to generate text autoregressively.

## Imports and setup

The script imports `load_tinystories` from `tinystories_dataset.py` for data preparation and `time` for timing. It uses `torch.nn` for model components and `torch.nn.functional` for the cross-entropy loss.

```python
torch.set_default_device('cuda' if torch.cuda.is_available() else 'cpu')
```

This single line ensures all tensors are created on the GPU when available, removing per-tensor `.to(device)` boilerplate throughout the script.

## Hyperparameters

All tunable values are collected at the top of the script for easy experimentation:

| Name | Value | Description |
|---|---|---|
| `num_stories` | 200 | Stories loaded from TinyStories |
| `context_size` | 4 | Previous tokens used to predict next |
| `embed_dim` | 256 | Token embedding dimension |
| `hidden_dim` | 150 | MLP hidden layer dimension |
| `batch_size` | 1024 | Samples per training step |
| `lr` | 0.5 | SGD learning rate |
| `n_steps` | 2001 | Total training steps |
| `temp` | 1.0 | Sampling temperature during generation |

## Data preparation

Data preparation is delegated to the shared loader:

```python
input_ids, target_ids, idx_to_char, token_ids = load_tinystories(num_stories=num_stories, context_size=context_size)
input_ids, target_ids = torch.tensor(input_ids), torch.tensor(target_ids)
```

`load_tinystories(...)` streams TinyStories from Hugging Face, keeps the first `num_stories` records, builds the character vocabulary, and encodes the text as sliding context windows. The loader returns Python lists, which are converted to PyTorch tensors before training.

### Context windows

For every position in the text, the input is the previous `context_size` characters and the target is the next character:

```python
encoded = [char_to_id[c] for c in text]
inputs  = [encoded[i:i+context_size] for i in range(len(encoded)-context_size)]
targets = encoded[context_size:]
```

So the model learns: given 4 characters, what character comes next?

## Model architecture

The model is a small feed-forward network with a learned character embedding table and one hidden ReLU layer.

### Embedding table

```python
embedding = nn.Embedding(len(idx_to_char), embed_dim)
```

Each character ID maps to a learned vector of size `embed_dim = 256`. For a 4-character context, the 4 embedding vectors are concatenated into one flat feature vector of size `context_size * embed_dim`.

### MLP layers

```python
mlp = nn.Sequential(
    nn.Linear(context_size * embed_dim, hidden_dim), nn.ReLU(),
    nn.Linear(hidden_dim, len(idx_to_char))
)
```

- The first linear layer projects the concatenated embeddings (`4 × 256 = 1024`) into a hidden layer of size `hidden_dim = 150`, followed by ReLU.
- The second linear layer maps the hidden activations to logits over the full vocabulary.

PyTorch's `nn.Linear` includes bias terms by default, and `nn.Sequential` handles the forward pass automatically — no manual forward/backward code required.

### Optimizer

```python
params    = list(embedding.parameters()) + list(mlp.parameters())
optimizer = torch.optim.SGD(params, lr=lr)
```

SGD with `lr=0.5` is used. The parameter list is built explicitly so it mirrors the style of larger models in this repo where multiple submodules are combined.

## Training loop

The script trains for `n_steps = 2001` steps using mini-batches of size 1024.

### Batch sampling and forward pass

```python
batch_idx = torch.randint(0, len(input_ids), (batch_size,))
x = embedding(input_ids[batch_idx]).view(batch_size, -1)
loss = F.cross_entropy(mlp(x), target_ids[batch_idx])
```

1. Random batch indices are sampled each step.
2. Embeddings for the context window are looked up and flattened with `.view(batch_size, -1)`.
3. `F.cross_entropy` fuses softmax and negative log-likelihood into a single stable operation.

### Backward pass

```python
optimizer.zero_grad(); loss.backward(); optimizer.step()
```

PyTorch autograd handles all gradient computation. The three operations — clear old gradients, backpropagate, update weights — are kept on one line to emphasise the standard SGD idiom.

### Evaluation during training

Every 200 steps, the full dataset is run through the model to measure accuracy:

```python
with torch.no_grad():
    pred_ids = mlp(embedding(input_ids).view(len(input_ids), -1)).argmax(1)
    print(f"Step {step:4d} | Loss: {loss:.4f} | Acc: {(pred_ids == target_ids).float().mean():.1%} | {time.time()-start:.1f}s")
```

`torch.no_grad()` disables gradient tracking during evaluation, saving memory and compute.

## Generation

After training, the script defines `generate(num_chars=200, context_ids=..., temp=temp)`.

```python
@torch.no_grad()
def generate(num_chars=200, context_ids=list(token_ids[:context_size]), temp=temp):
    output_chars = [idx_to_char[i] for i in context_ids]
    for _ in range(num_chars):
        x = embedding(torch.tensor([context_ids])).view(1, -1)
        next_token_probs = torch.softmax(mlp(x) / temp, 1)
        next_token = torch.multinomial(next_token_probs, 1).item()
        context_ids, output_chars = context_ids[1:] + [next_token], output_chars + [idx_to_char[next_token]]
    return ''.join(output_chars)
```

It starts from the first `context_size` characters of the training data as seed context, then repeatedly:

1. Embeds the current context window and flattens it.
2. Divides logits by `temp` before softmax — lower values make the distribution sharper (more confident), higher values make it more uniform (more random).
3. Samples the next character with `torch.multinomial`.
4. Slides the context window forward by one character.

This is autoregressive generation: each new character is predicted from the previously generated characters.

## Embedding dimension (`edim`) experiment summary

| Embedding dimension (`edim`) | Final accuracy (step 2000) | Training time |
|---|---:|---:|
| 10 | 52.3% | 3.1s |
| 32 | 55.7% | 3.2s |
| 64 | 57.3% | 3.4s |
| 128 | 58.8% | 3.2s |
| 256 | 59.3% | 3.3s |
| 512 | 59.9% | 4.2s |
| 1024 | 60.2% | 6.4s |

### Key findings

- **Sweet spot (128–256):** `edim=256` improves over `edim=128` (59.3% vs 58.8%) at only +0.1s cost.
- **Diminishing returns (512–1024):** 256→512 costs +0.9s for +0.6%; 512→1024 costs +2.2s for +0.3%.
- **Why time rises:** Larger `edim` makes the `emb_cat @ W1` multiply heavier; memory bandwidth becomes the bottleneck.

## Context size (`ctx`) and hidden size (`h`) experiment summary

| Configuration | Final accuracy | Train time | Acc/s (efficiency) |
|---|---:|---:|---:|
| `ctx=2, h=150` | 41.1% | 23.2s | 1.77 |
| `ctx=4, h=100` | 46.8% | 20.3s | 2.31 |
| `ctx=4, h=150` | 48.3% | 24.3s | 1.99 |
| `ctx=4, h=200` | 47.9% | 27.2s | 1.76 |
| `ctx=4, h=300` | **49.3%** | 35.3s | 1.40 |
| `ctx=10, h=150` | 48.6% | 29.2s | 1.66 |
| `ctx=15, h=200` | 48.2% | 40.2s | 1.20 |
| **`ctx=4, h=150 + mini-batch`** | **52.4%** | **4.4s** | **11.91** |

### Takeaways

- **Best accuracy:** `ctx=4, h=300` at **49.3%** (full-dataset training).
- **Best efficiency:** `ctx=4, h=150 + mini-batch` at **52.4%**, **4.4s**, **11.91 Acc/s** — about 6× more efficient than the best full-dataset run.
- **Context effect:** `ctx=2` is too small; gains beyond `ctx=4` are limited for an MLP.
- **Hidden-size effect:** improvements become inconsistent past `h=150` and cost more time.

## Why this model is useful

The script is intentionally minimal, but it demonstrates several core ideas:

1. Character-level language modeling with learned embeddings.
2. Using `nn.Sequential` to define a model without writing a `forward()` method.
3. Mini-batch SGD with PyTorch autograd — no manual gradient code.
4. Temperature-controlled sampling from a probabilistic model.
5. A direct comparison point for the Transformer models in this repo.

## Things to notice

PyTorch's `nn.Linear` includes bias terms by default, unlike the original CuPy version which had no biases. This is a small but meaningful architectural difference.

`torch.set_default_device` means no `.to(device)` calls appear anywhere in the script — all tensors land on the GPU automatically when available.

Because it is character-level, the model learns very local patterns. It can produce plausible-looking text fragments but lacks the capacity of the Transformer models in this repo.