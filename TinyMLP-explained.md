# TinyMLP Explained

This file walks through [TinyMLP.py](TinyMLP.py) from top to bottom. The script is a small character-level language model built with CuPy, trained on TinyStories text, and used to generate new text one character at a time.

## Contents

1. [What the script does](#what-the-script-does)
2. [Imports and setup](#imports-and-setup)
3. [Data preparation](#data-preparation)
4. [Model architecture](#model-architecture)
5. [Training loop](#training-loop)
6. [Generation](#generation)
7. [TinyMLP experiment summary](#tinymlp-experiment-summary)
8. [Embedding dimension experiment summary](#embedding-dimension-edim-experiment-summary)
9. [Context size and hidden size experiment summary](#context-size-ctx-and-hidden-size-h-experiment-summary-pre-mini-batch)
10. [Why this model is useful](#why-this-model-is-useful)
11. [Things to notice](#things-to-notice)
12. [Short summary](#short-summary)

## What the script does

At a high level, the script:

1. Loads a stream of TinyStories examples from Hugging Face Datasets.
2. Builds a character vocabulary from the first 200 stories.
3. Converts the text into fixed-size context windows of 4 characters.
4. Trains a tiny MLP to predict the next character from that context.
5. Samples from the trained model to generate text autoregressively.

## Imports and setup

The script imports `load_dataset` for data access, `itertools` for slicing the stream, `time` for timing, and `warnings` to silence warnings. It uses `cupy` as `cp` for GPU arrays and `numpy` as `np` for sampling.

The `softmax` helper converts raw logits into probabilities. It subtracts the row-wise max before exponentiating, which improves numerical stability.

## Data preparation

The dataset is loaded in streaming mode:

```python
dataset = load_dataset('karpathy/tinystories-gpt4-clean', split='train', streaming=True)
text = ''.join(s['text'] for s in itertools.islice(dataset, 200))
```

Only the first 200 stories are used. This keeps the experiment small and fast.

From that text, the script builds a character vocabulary:

```python
vocab = sorted(set(text))
vocab_size = len(vocab)
char_to_id = {c: i for i, c in enumerate(vocab)}
```

This makes the task purely character-level. Every unique character gets an integer ID.

### Context windows

The model uses a context size of 4:

```python
context_size = 4
```

The code then:

1. Encodes the whole text as integer IDs.
2. Builds `inputs` as sliding windows of length 4.
3. Builds `targets` as the next character after each window.
4. Uses the target IDs directly for cross-entropy training.

For every position in the text, the input is the previous 4 characters and the target is the next character. The script encodes the text as integer IDs and then builds:

```python
encoded = [char_to_id[c] for c in text]
inputs = cp.array([encoded[i:i+context_size] for i in range(len(encoded)-context_size)])
targets = cp.array(encoded[context_size:])
```

So the model learns the question: given 4 characters, what character comes next?

## Model architecture

The model is a small feed-forward network with learned character embeddings.

### Embedding table

```python
C = randn(vocab_size, emb_dim)
```

Each character ID maps to a learned vector of size `emb_dim = 256`. For a 4-character context, the 4 embedding vectors are concatenated into one long feature vector.

### Hidden layer

```python
W1 = randn(context_size * emb_dim, hidden_size)
```

The concatenated embeddings are projected into a hidden layer of size `hidden_size = 150`, then passed through ReLU. There are no bias terms in this version of the model.

### Output layer

```python
W2 = randn(hidden_size, vocab_size)
```

The hidden activations are mapped to a logit for every character in the vocabulary. After softmax, those logits become probabilities for the next character.

## Training loop

The script trains for 2001 epochs using mini-batches of size 1024.

### Batch sampling

Each iteration samples random training positions:

```python
idx = cp.random.randint(0, N, size=batch_size)
X, Y = inputs[idx], targets[idx]
```

This is stochastic gradient descent with random mini-batches.

If you are converting a full-dataset loop to mini-batching, the key training-loop edit is:

```python
for epoch in range(2001):
	# --- Mini-batching ---
	idx = cp.random.randint(0, N, size=batch_size)  # random sample indices
	X_batch = inputs[idx]                           # slice inputs
	Y_batch = targets[idx]                          # slice targets
```

Then, inside that loop:

1. Replace `inputs` with `X_batch`.
2. Replace `targets` with `Y_batch`.
3. Replace `N` with `batch_size` for batch-shaped operations, such as reshape sizes and gradient normalization.

### Forward pass

```python
emb = C[X].reshape(batch_size, -1)
h = cp.maximum(0, emb @ W1)
probs = softmax(h @ W2)
```

What happens here:

1. Look up embeddings for the 4 input characters.
2. Flatten them into one vector per example.
3. Apply a linear layer plus ReLU.
4. Produce logits over the vocabulary.
5. Convert logits into probabilities.

### Backward pass

The gradient for cross-entropy with softmax is simplified by directly subtracting 1 at the target index:

```python
probs[cp.arange(batch_size), Y] -= 1
probs /= batch_size
```

From there, gradients are computed for each parameter by the chain rule:

1. `W2` from the output layer.
2. Backpropagation into the hidden layer.
3. ReLU mask to zero out gradients where activations were negative.
4. `W1` from the hidden layer.
5. Gradients for the embedding table `C`.

### Embedding gradient accumulation

The embedding table is updated with:

```python
cp.add.at(dC, X.ravel(), (dh @ W1.T).reshape(-1, emb_dim))
```

This matters because the same character can appear multiple times in a batch. `add.at` accumulates repeated gradient contributions correctly.

### Parameter update

Each parameter is updated with simple SGD:

```python
p -= lr * g
```

The learning rate is `0.5`, which is fairly large, but the network is tiny and the experiment is deliberately lightweight.

### Evaluation during training

Every 200 epochs, the script computes full-dataset accuracy by running the whole training set through the model and comparing predicted next characters to the true targets.

## Generation

After training, the script defines `generate(num_chars=200)`.

It starts from the first 4 characters of the training data as a seed context, then repeatedly:

1. Runs the current context through the model.
2. Converts the output logits into probabilities.
3. Samples the next character from that distribution.
4. Slides the context window forward by one character.

This is autoregressive generation: each new character is predicted from the previously generated characters.

## TinyMLP experiment summary

- Baseline config is **ctx=4, h=150**.
- Full-dataset training reached **53.0%** in **48.9s**.
- Mini-batch training reached **52.3%** in **3.1s**.
- Mini-batching is the practical default here: only **0.7%** lower accuracy for about **15.8x** faster training.

## Embedding dimension (`edim`) experiment summary

| Embedding dimension (`edim`) | Final accuracy (epoch 2000) | Training time |
|---|---:|---:|
| 10 | 52.3% | 3.1s |
| 32 | 55.7% | 3.2s |
| 64 | 57.3% | 3.4s |
| 128 | 58.8% | 3.2s |
| 256 | 59.3% | 3.3s |
| 512 | 59.9% | 4.2s |
| 1024 | 60.2% | 6.4s |

### Key findings

- **Sweet spot (128-256):** `edim=256` looks best overall: it improves accuracy over `edim=128` (59.3% vs 58.8%) at only about +0.1s training time (3.3s vs 3.2s).
- **Diminishing returns (512-1024):** 256 -> 512 costs +0.9s for +0.6% accuracy. 512 -> 1024 costs +2.2s (about +52% time) for +0.3%.
- **Why time rises:** Larger `edim` makes `emb_cat @ W1` much heavier, so matrix multiply and memory bandwidth become the bottleneck.

## Context size (`ctx`) and hidden size (`h`) experiment summary (pre-mini-batch)

These runs were measured before mini-batching was introduced.

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

- **Best pre-mini-batch accuracy:** `ctx=4, h=300` at **49.3%**.
- **Best pre-mini-batch efficiency:** `ctx=4, h=150` is a better speed/quality balance than larger hidden sizes.
- **Context effect:** `ctx=2` is too small; gains beyond `ctx=4` are limited for this MLP.
- **Hidden-size effect at `ctx=4`:** improvements are inconsistent past `h=150` and cost more time.
- **Overall winner:** `ctx=4, h=150 + mini-batch` at **52.4%**, **4.4s**, **11.91 Acc/s** (about 6x more efficient than the best full-dataset run).

## Why this model is useful

The script is intentionally minimal, but it demonstrates several core ideas:

1. Character-level language modeling.
2. Learned embeddings.
3. A manual forward/backward pass.
4. Mini-batch SGD.
5. Sampling from a probabilistic model.

It is a good reference if you want to understand how a language model works without the complexity of a Transformer.

## Things to notice

The implementation is fully manual. There is no deep learning framework autograd involved, so the gradients are explicit and easy to inspect.

The code assumes a CuPy-compatible GPU setup. On a machine without CUDA, it will not run as written.

Because it is character-level, the model learns very local patterns. It can produce plausible-looking text fragments, but it does not have the capacity of larger sequence models.

## Short summary

TinyMLP is a compact character MLP that learns to predict the next character from a 4-character window. It uses embeddings, one hidden ReLU layer, and a softmax output, then samples from the learned distribution to generate text.