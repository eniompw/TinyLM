# TinyMLP Explained

This file walks through [TinyMLP.py](TinyMLP.py) from top to bottom. The script is a small character-level language model built with CuPy, trained on TinyStories text, and used to generate new text one character at a time.

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
char_to_idx = {c: i for i, c in enumerate(vocab)}
idx_to_char = {i: c for i, c in enumerate(vocab)}
```

This makes the task purely character-level. Every unique character gets an integer ID.

### Context windows

The model uses a context size of 4:

```python
context_size = 4
```

For every position in the text, the input is the previous 4 characters and the target is the next character. So the model learns the question: given 4 characters, what character comes next?

The code then:

1. Encodes the whole text as integer IDs.
2. Builds `inputs` as sliding windows of length 4.
3. Builds `targets` as the next character after each window.
4. Converts targets into one-hot vectors for cross-entropy training.

## Model architecture

The model is a small feed-forward network with learned character embeddings.

### Embedding table

```python
C = init_randn(vocab_size, emb_dim)
```

Each character ID maps to a learned vector of size `emb_dim = 10`. For a 4-character context, the 4 embedding vectors are concatenated into one long feature vector.

### Hidden layer

```python
W1 = init_randn(context_size * emb_dim, hidden_size)
b1 = cp.zeros((1, hidden_size), dtype=cp.float32)
```

The concatenated embeddings are projected into a hidden layer of size `hidden_size = 150`, then passed through ReLU.

### Output layer

```python
W2 = init_randn(hidden_size, vocab_size)
b2 = cp.zeros((1, vocab_size), dtype=cp.float32)
```

The hidden activations are mapped to a logit for every character in the vocabulary. After softmax, those logits become probabilities for the next character.

## Training loop

The script trains for 2001 epochs using mini-batches of size 1024.

### Batch sampling

Each iteration samples random training positions:

```python
idx = cp.random.randint(0, N, size=batch_size)
X_batch = inputs[idx]
Y_batch = one_hot_targets[idx]
```

This is stochastic gradient descent with random mini-batches.

Summary note: no mini-batching reached 53.0% at epoch 2000 in 48.9s, while mini-batching reached 52.3% in 3.1s. That is only a 0.7% absolute accuracy gap, but mini-batching is about 15.8x faster, so it offers nearly the same quality for dramatically lower wall-clock time.

If you are converting a full-dataset loop to mini-batching, the key training-loop edit is:

```python
for epoch in range(2001):
	# --- Mini-batching ---
	idx = cp.random.randint(0, N, size=batch_size)              # random sample indices
	X_batch = inputs[idx]                                       # slice inputs
	Y_batch = one_hot_targets[idx]                              # slice targets
```

Then, inside that loop:

1. Replace `inputs` with `X_batch`.
2. Replace `one_hot_targets` with `Y_batch`.
3. Replace `N` with `batch_size` for batch-shaped operations (for example reshape sizes and gradient normalization).

### Forward pass

```python
emb_cat = C[X_batch].reshape(batch_size, -1)
h = cp.maximum(0, emb_cat @ W1 + b1)
logits = h @ W2 + b2
probs = softmax(logits)
```

What happens here:

1. Look up embeddings for the 4 input characters.
2. Flatten them into one vector per example.
3. Apply a linear layer plus ReLU.
4. Produce logits over the vocabulary.
5. Convert logits into probabilities.

### Backward pass

The gradient for cross-entropy with softmax is simplified as:

```python
dlogits = (probs - Y_batch) / batch_size
```

From there, gradients are computed for each parameter by the chain rule:

1. `W2` and `b2` from the output layer.
2. Backpropagation into the hidden layer.
3. ReLU mask to zero out gradients where activations were negative.
4. `W1` and `b1` from the hidden layer.
5. Gradients for the embedding table `C`.

### Embedding gradient accumulation

The embedding table is updated with:

```python
cp.add.at(dC, X_batch.ravel(), demb.reshape(-1, emb_dim))
```

This matters because the same character can appear multiple times in a batch. `add.at` accumulates repeated gradient contributions correctly.

### Parameter update

Each parameter is updated with simple SGD:

```python
param -= lr * grad
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

## TinyMLP experiment summary

- Notation: **(4/150)** means **context size = 4** characters and **hidden size = 150** neurons.
- No mini-batching (4/150): **53.0%** at epoch 2000, **48.9s** total training time.
- Mini-batching (4/150): **52.3%** at epoch 2000, **3.1s** total training time.
- Final accuracy gap: no mini-batching is +0.7% absolute (53.0% vs 52.3%).
- Speed gap: mini-batching is about **15.8x faster** (48.9s vs 3.1s).
- Practical trade-off: mini-batching gives nearly the same final quality for dramatically lower wall-clock time.
- Epoch checkpoints (no mini-batching): 0: 17.2%, 200: 34.8%, 400: 39.3%, 600: 43.3%, 800: 46.3%, 1000: 48.3%, 1200: 49.4%, 1400: 50.4%, 1600: 51.2%, 1800: 52.1%, 2000: 53.0%.
- Epoch checkpoints (mini-batching): 0: 17.0%, 200: 34.9%, 400: 38.2%, 600: 43.0%, 800: 46.3%, 1000: 48.4%, 1200: 48.8%, 1400: 50.7%, 1600: 51.7%, 1800: 50.5%, 2000: 52.3%.
- Takeaway: in this setup, mini-batching is the better default because it preserves almost all accuracy while reducing training time by about an order of magnitude.

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

## Short summary

TinyMLP is a compact character MLP that learns to predict the next character from a 4-character window. It uses embeddings, one hidden ReLU layer, and a softmax output, then samples from the learned distribution to generate text.