# TinyLM

TinyLM is a small character-level language modeling playground.

It contains two compact implementations that train on TinyStories text and generate characters autoregressively:

- a NumPy single-layer bigram model
- a PyTorch MLP model with token + positional embeddings

The code is intentionally short so you can read end-to-end training and sampling in one sitting.

## Repository contents

- `TinySLP.py`: NumPy baseline (single linear layer over one-hot inputs).
- `TorchLinear.py`: PyTorch character model with learned token/position embeddings and a 2-layer MLP head.
- `TorchLinear.ipynb`: notebook version for interactive experimentation.

## Models

### 1) `TinySLP.py` (NumPy bigram model)

- Builds one-hot bigram pairs: current character -> next character.
- Trains a linear classifier with softmax and gradient descent.
- Streams 100 TinyStories samples.
- Runs on CPU.

### 2) `TorchLinear.py` (PyTorch MLP)

- Uses context windows of length 16 (`block_size`).
- Learns:
	- token embeddings
	- positional embeddings
	- 2-layer MLP head for next-character prediction
- Streams 5000 TinyStories samples.
- Trains for 3000 steps and reports loss every 500 steps.
- Generates text by sampling from the output distribution.

## Requirements

- Python 3.10+
- Internet connection for first dataset download

Python packages:

- `numpy`
- `datasets`
- `torch`

Hardware notes:

- `TinySLP.py` is CPU-friendly.
- `TorchLinear.py` currently calls `.cuda()` directly, so it requires a CUDA-capable GPU as written.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install numpy datasets torch
```

## Run

Run the NumPy baseline:

```bash
python TinySLP.py
```

Run the PyTorch model:

```bash
python TorchLinear.py
```

## Dataset

- Source: `karpathy/tinystories-gpt4-clean` via Hugging Face Datasets (streaming mode).
- Tokenization is character-level, keeping the project simple and educational.

## Suggested next improvements

- Add device fallback (`cuda`/`cpu`) to `TorchLinear.py`.
- Expose hyperparameters through command-line arguments.
- Add checkpoint save/load support.
- Add deterministic seeding and simple evaluation metrics.