# TinyLM

TinyLM is a small character-level language modeling playground.

Conceptually, it sits about halfway between [MLP-Digits-Classifier](https://github.com/eniompw/MLP-Digits-Classifier) and [MicroGPT](https://github.com/eniompw/MicroGPT).

It contains two compact implementations that train on TinyStories text and generate characters autoregressively:

- a CuPy character MLP with learned embeddings
- a PyTorch MLP model with token + positional embeddings

The code is intentionally short so you can read end-to-end training and sampling in one sitting.

## Repository contents

- `TinyMLP.py`: CuPy character MLP with a learned embedding table and one hidden layer.
- `TinyMLP.ipynb`: notebook version of the CuPy model.
- `TinyMLP explained.md`: a short walkthrough of `TinyMLP.py`, including data flow, tensor shapes, and manual gradient steps.
- `TorchLinear.py`: PyTorch character model with learned token/position embeddings and a 2-layer MLP head.
- `TorchLinear.ipynb`: notebook version for interactive experimentation.

## Models

### 1) `TinyMLP.py` (CuPy character MLP)

- Uses context windows of length 4 (`context_size`).
- Learns character embeddings, a ReLU hidden layer, and an output projection.
- Streams 200 TinyStories samples.
- Trains with manual forward/backward passes in CuPy using `float32` weights and mini-batches.
- Uses vectorized embedding-gradient accumulation instead of a Python loop.
- Generates text autoregressively from the trained model.

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
- NVIDIA T4 GPU (Google Colab GPU runtime assumed)

Python packages:

- `numpy`
- `cupy`
- `datasets`
- `torch`

Hardware notes:

- `TinyMLP.py` uses CuPy, so it expects a compatible CUDA setup.
- `TorchLinear.py` assumes CUDA is available and will run on GPU.

## Colab (T4) quick start

1. In Colab, open Runtime -> Change runtime type -> Hardware accelerator -> `T4 GPU`.
2. Install dependencies:

```bash
pip install --upgrade pip
pip install datasets torch
pip install cupy-cuda12x
```

3. Clone and run:

```bash
git clone https://github.com/eniompw/TinyLM.git
cd TinyLM
python TinyMLP.py
python TorchLinear.py
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install numpy datasets torch
# Install a CuPy build that matches your CUDA version, for example:
pip install cupy-cuda12x
```

## Run

Run the CuPy MLP:

```bash
python TinyMLP.py
```

Run the PyTorch model:

```bash
python TorchLinear.py
```

## Dataset

- Source: `karpathy/tinystories-gpt4-clean` via Hugging Face Datasets (streaming mode).
- Tokenization is character-level, keeping the project simple and educational.

## Suggested next improvements

- Add CPU/NumPy fallback to `TinyMLP.py` for non-CUDA environments.
- Add device fallback (`cuda`/`cpu`) to `TorchLinear.py`.
- Expose hyperparameters through command-line arguments.
- Add checkpoint save/load support.
- Add deterministic seeding and simple evaluation metrics.