# TinyLM

TinyLM is a small character-level language modeling playground.

Conceptually, it sits about halfway between [MLP-Digits-Classifier](https://github.com/eniompw/MLP-Digits-Classifier) and [MicroGPT](https://github.com/eniompw/MicroGPT).

It contains three compact implementations that train character-level models and generate text autoregressively:

- a NumPy single-layer perceptron baseline for names generation
- a CuPy character MLP with learned embeddings
- a PyTorch character MLP equivalent to `TinyMLP.py`

The code is intentionally short so you can read end-to-end training and sampling in one sitting.

## Contents

- [Repository contents](#repository-contents)
- [Models](#models)
- [Requirements](#requirements)
- [Colab (T4) quick start](#colab-t4-quick-start)
- [Setup](#setup)
- [Run](#run)
- [Dataset](#dataset)
- [Suggested next improvements](#suggested-next-improvements)

## Repository contents

- `nameSLP.py`: NumPy single-layer perceptron trained on character windows from the names dataset.
- `names_dataset.py`: Karpathy names data loader with character encoding and one-hot context features.
- `TinyMLP.py`: CuPy character MLP with a learned embedding table and one hidden layer.
- `tinystories_dataset.py`: shared TinyStories data loader and character-level preprocessing utility.
- `TinyMLP.ipynb`: notebook version of the CuPy model.
- `TinyMLP-explained.md`: a short walkthrough of `TinyMLP.py`, including data flow, tensor shapes, and manual gradient steps.
- `TorchMLP.py`: PyTorch equivalent of `TinyMLP.py` using autograd and the same core architecture.

## Models

### 1) `nameSLP.py` (NumPy SLP baseline)

- Uses context windows of length 6 (`context_size`).
- Loads names data via `load_names(...)` from `names_dataset.py`.
- Trains a single linear softmax classifier with gradient descent.
- Uses one-hot flattened context features from the names dataset.
- Prints periodic training accuracy and samples generated character sequences.

### 2) `TinyMLP.py` (CuPy character MLP)

- Uses context windows of length 4 (`context_size`).
- Loads TinyStories data via `load_tinystories(...)` from `tinystories_dataset.py`.
- Learns character embeddings, a ReLU hidden layer, and an output projection.
- Streams 200 TinyStories samples.
- Trains with manual forward/backward passes in CuPy using `float32` weights and mini-batches.
- Uses vectorized embedding-gradient accumulation instead of a Python loop.
- Generates text autoregressively from the trained model.

### 3) `TorchMLP.py` (PyTorch equivalent of `TinyMLP.py`)

- Uses context windows of length 4 (`context_size`).
- Loads TinyStories data via `load_tinystories(...)` from `tinystories_dataset.py`.
- Uses the same embedding -> ReLU hidden -> output projection architecture as `TinyMLP.py`.
- Trains with PyTorch autograd and SGD-style parameter updates.
- Uses automatic device selection (`cuda` when available, otherwise `cpu`).
- Generates text autoregressively from the trained model.

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
- `TorchMLP.py` auto-selects `cuda` when available and otherwise runs on `cpu`.

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
python nameSLP.py
python TinyMLP.py
python TorchMLP.py
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

Run the NumPy SLP baseline:

```bash
python nameSLP.py
```

Run the CuPy MLP:

```bash
python TinyMLP.py
```

Run the PyTorch model:

```bash
python TorchMLP.py
```

## Dataset

- Source: `names.txt` from `karpathy/makemore` (downloaded directly from GitHub).
- Source: `karpathy/tinystories-gpt4-clean` via Hugging Face Datasets (streaming mode).
- Helper loaders: `names_dataset.py` (for `nameSLP.py`) and `tinystories_dataset.py` (for `TinyMLP.py` and `TorchMLP.py`).
- Tokenization is character-level, keeping the project simple and educational.

## Suggested next improvements

- Add CPU/NumPy fallback to `TinyMLP.py` for non-CUDA environments.
- Expose hyperparameters through command-line arguments.
- Add checkpoint save/load support.
- Add deterministic seeding and simple evaluation metrics.