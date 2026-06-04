# TinyLM

TinyLM is a small character-level language modeling playground.

> Follows on from [MLP-Digits-Classifier](https://github.com/eniompw/MLP-Digits-Classifier), extending multi-class neural networks from image classification to character-level language modeling.
>
> This series continues with [MicroGPT](https://github.com/eniompw/MicroGPT), where the character-level model is scaled up to a decoder-only transformer.

It contains four compact implementations that train character-level models and generate text autoregressively:

- a NumPy single-layer perceptron baseline for names generation
- a CuPy character MLP with learned embeddings
- a PyTorch character MLP equivalent to `TinyMLP.py`
- a compact PyTorch transformer encoder language model

The code is intentionally short so you can read end-to-end training and sampling in one sitting.

## Contents

- [Repository contents](#repository-contents)
- [Models](#models)
- [Requirements](#requirements)
- [Accuracy tracker](#accuracy-tracker)
- [Dataset](#dataset)
- [Suggested next improvements](#suggested-next-improvements)

## Repository contents

| File | Description |
|------|-------------|
| `NameSLP.py` | NumPy single-layer perceptron trained on character windows from the names dataset |
| `names_dataset.py` | Karpathy names data loader with character encoding and one-hot context features |
| `TinyMLP.py` | CuPy character MLP with a learned embedding table and one hidden layer |
| `tinystories_dataset.py` | Shared TinyStories data loader and character-level preprocessing utility |
| `TinyMLP.ipynb` | Notebook version of the CuPy model |
| `TinyMLP-explained.md` | Walkthrough of `TinyMLP.py` including data flow, tensor shapes, and manual gradient steps |
| `TorchMLP.py` | PyTorch equivalent of `TinyMLP.py` using autograd and the same core architecture |
| `TinyTransformer.py` | PyTorch character-level transformer encoder with token + positional embeddings, mixed precision, and autoregressive sampling |
| `TinyTransformer-explained.md` | Walkthrough of `TinyTransformer.py` including architecture choices, training flow, and speed/quality optimization notes |

## Models

### 1) `NameSLP.py` (NumPy SLP baseline)

- Uses context windows of length 6 (`context_size`).
- Loads names data via `load_names(...)` from `names_dataset.py`.
- Trains a single linear softmax classifier with gradient descent.
- Uses one-hot flattened context features from the names dataset.
- Prints periodic training accuracy and samples generated character sequences.

### 2) `TinyMLP.py` (CuPy character MLP)

- Uses context windows of length 4 (`context_size`).
- Loads TinyStories data via `load_tinystories(...)` from `tinystories_dataset.py`.
- Uses `num_stories=200` to choose how many streamed TinyStories records to train on.
- Learns character embeddings, a ReLU hidden layer, and an output projection.
- Streams 200 TinyStories samples.
- Trains with manual forward/backward passes in CuPy using `float32` weights and mini-batches.
- Uses vectorized embedding-gradient accumulation instead of a Python loop.
- Generates text autoregressively from the trained model.

### 3) `TorchMLP.py` (PyTorch equivalent of `TinyMLP.py`)

- Uses context windows of length 4 (`context_size`).
- Loads TinyStories data via `load_tinystories(...)` from `tinystories_dataset.py`.
- Uses `num_stories=200` to choose how many streamed TinyStories records to train on.
- Uses the same embedding -> ReLU hidden -> output projection architecture as `TinyMLP.py`.
- Trains with PyTorch autograd and SGD-style parameter updates.
- Uses automatic device selection (`cuda` when available, otherwise `cpu`).
- Generates text autoregressively from the trained model.

### 4) `TinyTransformer.py` (PyTorch character transformer)

- Uses context windows of length 8 (`context_size` / block size).
- Loads TinyStories data via `load_tinystories(...)` from `tinystories_dataset.py`.
- Uses `num_stories=1000` to choose how many streamed TinyStories records to train on.
- Architecture: token embedding + positional embedding -> 2-layer transformer encoder -> output projection.
- Trains with AdamW, gradient scaling, autocast mixed precision, gradient clipping, and cosine learning-rate decay.
- Uses automatic device selection (`cuda` when available, otherwise `cpu`) via `torch.set_default_device(...)`.
- Generates text autoregressively with temperature-controlled sampling.

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
- `TinyTransformer.py` is optimized for CUDA (`torch.compile`, AMP, fused AdamW) and is best run with a modern PyTorch + GPU setup.

## Accuracy tracker

Training accuracy snapshots and generated sample comparisons are tracked in [model_accuracy_tracker.md](model_accuracy_tracker.md).

## Dataset

- Source: `names.txt` from `karpathy/makemore` (downloaded directly from GitHub).
- Source: `karpathy/tinystories-gpt4-clean` via Hugging Face Datasets (streaming mode).
- Helper loaders: `names_dataset.py` (for `NameSLP.py`) and `tinystories_dataset.py` (for `TinyMLP.py`, `TorchMLP.py`, and `TinyTransformer.py`).
- Tokenization is character-level, keeping the project simple and educational.

## Suggested next improvements

- Add CPU/NumPy fallback to `TinyMLP.py` for non-CUDA environments.
- Expose hyperparameters through command-line arguments.
- Add checkpoint save/load support.
- Add deterministic seeding and simple evaluation metrics.
