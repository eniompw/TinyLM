# TinyLM

TinyLM is a small character-level language modeling playground.

> Follows on from [MLP-Digits-Classifier](https://github.com/eniompw/MLP-Digits-Classifier), extending multi-class neural networks from image classification to character-level language modeling.
>
> This series continues with [MicroGPT](https://github.com/eniompw/MicroGPT), where the character-level model is scaled up to a decoder-only transformer.

It contains seven compact implementations that train character-level models and generate text autoregressively:

- a NumPy single-layer perceptron baseline for names generation
- a CuPy character MLP with learned embeddings
- a PyTorch character MLP equivalent to `TinyMLP.py`
- a simplified PyTorch transformer encoder as a bridge between MLP and full transformer
- a compact PyTorch transformer encoder language model
- an OOP refactor of the transformer using `nn.Module`
- a modern Llama-style transformer with RoPE, RMSNorm, SiLU, and `torch.compile`

The code is intentionally short so you can read end-to-end training and sampling in one sitting.

## Contents

- [Repository contents](#repository-contents)
- [Models](#models)
- [Requirements](#requirements)
- [Benchmarks](#benchmarks)
- [Dataset](#dataset)
- [Suggested next improvements](#suggested-next-improvements)

## Repository contents

| File | Description |
|------|-------------|
| [README.md](README.md) | Project overview, model summaries, requirements, and roadmap |
| [LICENSE](LICENSE) | Repository license |
| [BENCHMARKS.md](BENCHMARKS.md) | Training snapshots and generated sample comparisons |
| [TODO.md](TODO.md) | Task list and planned follow-ups |
| [NameSLP.py](NameSLP.py) | NumPy single-layer perceptron trained on character windows from the names dataset |
| [names_dataset.py](names_dataset.py) | Karpathy names data loader with character encoding and one-hot context features |
| [TinyMLP.py](TinyMLP.py) | CuPy character MLP with a learned embedding table and one hidden layer |
| [TinyMLP-explained.md](TinyMLP-explained.md) | Walkthrough of `TinyMLP.py` including data flow, tensor shapes, and manual gradient steps |
| [tinystories_dataset.py](tinystories_dataset.py) | Shared TinyStories data loader and character-level preprocessing utility |
| [TorchMLP.py](TorchMLP.py) | PyTorch equivalent of `TinyMLP.py` using autograd and the same core architecture |
| [TorchMLP.ipynb](TorchMLP.ipynb) | Notebook version of `TorchMLP.py` |
| [SimpleTransformer.py](SimpleTransformer.py) | Simplified PyTorch transformer — same structure as `TorchMLP.py` but with token + positional embeddings and a 2-layer transformer encoder |
| [SimpleTransformer.ipynb](SimpleTransformer.ipynb) | Notebook version of `SimpleTransformer.py` |
| [SimpleTransformer-explained.md](SimpleTransformer-explained.md) | Walkthrough of `SimpleTransformer.py` including architecture choices and the bridge from MLP to full transformer |
| [TinyTransformer.py](TinyTransformer.py) | PyTorch character-level transformer encoder with token + positional embeddings, mixed precision, and autoregressive sampling |
| [TinyTransformer-explained.md](TinyTransformer-explained.md) | Walkthrough of `TinyTransformer.py` including architecture choices, training flow, and speed/quality optimization notes |
| [TinyTransformerClass.py](TinyTransformerClass.py) | OOP refactor of `TinyTransformer.py` wrapping the model in an `nn.Module` class with a `get_batch()` helper function |
| [TinyLlama.py](TinyLlama.py) | Modern Llama-style transformer with RoPE, RMSNorm, SiLU, fused AdamW, mixed precision, and `torch.compile` |

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

### 4) `SimpleTransformer.py` (simplified PyTorch transformer)

- Uses context windows of length 8 (`context_size`).
- Loads TinyStories data via `load_tinystories(...)` from `tinystories_dataset.py`.
- Uses `num_stories=200` to choose how many streamed TinyStories records to train on.
- Architecture: token embedding + positional embedding -> 2-layer transformer encoder -> output projection.
- Trains with plain Adam and a fixed learning rate — no scheduler, no mixed precision.
- Uses full-dataset eval (OOM-safe at 200 stories); swap to 4096-subset eval for larger datasets.
- Designed as a teaching bridge between `TorchMLP.py` and `TinyTransformer.py`.
- Generates text autoregressively with temperature-controlled sampling.

### 5) `TinyTransformer.py` (PyTorch character transformer)

- Uses context windows of length 8 (`context_size` / block size).
- Loads TinyStories data via `load_tinystories(...)` from `tinystories_dataset.py`.
- Uses `num_stories=1000` to choose how many streamed TinyStories records to train on.
- Architecture: token embedding + positional embedding -> 2-layer transformer encoder -> output projection.
- Trains with AdamW, gradient scaling, autocast mixed precision, gradient clipping, and cosine learning-rate decay.
- Uses automatic device selection (`cuda` when available, otherwise `cpu`) via `torch.set_default_device(...)`.
- Generates text autoregressively with temperature-controlled sampling.

### 6) `TinyTransformerClass.py` (OOP refactor of `TinyTransformer.py`)

- Same architecture and training setup as `TinyTransformer.py`.
- Wraps `tok_embed`, `pos_embed`, `transformer`, and `linear` in a `TinyTransformer(nn.Module)` class.
- Exposes a `forward(x)` method, enabling standard PyTorch patterns such as `model.parameters()`.
- Extracts batch sampling into a standalone `get_batch()` function.
- Keeps all original comments, section headers, and training/generation logic unchanged.

### 7) `TinyLlama.py` (modern Llama-style transformer)

- Uses context windows of length 256 (`context_size`).
- Loads TinyStories data via `load_tinystories(...)` from `tinystories_dataset.py`.
- Uses `num_stories=1000` to choose how many streamed TinyStories records to train on.
- Architecture: token embedding -> 2-layer `ModernBlock` (RoPE, RMSNorm, SiLU MLP) -> weight-tied lm_head.
- Replaces sinusoidal positional embeddings with Rotary Position Embeddings (RoPE).
- Replaces LayerNorm with RMSNorm and GELU with SiLU activation.
- Trains with fused AdamW, `GradScaler`, autocast float16, gradient clipping, and cosine LR decay.
- Compiled with `torch.compile` for ~2x GPU kernel speedup.
- Prints device info, param count, and elapsed time at each eval step.
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
- `SimpleTransformer.py` auto-selects `cuda` when available; runs on CPU but is slower without GPU.
- `TinyTransformer.py` is optimized for CUDA (`torch.compile`, AMP, fused AdamW) and is best run with a modern PyTorch + GPU setup.
- `TinyTransformerClass.py` shares the same hardware requirements as `TinyTransformer.py`.
- `TinyLlama.py` shares the same hardware requirements as `TinyTransformerClass.py`.

## Benchmarks

Training accuracy snapshots and generated sample comparisons are tracked in [BENCHMARKS.md](BENCHMARKS.md).

## Dataset

- Source: `names.txt` from `karpathy/makemore` (downloaded directly from GitHub).
- Source: `karpathy/tinystories-gpt4-clean` via Hugging Face Datasets (streaming mode).
- Helper loaders: `names_dataset.py` (for `NameSLP.py`) and `tinystories_dataset.py` (for `TinyMLP.py`, `TorchMLP.py`, `SimpleTransformer.py`, `TinyTransformer.py`, `TinyTransformerClass.py`, and `TinyLlama.py`).
- Tokenization is character-level, keeping the project simple and educational.

## Suggested next improvements

- Add CPU/NumPy fallback to `TinyMLP.py` for non-CUDA environments.
- Expose hyperparameters through command-line arguments.
- Add checkpoint save/load support.
- Add deterministic seeding and simple evaluation metrics.
