# TinyLM

TinyLM is a minimal character-level language model built with PyTorch. It trains a small MLP on TinyStories text and then samples new characters autoregressively.

The project is intentionally compact so you can understand the full training and generation flow in one script.

## What is in this repo

- `TorchLinear.py`: end-to-end training and sampling script.
- `TorchLinear.ipynb`: notebook version for interactive experimentation.

## Model summary

The script trains a character model with:

- learned token embeddings
- learned positional embeddings
- a 2-layer MLP head

Given a fixed context window of characters, the model predicts the next character.

## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA (current script uses `.cuda()` directly)
- Internet connection for first dataset download

Python packages:

- `torch`
- `datasets`

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch datasets
```

## Run

```bash
python TorchLinear.py
```

During training, the script prints loss every 500 steps, then prints a generated sample at the end.

## Notes

- Dataset source: `karpathy/tinystories-gpt4-clean` (streamed via Hugging Face Datasets).
- Training data in the script is limited to the first 5000 streamed stories.
- Tokenization is character-level, so output quality is simple but useful for learning.

## Next improvements

- Add CPU fallback (`device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`).
- Move hyperparameters to command-line args.
- Save and reload model checkpoints.