# 🤖 TinyLM: Build Your Own Mini AI

TinyLM is a hands-on playground for building and training small language models from scratch. Models start by predicting the *next letter* in a sentence, and progress all the way to predicting *subword tokens* using BPE tokenization — the same technique used by GPT.

The code is intentionally kept short so you can read the entire training and text-generation process in one sitting.

> 🧠 **Prerequisite:** Check out [MLP-Digits-Classifier](https://github.com/eniompw/MLP-Digits-Classifier) first to see how neural networks classify images. TinyLM takes those same concepts and applies them to generating text!
>
> 🚀 **What's Next?** This series continues with [MicroGPT](https://github.com/eniompw/MicroGPT), where we scale up to a full, modern decoder-only transformer.

---

## 📚 The Evolution of Our Models

We didn't start with a complex AI — we built up to it, upgrading one thing at a time. Each model is an **experiment**: we change one major thing and measure whether it makes the AI smarter or faster. See [BENCHMARKS.md](BENCHMARKS.md) for the full results and a guide to experiments vs. ablations.

### Level 1: The Basics (Predicting Letters)

- **[NameSLP.py](NameSLP.py) — The NumPy Baseline:** A minimal model in raw NumPy. It looks at 6 letters and guesses the next one to generate fake names (like "Emma" or "Oliver").
- **[TinyMLP.py](TinyMLP.py) — The CuPy MLP:** Upgrades to a Multi-Layer Perceptron (MLP) using CuPy for GPU speed, and adds learned embeddings — a lookup table that helps the model understand letters.
- **[TorchMLP.py](TorchMLP.py) — The PyTorch MLP:** The same model as above, rewritten in PyTorch. This unlocks autograd (automatic gradient calculation) so we no longer do the calculus by hand.

### Level 2: Enter the Transformer (The "Brain" Upgrade)

- **[SimpleTransformer.py](SimpleTransformer.py) — The Bridge:** Adds a 3-layer Transformer Encoder on top of the PyTorch MLP. This gives the model *attention* — the ability to read the whole context window, not just the last few letters.
- **[TinyTransformer.py](TinyTransformer.py) — The Workhorse:** Our baseline for all experiments in BENCHMARKS.md. Adds mixed precision (float16) and a cosine learning rate schedule for faster, more stable training.
- **[TinyTransformerClass.py](TinyTransformerClass.py) — The Cleanup:** Identical to the above, reorganized using Object-Oriented Programming (OOP) to match standard professional PyTorch style.

### Level 3: Modern AI (Llama Architecture & Subword Tokenization)

- **[TinyLlama.py](TinyLlama.py) — The Modern Era:** Rebuilds the model using the same architectural tricks as Meta's Llama. Swaps in **RoPE** (smarter positional encoding), **RMSNorm** (better layer normalization), and **SiLU** (a smoother activation function). Uses `torch.compile` for a significant GPU speedup.
- **[TinyBPE.py](TinyBPE.py) — The Tokenizer Upgrade:** Replaces character-level tokenization with a custom **BPE (Byte-Pair Encoding)** tokenizer trained directly on TinyStories. A context window of 32 BPE tokens now covers ~20 words instead of ~5 characters — breaking the information ceiling that all character-level models share. Same architecture as `TinyTransformer.py`, smarter data.

---

## 📁 Repository Contents

| File | Description |
| :--- | :--- |
| [BENCHMARKS.md](BENCHMARKS.md) | The lab notebook — all experiments, ablations, and accuracy results. |
| [TODO.md](TODO.md) | Planned follow-ups and known gaps. |
| [LICENSE](LICENSE) | Repository license. |
| **Datasets** | |
| [names_dataset.py](names_dataset.py) | Loads the names dataset used by `NameSLP.py`. |
| [tinystories_dataset.py](tinystories_dataset.py) | Streams TinyStories for all models. Provides `load_tinystories` (character-level) and `load_tinystories_bpe` (subword BPE). |
| [TinyBPE.py](TinyBPE.py) | Trains and applies a custom BPE tokenizer on TinyStories, then trains a transformer over subword tokens. |
| **Explained Guides** | |
| [TinyMLP-explained.md](TinyMLP-explained.md) | Walkthrough of `TinyMLP.py`: data flow, tensor shapes, and gradients. |
| [TorchMLP-Explained.md](TorchMLP-Explained.md) | Walkthrough of `TorchMLP.py`: the PyTorch rewrite and autograd. |
| [SimpleTransformer-explained.md](SimpleTransformer-explained.md) | Walkthrough of the MLP → Transformer transition. |
| [TinyTransformer-explained.md](TinyTransformer-explained.md) | Walkthrough of architecture choices and speed/quality optimizations. |

---

## 💻 Requirements & Setup

This project is designed to run on **Google Colab** using a free T4 GPU.

**You will need:**
- Python 3.10+
- An internet connection (to download the datasets on the first run)
- Google Colab (or a local PC with an NVIDIA GPU)

**Python Packages:**
- `numpy` — math
- `cupy` — GPU math for the early models
- `datasets` — Hugging Face library to stream TinyStories
- `tokenizers` — Hugging Face library to train and run the BPE tokenizer (`TinyBPE.py`)
- `torch` — PyTorch, for building and training the neural networks

*Note: [TorchMLP.py](TorchMLP.py) and [SimpleTransformer.py](SimpleTransformer.py) can run on CPU, but will be significantly slower.*

---

## 📖 The Datasets

- **`names.txt`** — A list of real names from Andrej Karpathy's `makemore` project. Used to teach the AI how to spell names.
- **TinyStories** — Simple, AI-generated children's stories from Hugging Face. Character-level models process this letter by letter; `TinyBPE.py` uses a custom BPE tokenizer trained on the same corpus.

---

## 💡 Ideas for Contributors

Want to keep tinkering? Here are some open experiments worth trying:

- Add a CPU/NumPy fallback to [TinyMLP.py](TinyMLP.py) so it runs without a GPU.
- Add command-line arguments to change hyperparameters without editing the source code.
- Add a Save/Load checkpoint feature so you don't have to retrain from scratch each run.
- Add deterministic seeding throughout so results are exactly reproducible.
