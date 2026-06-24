# 🤖 TinyLM: Build Your Own Mini AI

TinyLM is a hands-on playground for building and training small, character-level language models from scratch. Instead of guessing whole words like ChatGPT, these models learn to guess the *next letter* in a sentence. 

The code is intentionally kept short so you can read the entire training and text-generation process in one sitting.

> 🧠 **Prerequisite:** Check out [MLP-Digits-Classifier](https://github.com/eniompw/MLP-Digits-Classifier) first to see how neural networks classify images. TinyLM takes those same concepts and applies them to generating text!
>
> 🚀 **What's Next?** This series continues with [MicroGPT](https://github.com/eniompw/MicroGPT), where we scale up to a full, modern decoder-only transformer.

---

## 🔬 The Science of AI: Experiments vs. Ablations

In real AI research, you don't just write code and hope it works—you use the **Scientific Method**. As you build these models, you will be doing two types of scientific tests. Think of the AI model like a recipe or a custom PC build:

*   **🧪 Experiments (Upgrading):** Trying a *new feature* or *upgrading a setting* to see if it makes the model better. 
    *   *Example:* "What if we add more layers to the model's brain?" or "What if we double the memory?"
*   **✂️ Ablations (Removing):** Taking an existing feature *away* to prove that it's actually necessary. If the model gets worse, you've proven that feature matters!
    *   *Example:* "What if we remove the model's ability to know the order of words?" (Spoiler: It turns into word soup). 

Head over to **[BENCHMARKS.md](BENCHMARKS.md)** after reading this to see the actual results of our experiments and ablations!

---

## 📚 The Evolution of Our Models (Leveling Up)

We didn't start with a complex AI. We built up to it, upgrading the model step-by-step. Each model here is an **experiment**—we change one major thing to see if it makes the AI smarter or faster.

### Level 1: The Basics (Predicting Letters)
*   **1. `NameSLP.py` (The NumPy Baseline):** A super simple model written in raw NumPy. It looks at 6 letters and guesses the next one to make up fake names (like "Emma" or "Oliver"). 
*   **2. `TinyMLP.py` (The CuPy MLP):** We upgrade to a Multi-Layer Perceptron (MLP) using CuPy (for GPU speed). We also give it "learned embeddings"—a cheat sheet that helps it understand letters.
*   **3. `TorchMLP.py` (The PyTorch MLP):** The exact same model as #2, but rewritten in PyTorch. This lets us use PyTorch's "autograd" (automatic gradient calculator) so we don't have to do the hard calculus by hand!

### Level 2: Enter the Transformer (The "Brain" Upgrade)
*   **4. `SimpleTransformer.py` (The Bridge):** We take our PyTorch MLP and add a 2-layer Transformer Encoder. This gives the AI "attention"—the ability to look at the context of a whole sentence, not just the letter right before it.
*   **5. `TinyTransformer.py` (The Workhorse):** This is our baseline model for all the experiments in the `BENCHMARKS.md` file! We add fancy speed tricks like mixed precision (using smaller numbers to calculate faster) and cosine learning rates.
*   **6. `TinyTransformerClass.py` (The Cleanup):** The exact same model as #5, but we organize the code using Object-Oriented Programming (OOP). This makes the code look like standard, professional PyTorch.

### Level 3: Modern AI (Llama Architecture)
*   **7. `TinyLlama.py` (The Modern Era):** We rebuild the model using the exact same architecture tricks used in Meta's Llama models. We swap old math for new math: **RoPE** (a smarter way to understand word order), **RMSNorm** (better stabilization), and **SiLU** (a better activation function). We also use `torch.compile` to double the GPU speed.

---

## 📁 Repository Contents

| File | What it does |
| :--- | :--- |
| `README.md` | You are here! Project overview and guide. |
| `BENCHMARKS.md` | **The Lab Notebook!** Track our experiments, ablations, and accuracy results. |
| `TODO.md` | Task list and planned follow-ups. |
| `LICENSE` | Repository license. |
| **Datasets & Loaders** | |
| `names_dataset.py` | Loads the names dataset for `NameSLP.py`. |
| `tinystories_dataset.py` | Loads the TinyStories dataset for all other models. |
| **The Models** | |
| `NameSLP.py` | NumPy single-layer perceptron. |
| `TinyMLP.py` | CuPy character MLP. |
| `TorchMLP.py` | PyTorch character MLP. |
| `SimpleTransformer.py` | Simplified PyTorch transformer. |
| `TinyTransformer.py` | PyTorch character transformer (Our Baseline). |
| `TinyTransformerClass.py` | OOP refactor of the TinyTransformer. |
| `TinyLlama.py` | Modern Llama-style transformer. |
| **Explained Guides** | |
| `TinyMLP-explained.md` | Walkthrough of `TinyMLP.py` (data flow, tensor shapes, gradients). |
| `SimpleTransformer-explained.md` | Walkthrough of the bridge from MLP to Transformer. |
| `TinyTransformer-explained.md` | Walkthrough of architecture choices and speed/quality optimization. |

---

## 💻 Requirements & Setup

This project is designed to run on **Google Colab** using a free T4 GPU. 

**You will need:**
*   Python 3.10+
*   An internet connection (to download the datasets on the first run)
*   Google Colab (or a local PC with an NVIDIA GPU)

**Python Packages:**
*   `numpy` (for math)
*   `cupy` (for GPU math in the early models)
*   `datasets` (Hugging Face library to stream the TinyStories text)
*   `torch` (PyTorch, for building the neural networks)

*Note: `TorchMLP.py` and `SimpleTransformer.py` can run on a normal CPU if you don't have a GPU, but they will be much slower!*

---

## 📖 The Datasets

We use two datasets to train our AI:
1.  **`names.txt`**: A list of real names (from Andrej Karpathy's `makemore` project). Used to teach the AI how to spell names.
2.  **TinyStories**: A dataset of simple, AI-generated children's stories (from Hugging Face). We use this to teach the AI how to write basic sentences. We look at the data **character by character**, keeping the project simple and educational.

---

## 🚀 Suggested Next Improvements

Want to keep tinkering? Here are some future experiments you could try:
*   Add a CPU/NumPy fallback to `TinyMLP.py` so it runs without a GPU.
*   Add command-line arguments so you can change hyperparameters without editing the code.
*   Add a "Save/Load" feature so you don't have to retrain the model every time.
*   Add deterministic seeding so you get the exact same results every time you run an experiment.
```