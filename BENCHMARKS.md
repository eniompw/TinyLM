# AI Lab Notebook: Training Tiny Language Models

This file tracks our training experiments on character-level language models trained on the **TinyStories** dataset. 

Our baseline model is **TinyTransformer.py** (a 2-layer transformer, float16 precision, ReLU activation, learned positional embeddings, and a context size of 8 characters). All runs use a standard Google Colab T4 GPU.

> ⚠️ **The Colab Lottery:** Google Colab assigns T4 GPUs from a shared pool. Sometimes you get a fast one, sometimes a slow one. Warm run times can vary from ~19.7s to ~27.3s. Always run your code at least twice to get a fair speed measurement!

## 🧠 How to Read This Document

Before we dive in, here are two key scientific concepts we use to test AI models:

*   **Experiment:** Trying out a new feature or changing a setting to see if it makes the model better. (e.g., *“What if we make the model deeper?”*)
*   **Ablation:** Taking an existing feature away to prove that it’s actually necessary. It’s like removing the salt from a recipe to see if it actually matters. (e.g., *“What if we remove the model's ability to know word order?”*)

Every entry below changes **only one thing at a time**. This is the scientific method—if we change 5 things and the model gets better, we won't know which of the 5 caused the improvement!

---

## 📊 The Leaderboard: Model Comparison

*Best configuration for each architecture we tested.*

| Model | Best Accuracy | Steps Taken | Training Time |
| :--- | ---: | ---: | ---: |
| NameSLP.py | 39.6% | 2000 | 35.1s |
| TinyMLP.py | 59.4% | 2000 | 3.9s |
| TorchMLP.py | 62.4% | 2000 | 3.6s |
| SimpleTransformer.py | 67.2% | 2000 | 35.6s |
| **TinyTransformer.py (2 layers)** 🥇 | **68.4%** | **2000** | **19.7s** |
| TinyTransformer.py (context=64) | 68.5% | 1800 | 197.5s |
| TinyTransformer.py (4 layers) | 73.1% | 3400 | 79.9s |
| microgpt_lite.py | 79.4% | 3500 | 202.0s |

---

## 🔬 Ablation & Experiment Summary

Here is the quick cheat sheet of what we learned. All tests below are single changes made to our baseline 2-layer TinyTransformer (~68% accuracy, ~21s training time).

| Change Tested | Accuracy Δ | Speed Δ | The Verdict |
| :--- | ---: | ---: | :--- |
| **`torch.compile`** (Cold vs Warm) | Neutral | ~2.3× faster | ✅ Always "warm up" your model before timing it! |
| **Depth:** 2 → 4 layers | +1.2% | 2.2× slower | ✅ Worth it if you have time (73.1% at 3400 steps) |
| **Context Size:** 8 → 64 | +1.1% | 7.8× slower | ⚠️ Not worth the massive speed cost... yet. |
| **Precision:** float16 → bfloat16 | +0.2% | 4.2× slower | ❌ The T4 GPU doesn't have native bfloat16 hardware. |
| **Weight Tying** (sharing layers) | −3.0% | Neutral | ❌ Confused the model and hurt accuracy. |
| **Activation:** ReLU → GELU | Neutral | 14% slower | ❌ GELU is too math-heavy for this small model. |
| **Ablation:** Remove Positional Embeddings | −7.7% | Negligible | ❌ Without this, the AI reads sentences as "word soup." |
| **Loss:** Last-word vs Full-sequence | Neutral | 1.47× slower | ⚠️ Learns faster early on, but hits the same ceiling. |
| **Shape:** Wide/Short → Narrow/Deep | +1.0% | 20% slower | ✅ Depth beats width, even with half the parameters! |
| **Flash Attention** + Context 32 | +0.2% | 3.2× slower | ⚠️ Proves memory-efficient math works, but model needs to be bigger to use it. |

---

## 📈 Step-by-Step Accuracy Graph Data

Want to graph our progress? Here is the accuracy of each model at different points in training. *(Blank cells mean we stopped training that model early).*

| Step | NameSLP | TinyMLP | SimpleTrans | **TinyTrans (2L)** | TinyTrans (4L) | microgpt |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 3.5% | 4.7% | 4.0% | 19.3% | 19.3% | 1.7% |
| 200 | 37.1% | 44.8% | 53.5% | 54.8% | 56.8% | 53.6% |
| 400 | 38.2% | 48.9% | 58.6% | 58.3% | 60.7% | 65.2% |
| 800 | 38.9% | 55.0% | 62.4% | 63.2% | 64.6% | 71.4% |
| 1200 | 39.2% | 56.7% | 64.7% | 65.5% | 66.6% | 73.3% |
| 1600 | 39.5% | 58.3% | 66.2% | 67.0% | 68.0% | 76.0% |
| 2000 | 39.6% | 59.4% | 67.2% | 67.4% | 68.9% | 77.0% |
| 3400 | - | - | - | - | 73.1% | - |
| 3500 | - | - | - | - | - | 79.4% |

*(Note: Some models like TinyTrans-4L take more steps to reach their peak, while simple models plateau very early).*

---

## 📝 Experiment Details & Lessons Learned

### 1. `torch.compile`: Cold vs Warm Run
**The Experiment:** `torch.compile` speeds up code, but it takes time to "translate" the Python code into fast machine code on the first run.
*   **Cold Time (1st run):** 46.3s
*   **Warm Time (2nd run):** 19.7s
**The Takeaway:** The 26-second penalty happens entirely at Step 0. Always run your code once, throw away the time, and run it again to see the true speed.

### 2. Layer Depth (2 vs 4 layers)
**The Experiment:** We doubled the layers from 2 to 4 (adding 1.5 million parameters).
**Result:** 4 layers gets +1.2% accuracy at 2000 steps, but is 2.2× slower. However, if we let it run longer (3400 steps), it hits 73.1%!
**The Takeaway:** Deeper models are slower per step, but they can keep learning long after shallow models have hit their limit. 

### 3. Context Size (8 vs 64 characters)
**The Experiment:** We gave the model a bigger "short-term memory," letting it look at 64 characters at once instead of 8.
**Result:** Accuracy barely moved (+1.1%), but training time exploded from 25s to 197s!
**The Takeaway:** Attention math scales quadratically (if you double the context, you quadruple the math). 8x the context meant 7.8x the time. This is exactly why researchers invented "Flash Attention" to fix this later.

### 4. float16 vs bfloat16 Precision
**The Experiment:** We swapped standard float16 math for bfloat16 (a newer format that handles big numbers better).
**Result:** bfloat16 was 4.2× slower on our T4 GPU!
**The Takeaway:** Hardware matters. The older T4 Turing GPU doesn't have physical circuits for bfloat16, so it fakes it using float32, which is slow. bfloat16 is amazing, but only on newer GPUs like the A100.

### 5. Weight Tying
**The Experiment:** We forced the "input reading" layer and the "output guessing" layer to share the exact same brain cells (weights).
**Result:** Accuracy dropped by 3%. At step 0, the loss exploded.
**The Takeaway:** Weight tying works great for big models with 50,000-word vocabularies. But for our tiny 65-character alphabet, it just confused the model because the layers started with mismatched "random" settings.

### 6. GELU vs ReLU Activation
**The Experiment:** We swapped ReLU (a simple "if negative, make zero" math rule) for GELU (a complex curve used in GPT models).
**Result:** Identical accuracy, but 14% slower.
**The Takeaway:** Don't use complex math if simple math works just as well. GELU's complex calculations slowed the GPU down with no benefit at this small scale.

### 7. Ablation: Positional Embeddings
**The Ablation:** We removed the code that tells the AI the order of the letters. The AI now sees "tac" and "cat" as the exact same thing.
**Result:** Accuracy crashed by 7.7%. 
**The Takeaway:** Transformers are "permutation invariant" by default—they have no concept of order. Without positional embeddings, an AI is just looking at a bag of scrambled letters. Order matters!

### 8. Full-Sequence Causal Loss
**The Experiment:** Instead of just asking the AI to guess the *last* word of a sentence, we asked it to guess *every* word in the sentence as it goes along (using a "causal mask" so it can't cheat and look ahead).
**Result:** It learned much faster early on (55.9% vs 52.2% at step 200) because it gets 8x more feedback per batch. But it hit the same final ceiling.
**The Takeaway:** This is the industry standard for training LLMs. It’s slower per step, but vastly more efficient at teaching the model how language works.

### 9. Narrow-Deep vs Wide-Short
**The Experiment:** We halved the width of the model (256d to 128d) but doubled the depth (2 to 4 layers). We cut the total parameters in half!
**Result:** With half the parameters, the narrow-deep model actually beat the baseline (69.1% vs 68.1%).
**The Takeaway:** Depth is incredibly powerful. A tall, skinny model generalizes better than a short, wide one. 

### 10. Flash/SDPA Attention
**The Experiment:** We turned on Flash Attention (specifically PyTorch's Memory-Efficient SDPA) and bumped the context to 32.
**Result:** It worked! It proved we can bypass the slow O(T²) math from Experiment #3. However, accuracy only went up 0.2%.
**The Takeaway:** The memory-efficient math works perfectly, but our model's "brain" (128 dimensions) is now too small to actually use the extra context. The bottleneck is no longer memory; it's model capacity.

---

## 📖 Generated Samples (Seeing is Believing)

Numbers are great, but what does the AI actually write? Here are samples from our models, showing how they get smarter.

**NameSLP.py (39.6% Acc - Just learns names)**
> emma, osola, riganna, ahala, horme, rayly...

**TinyMLP.py (59.4% Acc - Letters work, words are broken)**
> `Once tichec. Ther. She said outned. Sker to. Hif even very the box. It. I mesis momors.`

**SimpleTransformer.py (67.2% Acc - Almost real sentences)**
> `Once there was a faster. They learned the pusiade of the yell socked up and played together.`

**TinyTransformer.py (68.4% Acc - Baseline Transformer)**
> `Once there. She wise her bird was family face on on the thought it was so happy and put the tent down and said, "Mom, Tim, and they also much fun.`

**TinyTransformer.py - 4 Layers, 3400 steps (73.1% Acc - Grammatically solid!)**
> `Once there was a little girl named Sam. Sam was so happy and started to play with the camera. They made a big hill and the birds fly something shine and saw a big tree.`

**microgpt_lite.py (79.4% Acc - Nearly perfect TinyStory)**
> `Once upon a time, there was a little boy named Tim. He loved to measure his favorite toy. One day, he saw a big, deep broken shirt. He thought it would be fun to play with it.`
