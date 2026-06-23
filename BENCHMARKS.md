# AI Lab Notebook: Training Tiny Language Models

This file tracks our training experiments on character-level language models trained on the **TinyStories** dataset.

Our baseline model is **TinyTransformer.py** (a 2-layer transformer, float16 precision, ReLU activation, learned positional embeddings, and a context size of 8 characters). All runs use a standard Google Colab T4 GPU.

> ⚠️ **The Colab Lottery:** Google Colab assigns T4 GPUs from a shared pool. Sometimes you get a fast one, sometimes a slow one. Warm run times can vary from ~19.7s to ~27.3s. Always run your code at least twice to get a fair speed measurement!

---

## 🧠 How to Read This Document

Before we dive in, here are two key scientific concepts we use to test AI models. Think of the model like a recipe or a PC build:

*   **🧪 Experiment:** Trying out a *new feature* or *upgrading a setting* to see if it makes the model better. (e.g., *"What if we add more layers to the model's brain?"* or *"What if we double the memory?"*)
*   **✂️ Ablation:** Taking an existing feature *away* to prove that it's actually necessary. It's like removing the baking powder from a cake recipe to see if it actually matters. (e.g., *"What if we remove the model's ability to know word order?"*)

**The Golden Rule of AI Testing:** Every entry below changes **only one thing at a time**. This is the scientific method—if we change 5 things and the model gets better, we won't know which of the 5 actually caused the improvement!

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
| TinyTransformer.py (Narrow-Deep 4L, 810K params) | 68.9% | 2400 | 68.0s |
| TinyTransformer.py (Efficient-Deep 4L, ffn=512) | 70.8% | 2000 | 45.5s |
| TinyTransformer.py (Balanced Narrow-Deep 4L, 192d) | 70.8% | 2400 | 57.7s |
| TinyTransformer.py (3 layers, Wider FFN=2048) | 71.8% | 2200 | 59.1s |
| TinyTransformer.py (3 layers, batch=1024, lr=2e-3) | 72.4% | 2200 | 49.0s |
| TinyTransformer.py (4 layers) | 73.1% | 3400 | 79.9s |
| TinyTransformer.py (3 layers) | 73.5% | 2200 | ~33s |
| TinyTransformer.py (3 layers, batch=1536, lr=2e-3) | 75.2% | 2200 | 75.3s |
| **TinyTransformer.py (3 layers, batch=2048, lr=2e-3)** 🥇 | **76.1%** | **2200** | **90.7s** |
| microgpt_lite.py | 79.4% | 3500 | 202.0s |

---

## 🔬 Ablation & Experiment Summary

Here is the quick cheat sheet of what we learned. All tests below are single changes made to our baseline 2-layer TinyTransformer (~68% accuracy, ~21s training time).

| Type | Change Tested | Accuracy Δ | Speed Δ | The Verdict |
| :--- | :--- | ---: | ---: | :--- |
| **Exp** | **`torch.compile`** (Cold vs Warm) | Neutral | ~2.3× faster | ✅ Always "warm up" your model before timing it! |
| **Exp** | **Depth:** 2 → 3 layers | +5.1% | 1.8× slower | ✅ Best speed/accuracy tradeoff. |
| **Exp** | **Depth:** 2 → 4 layers | +1.2% | 2.2× slower | ✅ Worth it if you have time (73.1% at 3400 steps). |
| **Exp** | **Context Size:** 8 → 64 | +1.1% | 7.8× slower | ⚠️ Not worth the massive speed cost... yet. |
| **Exp** | **Precision:** float16 → bfloat16 | +0.2% | 4.2× slower | ❌ The T4 GPU doesn't have native bfloat16 hardware. |
| **Exp** | **Weight Tying** (sharing layers) | −3.0% | Neutral | ❌ Confused the model and hurt accuracy. |
| **Exp** | **Activation:** ReLU → GELU | Neutral | 14% slower | ❌ GELU is too math-heavy for this small model. |
| **Abl** | **Remove Positional Embeddings** | −7.7% | Negligible | ❌ Without this, the AI reads sentences as "word soup." |
| **Exp** | **Loss:** Last-word vs Full-sequence | Neutral | 1.47× slower | ⚠️ Learns faster early on, but hits the same ceiling. |
| **Exp** | **Shape:** Wide/Short → Narrow/Deep | +1.0% | 20% slower | ✅ Depth beats width, even with half the parameters! |
| **Exp** | **Flash Attention** + Context 32 | +0.2% | 3.2× slower | ⚠️ Proves memory-efficient math works, but model is too small to use it. |
| **Exp** | **Narrow-Deep Alt. HPs** (128d, 4L) | +0.5% | 3.4× slower | ⚠️ Half the params, competitive accuracy. Under-trained. |
| **Exp** | **Efficient-Deep** (256d, ffn=512, 4L) | +2.4% | 2.3× slower | ⚠️ Strong mid-training but peaks early. Anomalous step-0 loss spike. |
| **Exp** | **Balanced Narrow-Deep** (192d, 4L) | +2.4% | 2.8× slower | ⚠️ Ties Efficient-Deep but still climbing at step 2400. |
| **Exp** | **Wider FFN** (3L, ffn=2048) | +3.4% | 3.0× slower | ⚠️ Bigger MLP helps, but not enough to beat standard 3L. |
| **Exp** | **High LR Fast Convergence** (batch=1024) | +4.0% | 2.5× slower | ⚠️ Faster, but high LR makes training unstable. |
| **Exp** | **Middle Ground** (batch=1536) | +6.8% | 3.8× slower | ✅ Excellent compromise. |
| **Exp** | **Large Batch + High LR** (batch=2048) | +7.7% | 4.6× slower | ✅ Huge accuracy win — best non-microgpt result so far. |

---

## 📈 Step-by-Step Accuracy Graph Data

Want to graph our progress? Here is the accuracy of each model at different points in training. *(Blank cells mean we stopped training that model early).*

*Teacher Note: Look at the 2200 step mark. You'll see many models "peak" here and then start to drop. This means the model has learned all it can, and continuing to train actually makes it worse (overfitting)!*

| Step | NameSLP | TinyMLP | SimpleTrans | **TinyTrans (2L)** | TinyTrans (3L) Run 1 | TinyTrans (3L) Run 2 | TinyTrans (4L) | Narrow-Deep 4L | Efficient-Deep 4L | Balanced ND 4L | Wider FFN 3L | High LR 3L | Middle Ground 3L | Large Batch+LR 3L | microgpt |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 3.5% | 4.7% | 4.0% | 19.3% | 19.3% | 19.3% | 19.3% | 10.6% | 5.2% | 19.3% | 20.2% | 19.3% | 19.3% | 19.3% | 1.7% |
| 200 | 37.1% | 44.8% | 53.5% | 54.8% | 55.8% | 56.1% | 56.8% | 54.3% | 55.1% | 56.6% | 55.4% | 55.3% | 57.5% | 58.0% | 53.6% |
| 400 | 38.2% | 48.9% | 58.6% | 58.3% | 59.7% | 59.8% | 60.7% | 58.7% | 60.7% | 60.7% | 59.9% | 58.3% | 60.2% | 60.8% | 65.2% |
| 800 | 38.9% | 55.0% | 62.4% | 63.2% | 64.8% | 64.6% | 64.6% | 63.0% | 63.9% | 65.6% | 64.7% | 62.9% | 65.5% | 66.9% | 71.4% |
| 1200 | 39.2% | 56.7% | 64.7% | 65.5% | 66.6% | 66.2% | 66.6% | 66.0% | 67.1% | 67.1% | 67.4% | 64.5% | 67.2% | 68.6% | 73.3% |
| 1600 | 39.5% | 58.3% | 66.2% | 67.0% | 67.6% | 67.4% | 68.0% | 67.8% | 68.4% | 70.0% | 70.2% | 66.2% | 69.9% | 71.0% | 76.0% |
| 2000 | 39.6% | 59.4% | 67.2% | 67.4% | 70.2% | 68.8% | 68.9% | 69.4% | **70.8%** ⭐ | 70.5% | 71.1% | 68.8% | 70.8% | 72.3% | 77.0% |
| 2200 | - | - | - | - | **73.5%** ⭐ | **72.9%** ⭐ | - | 68.1% | 69.7% | 70.4% | **71.8%** ⭐ | **72.4%** ⭐ | **75.2%** ⭐ | **76.1%** ⭐ | - |
| 2400 | - | - | - | - | 71.7% | 71.2% | - | 68.9% | - | **70.8%** ⭐ | - | 71.1% | 73.0% | - | - |
| 2600 | - | - | - | - | - | 70.9% | - | - | - | - | - | 70.6% | - | - | - |
| 2800 | - | - | - | - | - | 71.5% | - | - | - | - | - | 71.5% | - | - | - |
| 3000 | - | - | - | - | - | 71.5% | - | - | - | - | - | 70.8% | - | - | - |
| 3400 | - | - | - | - | - | - | 73.1% | - | - | - | - | - | - | - | - |
| 3500 | - | - | - | - | - | - | - | - | - | - | - | - | - | - | 79.4% |

*(Note: ⭐ marks the confirmed peak for each model. Notice how most models hit their peak around step 2200 and then plateau or drop. This means leaving them training longer is just a waste of electricity!)*

---

## 📝 Experiment Details & Lessons Learned

### 🧪 EXPERIMENTS: Upgrading the Engine
*These tests try adding or changing features to see if we can build a better, faster, or smarter model.*

**1. `torch.compile`: Cold vs Warm Run**
*   **The Change:** `torch.compile` speeds up code, but it takes time to "translate" the Python code into fast machine code on the first run.
*   **Result:** Cold Time (1st run) = 46.3s. Warm Time (2nd run) = 19.7s.
*   **The Takeaway:** The 26-second penalty happens entirely at Step 0. Always run your code once, throw away the time, and run it again to see the true speed. Like warming up a car engine!

**2. Layer Depth (2 vs 3 layers)**
*   **The Change:** We added one extra layer (going from 2 to 3), keeping everything else identical. We ran this twice to verify.
*   **Result:** Both runs peaked at **step 2200** (~73%) then flatlined. The 3-layer model matches the 4-layer model's best accuracy (73.1%) in **less than half the training time** (~33s vs 79.9s).
*   **The Takeaway:** 3 layers is the **sweet spot** for this architecture. Stop training at step 2200—running longer only wastes compute.

**3. Layer Depth (2 vs 4 layers)**
*   **The Change:** We doubled the layers from 2 to 4 (adding 1.5 million parameters).
*   **Result:** 4 layers gets +1.2% accuracy at 2000 steps, but is 2.2× slower. However, if we let it run longer (3400 steps), it hits 73.1%!
*   **The Takeaway:** Deeper models are slower per step, but they can keep learning long after shallow models have hit their limit. 

**4. Context Size (8 vs 64 characters)**
*   **The Change:** We gave the model a bigger "short-term memory," letting it look at 64 characters at once instead of 8.
*   **Result:** Accuracy barely moved (+1.1%), but training time exploded from 25s to 197s!
*   **The Takeaway:** Attention math scales quadratically (if you double the context, you quadruple the math). 8x the context meant 7.8x the time. This is exactly why researchers invented "Flash Attention" to fix this later.

**5. float16 vs bfloat16 Precision**
*   **The Change:** We swapped standard float16 math for bfloat16 (a newer format that handles big numbers better).
*   **Result:** bfloat16 was 4.2× slower on our T4 GPU!
*   **The Takeaway:** Hardware matters. The older T4 Turing GPU doesn't have physical circuits for bfloat16, so it fakes it using float32, which is slow. bfloat16 is amazing, but only on newer GPUs.

**6. Weight Tying**
*   **The Change:** We forced the "input reading" layer and the "output guessing" layer to share the exact same brain cells (weights).
*   **Result:** Accuracy dropped by 3%. At step 0, the loss exploded.
*   **The Takeaway:** Weight tying works great for big models with 50,000-word vocabularies. But for our tiny 65-character alphabet, it just confused the model.

**7. GELU vs ReLU Activation**
*   **The Change:** We swapped ReLU (a simple "if negative, make zero" math rule) for GELU (a complex curve used in GPT models).
*   **Result:** Identical accuracy, but 14% slower.
*   **The Takeaway:** Don't use complex math if simple math works just as well. GELU's complex calculations slowed the GPU down with no benefit at this small scale.

**8. Full-Sequence Causal Loss**
*   **The Change:** Instead of just asking the AI to guess the *last* word of a sentence, we asked it to guess *every* word in the sentence as it goes along (using a "causal mask" so it can't cheat and look ahead).
*   **Result:** It learned much faster early on (55.9% vs 52.2% at step 200) because it gets 8x more feedback per batch. But it hit the same final ceiling.
*   **The Takeaway:** This is the industry standard for training LLMs. It's slower per step, but vastly more efficient at teaching the model how language works.

**9. Narrow-Deep vs Wide-Short**
*   **The Change:** We halved the width of the model (256d to 128d) but doubled the depth (2 to 4 layers). We cut the total parameters in half!
*   **Result:** With half the parameters, the narrow-deep model actually beat the baseline (69.1% vs 68.1%).
*   **The Takeaway:** Depth is incredibly powerful. A tall, skinny model generalizes better than a short, wide one. 

**10. Flash/SDPA Attention**
*   **The Change:** We turned on Flash Attention and bumped the context to 32.
*   **Result:** It worked! It proved we can bypass the slow O(T²) math from Experiment #4. However, accuracy only went up 0.2%.
*   **The Takeaway:** The memory-efficient math works perfectly, but our model's "brain" (128 dimensions) is now too small to actually use the extra context. The bottleneck is no longer memory; it's model capacity.

*(Experiments 11-14: Architecture Tweaks)*
*   **Narrow-Deep Alt (128d, 4L, 810K params):** 68.9% acc. Half the params, competitive accuracy. Compelling proof that depth is an efficient use of parameters, though it stopped training too early.
*   **Efficient-Deep (256d, ffn=512, 4L):** 70.8% acc at 45.5s. Halving the FFN while keeping wide embeddings doesn't easily combine the best of both. Suffered from a weird step-0 loss spike.
*   **Balanced Narrow-Deep (192d, ffn=768, 4L):** 70.8% acc at 57.7s. Still improving at step 2400. Fixed the initialization problem from the previous run, but takes too long to be worth it.
*   **Wider FFN (3L, ffn=2048):** 71.8% acc at 59.1s. Simply making the FFN bigger is not efficient. At nearly 4 million params, it underperforms the standard 3-layer model.

**15. Large Batch + High LR (3 Layers, 2048 batch)**
*   **The Change:** Instead of making the model bigger, we doubled the **batch size** (data processed at once) from 1024 → 2048 and doubled the **learning rate** from 1e-3 → 2e-3 to match.
*   **Result:** A new best TinyTransformer result: **76.1% at step 2200**, beating the standard 3-layer model by +2.6%. Text quality is noticeably better: coherent names, dialogue, and far fewer broken words.
*   **The Takeaway:** For this dataset, **more data per step mattered more than more parameters**. Doubling the batch and LR gave the optimizer a better estimate of the gradient. The tradeoff is speed (90.7s), but if your goal is best quality, this is the champion.

**16. High LR Fast Convergence (3 Layers, 1024 batch)**
*   **The Change:** Keep the high learning rate (2e-3) but revert the batch size back to 1024 for speed. 
*   **Result:** Hit 72.4% at step 2200, but suffered from instability (initial loss exploded to 11.75, and accuracy declined immediately after the peak).
*   **The Takeaway:** High LR alone is not enough. Without the larger batch to stabilize the gradient, a high learning rate is simply too noisy. 

**17. Middle Ground (3 Layers, 1536 batch)**
*   **The Change:** A compromise. High learning rate (2e-3) but a 1536 batch size—halfway between 1024 and 2048.
*   **Result:** 75.2% at step 2200, only 0.9% behind the 2048-batch champion, while saving ~15 seconds. Step-0 loss returned to normal, showing the larger batch stabilizes the aggressive learning rate.
*   **The Takeaway:** This is the **best compromise**. It captures 98% of the accuracy for 83% of the time.

---

### ✂️ ABLATION: Proving What Matters
*This test removes a crucial feature to prove why the AI needs it in the first place.*

**1. Ablation: Positional Embeddings**
*   **The Feature Removed:** We removed the code that tells the AI the order of the letters. The AI now sees "tac" and "cat" as the exact same thing.
*   **Result:** Accuracy crashed by 7.7%. 
*   **The Takeaway:** Transformers are "permutation invariant" by default—they have no concept of order. Without positional embeddings, an AI is just looking at a bag of scrambled letters. Order matters!

---

## 📖 Generated Samples (Seeing is Believing)

Numbers are great, but what does the AI actually write? Here are samples from our models, showing how they get smarter as we change the architecture and training settings.

**NameSLP.py (39.6% Acc - Just learns names)**
> emma, osola, riganna, ahala, horme, rayly...

**TinyMLP.py (59.4% Acc - Letters work, words are broken)**
> `Once tichec. Ther. She said outned. Sker to. Hif even very the box. It. I mesis momors.`

**SimpleTransformer.py (67.2% Acc - Almost real sentences)**
> `Once there was a faster. They learned the pusiade of the yell socked up and played together.`

**TinyTransformer.py (68.4% Acc - Baseline Transformer)**
> `Once there. She wise her bird was family face on on the thought it was so happy and put the tent down and said, "Mom, Tim, and they also much fun.`

**TinyTransformer.py - 3 Layers, 2200 steps (73.5% Acc - Fluid sentences!)**
> `Once there was a little girl. They were happy. He saw a little boy named Tim went to the park. They ran away. They gave the chool. They raced and said, "Hi, I'm sorry, she decided to take his friendly.`

**TinyTransformer.py - 4 Layers, 3400 steps (73.1% Acc - Grammatically solid!)**
> `Once there was a little girl named Sam. Sam was so happy and started to play with the camera. They made a big hill and the birds fly something shine and saw a big tree.`

**TinyTransformer.py - 3 Layers, batch=2048, lr=2e-3 (76.1% Acc - New champion!)**
> `Once there was a little girl named Lily. She saw the new toy. He liked to play with her mom smiled and said, "Hello, Spot saw Tom was very happy with the cake was so smaller saw a big that she was happy`

**microgpt_lite.py (79.4% Acc - Nearly perfect TinyStory)**
> `Once upon a time, there was a little boy named Tim. He loved to measure his favorite toy. One day, he saw a big, deep broken shirt. He thought it would be fun to play with it.`
```
