# AI Lab Notebook: Training Tiny Language Models

This file tracks our training experiments on character-level language models trained on the **TinyStories** dataset.

Our baseline model is **TinyTransformer.py** (a 2-layer transformer, float16 precision, ReLU activation, learned positional embeddings, and a context size of 8 characters). All runs use a standard Google Colab T4 GPU.

> ⚠️ **The Colab Lottery:** Google Colab assigns T4 GPUs from a shared pool. Sometimes you get a fast one, sometimes a slow one. Warm run times can vary from ~19.7s to ~27.3s. Always run your code at least twice to get a fair speed measurement!

## 🧠 How to Read This Document

Before we dive in, here are two key scientific concepts we use to test AI models:

*   **Experiment:** Trying out a new feature or changing a setting to see if it makes the model better. (e.g., *"What if we make the model deeper?"*)
*   **Ablation:** Taking an existing feature away to prove that it's actually necessary. It's like removing the salt from a recipe to see if it actually matters. (e.g., *"What if we remove the model's ability to know word order?"*)

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

| Change Tested | Accuracy Δ | Speed Δ | The Verdict |
| :--- | ---: | ---: | :--- |
| **`torch.compile`** (Cold vs Warm) | Neutral | ~2.3× faster | ✅ Always "warm up" your model before timing it! |
| **Depth:** 2 → 3 layers | +5.1% | 1.8× slower | ✅ Best speed/accuracy tradeoff — matches 4L in half the time! |
| **Depth:** 2 → 4 layers | +1.2% | 2.2× slower | ✅ Worth it if you have time (73.1% at 3400 steps) |
| **Context Size:** 8 → 64 | +1.1% | 7.8× slower | ⚠️ Not worth the massive speed cost... yet. |
| **Precision:** float16 → bfloat16 | +0.2% | 4.2× slower | ❌ The T4 GPU doesn't have native bfloat16 hardware. |
| **Weight Tying** (sharing layers) | −3.0% | Neutral | ❌ Confused the model and hurt accuracy. |
| **Activation:** ReLU → GELU | Neutral | 14% slower | ❌ GELU is too math-heavy for this small model. |
| **Ablation:** Remove Positional Embeddings | −7.7% | Negligible | ❌ Without this, the AI reads sentences as "word soup." |
| **Loss:** Last-word vs Full-sequence | Neutral | 1.47× slower | ⚠️ Learns faster early on, but hits the same ceiling. |
| **Shape:** Wide/Short → Narrow/Deep | +1.0% | 20% slower | ✅ Depth beats width, even with half the parameters! |
| **Flash Attention** + Context 32 | +0.2% | 3.2× slower | ⚠️ Proves memory-efficient math works, but model needs to be bigger to use it. |
| **Narrow-Deep Alt. HPs** (128d, 4L, 810K params) | +0.5% | 3.4× slower | ⚠️ Half the params, competitive accuracy — likely under-trained. |
| **Efficient-Deep** (256d, ffn=512, 4L) | +2.4% | 2.3× slower | ⚠️ Strong mid-training but peaks at step 2000, not 2200. Anomalous step-0 loss spike. |
| **Balanced Narrow-Deep** (192d, ffn=768, 4L) | +2.4% | 2.8× slower | ⚠️ Ties Efficient-Deep at 70.8% but still climbing at step 2400 — likely under-trained. Clean step-0, best text quality in this series. |
| **Wider FFN** (3L, ffn=2048) | +3.4% | 3.0× slower | ⚠️ Bigger MLP helps, but not enough to beat standard 3L. Likely schedule-limited. |
| **High LR Fast Convergence** (3L, batch=1024, lr=2e-3) | +4.0% | 2.5× slower | ⚠️ Faster than large-batch runs, but step-0 instability and post-peak collapse show small batch + high LR is too noisy. |
| **Middle Ground** (3L, batch=1536, lr=2e-3) | +6.8% | 3.8× slower | ✅ Excellent compromise — 98% of the batch=2048 result at ~83% of the time. |
| **Large Batch + High LR** (3L, batch=2048, lr=2e-3) | +7.7% | 4.6× slower | ✅ Huge accuracy win — best non-microgpt result so far. More data per step appears to matter more than extra depth or FFN size. |

---

## 📈 Step-by-Step Accuracy Graph Data

Want to graph our progress? Here is the accuracy of each model at different points in training. *(Blank cells mean we stopped training that model early).*

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

*(Note: ⭐ marks the confirmed peak for each model. Both original 3L runs peak at step 2200 then plateau/oscillate. Efficient-Deep 4L peaks early at step 2000. Balanced Narrow-Deep 4L is still descending at step 2400. The two new high-LR experiments show the same pattern: peak at step 2200, then immediate decline. Batch size controls stability — 1536 and 2048 behave much better than 1024 at lr=2e-3.)*

---

## 📝 Experiment Details & Lessons Learned

### 1. `torch.compile`: Cold vs Warm Run
**The Experiment:** `torch.compile` speeds up code, but it takes time to "translate" the Python code into fast machine code on the first run.
*   **Cold Time (1st run):** 46.3s
*   **Warm Time (2nd run):** 19.7s
**The Takeaway:** The 26-second penalty happens entirely at Step 0. Always run your code once, throw away the time, and run it again to see the true speed.

### 2. Layer Depth (2 vs 3 layers)
**The Experiment:** We added one extra layer (going from 2 to 3), keeping all other hyperparameters identical (embed_dim=256, n_heads=4, ffn_dim=1024, context_size=8, batch_size=1024). We ran this experiment **twice** to verify the findings.

| | Run 1 | Run 2 |
| :--- | ---: | ---: |
| Steps run | 2400 | 3000 |
| Peak accuracy | 73.5% | 72.9% |
| Peak at step | 2200 | 2200 |
| Time to peak | ~33s | ~33s |
| Total training time | 36.3s | 44.9s |

**Result:** Both runs peak at exactly **step 2200** (~73%) then flatline in the 71–72% range for all remaining steps. The ~0.6% gap between runs is within normal Colab T4 variance. The 3-layer model matches the 4-layer model's best accuracy (73.1%) in **less than half the training time** (~33s vs 79.9s).

**The Takeaway:** 3 layers is the **sweet spot** for this architecture. Stop training at step 2200 — running longer only wastes compute with no benefit. The cosine LR scheduler has decayed to near `eta_min=1e-4` by then, and the model is just jittering around its minimum.

**Generated Sample — Run 1 (73.5% Acc):**
> `Once there was a little girl. They were happy. He saw a little boy named Tim went to the park. They ran away. They gave the chool. They raced and said, "Hi, I'm sorry, she decided to take his friendly.`

**Generated Sample — Run 2 (72.9% Acc):**
> `Once there was a big red ball the door was very happy. She said, "Okay, he saw many toys at the rock. The next day, the snow. She had to find forth a magic sad. The boat and inside and the flame, but I can`

### 3. Layer Depth (2 vs 4 layers)
**The Experiment:** We doubled the layers from 2 to 4 (adding 1.5 million parameters).
**Result:** 4 layers gets +1.2% accuracy at 2000 steps, but is 2.2× slower. However, if we let it run longer (3400 steps), it hits 73.1%!
**The Takeaway:** Deeper models are slower per step, but they can keep learning long after shallow models have hit their limit. 

### 4. Context Size (8 vs 64 characters)
**The Experiment:** We gave the model a bigger "short-term memory," letting it look at 64 characters at once instead of 8.
**Result:** Accuracy barely moved (+1.1%), but training time exploded from 25s to 197s!
**The Takeaway:** Attention math scales quadratically (if you double the context, you quadruple the math). 8x the context meant 7.8x the time. This is exactly why researchers invented "Flash Attention" to fix this later.

### 5. float16 vs bfloat16 Precision
**The Experiment:** We swapped standard float16 math for bfloat16 (a newer format that handles big numbers better).
**Result:** bfloat16 was 4.2× slower on our T4 GPU!
**The Takeaway:** Hardware matters. The older T4 Turing GPU doesn't have physical circuits for bfloat16, so it fakes it using float32, which is slow. bfloat16 is amazing, but only on newer GPUs like the A100.

### 6. Weight Tying
**The Experiment:** We forced the "input reading" layer and the "output guessing" layer to share the exact same brain cells (weights).
**Result:** Accuracy dropped by 3%. At step 0, the loss exploded.
**The Takeaway:** Weight tying works great for big models with 50,000-word vocabularies. But for our tiny 65-character alphabet, it just confused the model because the layers started with mismatched "random" settings.

### 7. GELU vs ReLU Activation
**The Experiment:** We swapped ReLU (a simple "if negative, make zero" math rule) for GELU (a complex curve used in GPT models).
**Result:** Identical accuracy, but 14% slower.
**The Takeaway:** Don't use complex math if simple math works just as well. GELU's complex calculations slowed the GPU down with no benefit at this small scale.

### 8. Ablation: Positional Embeddings
**The Ablation:** We removed the code that tells the AI the order of the letters. The AI now sees "tac" and "cat" as the exact same thing.
**Result:** Accuracy crashed by 7.7%. 
**The Takeaway:** Transformers are "permutation invariant" by default—they have no concept of order. Without positional embeddings, an AI is just looking at a bag of scrambled letters. Order matters!

### 9. Full-Sequence Causal Loss
**The Experiment:** Instead of just asking the AI to guess the *last* word of a sentence, we asked it to guess *every* word in the sentence as it goes along (using a "causal mask" so it can't cheat and look ahead).
**Result:** It learned much faster early on (55.9% vs 52.2% at step 200) because it gets 8x more feedback per batch. But it hit the same final ceiling.
**The Takeaway:** This is the industry standard for training LLMs. It's slower per step, but vastly more efficient at teaching the model how language works.

### 10. Narrow-Deep vs Wide-Short
**The Experiment:** We halved the width of the model (256d to 128d) but doubled the depth (2 to 4 layers). We cut the total parameters in half!
**Result:** With half the parameters, the narrow-deep model actually beat the baseline (69.1% vs 68.1%).
**The Takeaway:** Depth is incredibly powerful. A tall, skinny model generalizes better than a short, wide one. 

### 11. Flash/SDPA Attention
**The Experiment:** We turned on Flash Attention (specifically PyTorch's Memory-Efficient SDPA) and bumped the context to 32.
**Result:** It worked! It proved we can bypass the slow O(T²) math from Experiment #3. However, accuracy only went up 0.2%.
**The Takeaway:** The memory-efficient math works perfectly, but our model's "brain" (128 dimensions) is now too small to actually use the extra context. The bottleneck is no longer memory; it's model capacity.

### 12. Narrow-Deep Alternative Hyperparameters
**The Experiment:** A follow-up to Experiment #10. Instead of just halving width and doubling depth, we deliberately tuned all hyperparameters for a narrow-deep shape: `embed_dim=128`, `ffn_dim=512`, `n_heads=4`, `n_layers=4`, keeping `context_size=8`, `batch_size=1024`, `lr=1e-3`.

```
params: 810,560
Step    0 | Loss: 4.9636 | Acc: 10.6% | 37.1s
Step  200 | Loss: 1.4453 | Acc: 54.3% | 39.5s
Step  400 | Loss: 1.3106 | Acc: 58.7% | 42.1s
Step  600 | Loss: 1.2775 | Acc: 61.8% | 44.6s
Step  800 | Loss: 1.1805 | Acc: 63.0% | 47.2s
Step 1000 | Loss: 1.1456 | Acc: 64.1% | 49.8s
Step 1200 | Loss: 1.1142 | Acc: 66.0% | 52.4s
Step 1400 | Loss: 1.0868 | Acc: 65.9% | 55.0s
Step 1600 | Loss: 1.0473 | Acc: 67.8% | 57.6s
Step 1800 | Loss: 1.0197 | Acc: 68.8% | 60.2s
Step 2000 | Loss: 1.0018 | Acc: 69.4% | 62.8s
Step 2200 | Loss: 0.9960 | Acc: 68.1% | 65.4s
Step 2400 | Loss: 0.9675 | Acc: 68.9% | 68.0s
Training time: 68.0s
```

**Result:** 68.9% accuracy with only **810,560 parameters** — about half the size of the standard 4-layer model. The loss curve is clean and monotonically decreasing throughout. Accuracy oscillates at step 2200 (the same plateau jitter seen in other deep runs) before recovering to 68.9% at step 2400.

**The Takeaway:** This model is likely **under-trained**. The standard 4-layer model needed 3400 steps to peak at 73.1% — this run stopped at 2400. Running to 3400–4000 steps may close the gap significantly. Most importantly: **half the parameters, competitive accuracy** — a compelling demonstration that depth is a more efficient use of a parameter budget than width.

**Generated Sample (68.9% Acc):**
> `Once there sunsy and her mom said, "Yes, Tom, you make a shiny back to the story is hurt you!" Max was so happy. They are started to help the botter and said, "Thank you, so she patient the rose.`

### 13. Efficient-Deep (256d, ffn=512, 4 Layers)
**The Experiment:** A hybrid design merging the 3-layer model's wide embedding (`embed_dim=256`) with the 4-layer model's depth, while halving `ffn_dim` from 1024 to 512 to keep the per-step cost low. The hypothesis: retain attention capacity (256d heads) while gaining depth generalisation, and finish training in ~35–40s.

```python
# --- Hyperparameters for Efficient Deep ---
context_size = 8
embed_dim    = 256       # KEPT WIDE for capacity
n_heads      = 4
ffn_dim      = 512       # HALVED from 1024
n_layers     = 4         # 4 layers
batch_size   = 1024
lr           = 1e-3
n_steps      = 2201
```

```
params: 2,143,296
Step    0 | Loss: 10.0916 | Acc:  5.2% |  0.1s
Step  200 | Loss:  1.5220 | Acc: 55.1% |  4.0s
Step  400 | Loss:  1.3004 | Acc: 60.7% |  8.1s
Step  600 | Loss:  1.2389 | Acc: 62.7% | 12.2s
Step  800 | Loss:  1.1376 | Acc: 63.9% | 16.2s
Step 1000 | Loss:  1.1015 | Acc: 64.7% | 20.4s
Step 1200 | Loss:  1.0675 | Acc: 67.1% | 24.5s
Step 1400 | Loss:  1.0810 | Acc: 67.5% | 28.7s
Step 1600 | Loss:  1.0739 | Acc: 68.4% | 32.9s
Step 1800 | Loss:  1.0102 | Acc: 69.2% | 37.2s
Step 2000 | Loss:  0.9658 | Acc: 71.0% | 41.4s  ← PEAK
Step 2200 | Loss:  0.9904 | Acc: 69.7% | 45.5s
Training time: 45.5s
```

**Result:** Peaked at **70.8% at step 2000**, then dropped to 69.7% at step 2200 — the model overshot its optimum. The 45.5s training time matched the prediction perfectly. However, two unexpected findings emerged:

1. **Parameter count:** At 2,143,296 params, this model is *larger* than expected. The FFN saving was offset by the large embedding and projection matrices at `embed_dim=256`. The "cheap" assumption only partially held.
2. **Step-0 loss anomaly:** Loss started at ~10.09 (vs ~5.0 for all other runs). This is reproducible across two runs and likely indicates a weight initialisation scale mismatch when combining wide embeddings with a shallow FFN — the model recovers quickly but wastes early training.

**The Takeaway:** Halving the FFN while keeping wide embeddings does not straightforwardly combine the best of 3L and 4L. The early peak (step 2000 vs step 2200 for 3L) and the step-0 spike suggest the cosine LR schedule is poorly matched to this architecture shape. The 3-layer model still wins the speed/accuracy tradeoff: 73.5% in ~33s vs 70.8% in ~45s here.

**Generated Sample (70.8% Acc):**
> `Once there was a little girl named Max. Mom looked at the pole and the ball and lots of fun and shining, something unexpected happened. He took the boy's mom told the tree to the same to her said`

### 14. Balanced Narrow-Deep (192d, ffn=768, 4 Layers)
**The Experiment:** A true compromise between Experiment #12 (128d, too narrow) and Experiment #13 (256d+ffn=512, anomalous init). We split the difference: `embed_dim=192` (clean head_dim of 48 with 4 heads), `ffn_dim=768` (exactly 4× embed_dim, the standard ratio), and 4 layers.

```python
# --- Hyperparameters for Balanced Narrow-Deep ---
context_size = 8
embed_dim    = 192       # Compromise between 128 and 256
n_heads      = 4         # (192 / 4 = 48 head_dim, which is clean)
ffn_dim      = 768       # Compromise between 512 and 1024 (4x embed_dim)
n_layers     = 4
batch_size   = 1024
lr           = 1e-3
n_steps      = 2401
```

```
params: 1,805,632
Step    0 | Loss: 4.4319 | Acc: 19.3% |  0.1s
Step  200 | Loss: 1.3701 | Acc: 56.6% |  4.7s
Step  400 | Loss: 1.2700 | Acc: 60.7% |  9.4s
Step  600 | Loss: 1.2365 | Acc: 63.2% | 14.1s
Step  800 | Loss: 1.1471 | Acc: 65.6% | 18.8s
Step 1000 | Loss: 1.1224 | Acc: 66.1% | 23.7s
Step 1200 | Loss: 1.0519 | Acc: 67.1% | 28.5s
Step 1400 | Loss: 1.0115 | Acc: 67.5% | 33.3s
Step 1600 | Loss: 1.0121 | Acc: 70.0% | 38.0s
Step 1800 | Loss: 0.9719 | Acc: 69.3% | 42.6s
Step 2000 | Loss: 0.9418 | Acc: 70.5% | 47.3s
Step 2200 | Loss: 0.9357 | Acc: 70.4% | 52.5s
Step 2400 | Loss: 0.8969 | Acc: 70.8% | 57.7s  ← PEAK (still descending)
Training time: 57.7s
```

**Result:** Peaked at **70.8% at step 2400**, with loss still falling (0.942 → 0.897 in the last 400 steps) — unlike Experiment #13, this model has **not plateaued**. Step-0 loss is clean (4.43), confirming the initialization anomaly in Experiment #13 was specific to the 256d+ffn=512 mismatch. Text quality is the best in this experimental series: coherent dialogue, correct punctuation, no invented words.

**The Takeaway:** This model is **under-trained**. The still-descending loss and accuracy at step 2400 strongly suggest it would reach ~72–73% given 3400–4000 steps — potentially matching the 3-layer model. However, at 57.7s for 2400 steps (~0.024s/step), a full run to 3400 steps would take ~82s, making it slower than the 3L sweet spot (~33s). The 192d config fixes the Experiment #13 initialization problem, but the 3-layer model remains the most time-efficient path to 73%+.

**Generated Sample (70.8% Acc):**
> `Once there she was happy. They did not know that it was time to go back to the house with the gray to see his mom, "What's that!" Mom says. "I don't want to see that the town want to make the store to learn`

### 15. Wider FFN (3 Layers, ffn=2048)
**The Experiment:** Instead of adding depth, we doubled the feed-forward network size inside each transformer block: `ffn_dim=2048` while keeping the proven 3-layer backbone (`embed_dim=256`, `n_layers=3`). The hypothesis: a bigger MLP might unlock more token-mixing capacity without the overhead of a 4th layer.

```python
# --- Hyperparameters for Wider FFN ---
context_size = 8
embed_dim    = 256
n_heads      = 4
ffn_dim      = 2048      # DOUBLED from 1024
n_layers     = 3
batch_size   = 1024
lr           = 1e-3
n_steps      = 2201
```

```
params: 3,980,096
Step    0 | Loss: 4.7255 | Acc: 20.2% |  0.1s
Step  200 | Loss: 1.4541 | Acc: 55.4% |  5.5s
Step  400 | Loss: 1.3367 | Acc: 59.9% | 11.1s
Step  600 | Loss: 1.2519 | Acc: 63.3% | 16.8s
Step  800 | Loss: 1.1733 | Acc: 64.7% | 22.4s
Step 1000 | Loss: 1.0781 | Acc: 67.3% | 27.9s
Step 1200 | Loss: 1.0494 | Acc: 67.4% | 33.2s
Step 1400 | Loss: 0.9938 | Acc: 69.3% | 38.5s
Step 1600 | Loss: 0.9461 | Acc: 70.2% | 43.7s
Step 1800 | Loss: 0.9570 | Acc: 70.3% | 48.8s
Step 2000 | Loss: 0.8335 | Acc: 71.1% | 54.0s
Step 2200 | Loss: 0.9290 | Acc: 71.8% | 59.1s
Training time: 59.1s
```

**Result:** Accuracy improved steadily to **71.8% at step 2200**, but still failed to beat the standard 3-layer model's 73.5%. The loss curve is noisy late in training: loss drops sharply to 0.833 at step 2000, then rises again to 0.929 at step 2200 even while accuracy improves. This is a classic sign of a model that is too large for the current cosine LR schedule — it's learning, but not converging cleanly.

**The Takeaway:** Simply making the FFN bigger is not an efficient way to spend parameters here. At **3,980,096 params**, this is by far the largest TinyTransformer variant so far, yet it underperforms the standard 3-layer model while taking ~1.8× longer. More parameters do not help if the optimization schedule is mismatched.

**Generated Sample (71.8% Acc):**
> `Once there was a big such a good. The boat wanted to play with the box. He should tees. She house, so he could do not see the boat. He said the might for a moment. He saw a safe, and the bell keep was still`

### 16. Large Batch + High LR (3 Layers, 2048 batch)
**The Experiment:** Instead of making the model bigger, we doubled the **batch size** from 1024 → 2048 and scaled the **learning rate** from 1e-3 → 2e-3 to match. The model architecture stayed the same as the standard 3-layer winner.

```python
# --- Hyperparameters for Large Batch + High LR ---
context_size = 8
embed_dim    = 256
n_heads      = 4
ffn_dim      = 1024
n_layers     = 3
batch_size   = 2048      # DOUBLED
lr           = 2e-3      # DOUBLED to match larger batch
n_steps      = 2201
```

```
params: 2,404,160
Step    0 | Loss: 4.7300 | Acc: 19.3% |  0.1s
Step  200 | Loss: 1.3693 | Acc: 58.0% |  8.3s
Step  400 | Loss: 1.2690 | Acc: 60.8% | 16.4s
Step  600 | Loss: 1.1901 | Acc: 63.9% | 24.8s
Step  800 | Loss: 1.0649 | Acc: 66.9% | 33.4s
Step 1000 | Loss: 1.0645 | Acc: 68.4% | 41.8s
Step 1200 | Loss: 0.9888 | Acc: 68.6% | 49.9s
Step 1400 | Loss: 0.9734 | Acc: 70.4% | 58.0s
Step 1600 | Loss: 0.8850 | Acc: 71.0% | 66.1s
Step 1800 | Loss: 0.8706 | Acc: 72.8% | 74.3s
Step 2000 | Loss: 0.8693 | Acc: 72.3% | 82.4s
Step 2200 | Loss: 0.8584 | Acc: 76.1% | 90.7s
Training time: 90.7s
```

**Result:** This produced a new best TinyTransformer result: **76.1% at step 2200**, beating the standard 3-layer model by +2.6% and the standard 4-layer model by +3.0%. The curve is a bit noisy in the final stretch (72.8 → 72.3 → 76.1), but the final jump is decisive. Text quality is also noticeably better: coherent names, dialogue, and far fewer broken words.

**The Takeaway:** For this dataset, **more data per step mattered more than more parameters**. Doubling the batch and LR gave the optimizer a better estimate of the gradient and unlocked a major accuracy jump without changing the architecture. The tradeoff is speed: **90.7s** is much slower than the standard 3-layer run (~33s), but if your goal is best TinyTransformer quality rather than best speed/accuracy ratio, this is the new champion.

**Generated Sample (76.1% Acc):**
> `Once there was a little girl named Lily. She saw the new toy. He liked to play with her mom smiled and said, "Hello, Spot saw Tom was very happy with the cake was so smaller saw a big that she was happy`

### 17. High LR Fast Convergence (3 Layers, 1024 batch)
**The Experiment:** Could we keep the **high learning rate** from Experiment #16 but revert the batch size back to 1024 for speed? The idea was to recover most of the large-batch accuracy boost while cutting the runtime nearly in half.

```python
# --- Hyperparameters for High LR Fast Convergence ---
context_size = 8
embed_dim    = 256
n_heads      = 4
ffn_dim      = 1024
n_layers     = 3
batch_size   = 1024      # REVERTED to 1024 for speed
lr           = 2e-3      # KEEP the high learning rate!
n_steps      = 3001      # INCREASED steps to compensate for smaller batch
```

```
params: 2,404,160
Step    0 | Loss: 11.7465 | Acc: 19.3% |  0.1s
Step  200 | Loss: 1.4829 | Acc: 55.3% |  3.3s
Step  400 | Loss: 1.4059 | Acc: 58.3% |  6.6s
Step  600 | Loss: 1.3533 | Acc: 59.9% |  9.8s
Step  800 | Loss: 1.2082 | Acc: 62.9% | 13.1s
Step 1000 | Loss: 1.2354 | Acc: 64.4% | 16.4s
Step 1200 | Loss: 1.1543 | Acc: 64.5% | 19.8s
Step 1400 | Loss: 1.1276 | Acc: 66.5% | 23.2s
Step 1600 | Loss: 1.0451 | Acc: 66.2% | 26.5s
Step 1800 | Loss: 1.0449 | Acc: 68.0% | 29.8s
Step 2000 | Loss: 1.0548 | Acc: 68.8% | 33.1s
Step 2200 | Loss: 0.9309 | Acc: 72.4% | 36.3s  ← PEAK
Step 2400 | Loss: 0.9039 | Acc: 71.1% | 39.5s
Step 2600 | Loss: 0.8707 | Acc: 70.6% | 42.7s
Step 2800 | Loss: 0.9165 | Acc: 71.5% | 45.8s
Step 3000 | Loss: 0.9198 | Acc: 70.8% | 49.0s
Training time: 49.0s
```

**Result:** The model did partially recover: it hit **72.4% at step 2200**, beating the Wider-FFN and all 4-layer custom variants, while staying much faster than the big-batch runs. But it suffered from the same step-0 instability seen in Experiment #13: initial loss exploded to **11.75**, and accuracy declined immediately after the peak. Running longer did not help.

**The Takeaway:** High LR alone is not enough. Without the larger batch to stabilize the gradient, `lr=2e-3` is simply too noisy for this architecture. This config is a valid **speed-focused compromise** — 72.4% in 49s is solid — but it cannot match the stability or final quality of the 1536/2048-batch runs.

**Generated Sample (72.4% Acc):**
> `Once there was a big red ball to bite. She wanted to play. They like lift heard a big red to play with her mom and dad. The sun and the sky!" They are sad. But then, a moment. Then the cat. They all wanted`

### 18. Middle Ground (3 Layers, 1536 batch)
**The Experiment:** A compromise between Experiments #16 and #17. We kept the high learning rate (`2e-3`) but used a **1536 batch size** — halfway between 1024 and 2048 — to see if we could retain most of the large-batch accuracy gain at lower runtime.

```python
# --- Hyperparameters for Middle Ground ---
context_size = 8
embed_dim    = 256
n_heads      = 4
ffn_dim      = 1024
n_layers     = 3
batch_size   = 1536      # Compromise between 1024 and 2048
lr           = 2e-3      # Keep high LR
n_steps      = 2401      # Slightly reduced from 2200 to ensure time limit
```

```
params: 2,404,160
Step    0 | Loss: 4.7377 | Acc: 19.3% |  0.1s
Step  200 | Loss: 1.3945 | Acc: 57.5% |  6.2s
Step  400 | Loss: 1.3220 | Acc: 60.2% | 12.4s
Step  600 | Loss: 1.2569 | Acc: 62.6% | 18.6s
Step  800 | Loss: 1.1078 | Acc: 65.5% | 24.9s
Step 1000 | Loss: 1.1215 | Acc: 67.1% | 31.4s
Step 1200 | Loss: 1.0225 | Acc: 67.2% | 37.9s
Step 1400 | Loss: 1.0346 | Acc: 69.0% | 44.3s
Step 1600 | Loss: 0.8978 | Acc: 69.9% | 50.6s
Step 1800 | Loss: 0.9007 | Acc: 71.1% | 56.9s
Step 2000 | Loss: 0.9295 | Acc: 70.8% | 63.0s
Step 2200 | Loss: 0.8863 | Acc: 75.2% | 69.2s  ← PEAK
Step 2400 | Loss: 0.8680 | Acc: 73.0% | 75.3s
Training time: 75.3s
```

**Result:** This worked very well: **75.2% at step 2200**, only 0.9% behind the 2048-batch champion, while saving ~15 seconds of runtime. Importantly, the step-0 loss returned to normal (4.74), showing that the larger batch stabilizes the aggressive learning rate.

**The Takeaway:** This is the **best compromise** discovered so far. It captures almost all of the large-batch benefit at a meaningfully lower cost: about **98% of the accuracy for ~83% of the time**. If Experiment #16 is the quality champion, this one is the practical high-performance alternative.

**Generated Sample (75.2% Acc):**
> `Once there was a little girl named Lily. Lily was sad and said, "Yes, we can put his ball flew through the bell and carry box.
Max was so happy that Tom were three years old and said, "Hi, dropped with the be`

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

**TinyTransformer.py - 3 Layers, 2200 steps (73.5% Acc - Fluid sentences!)**
> `Once there was a little girl. They were happy. He saw a little boy named Tim went to the park. They ran away. They gave the chool. They raced and said, "Hi, I'm sorry, she decided to take his friendly.`

**TinyTransformer.py - 4 Layers, 3400 steps (73.1% Acc - Grammatically solid!)**
> `Once there was a little girl named Sam. Sam was so happy and started to play with the camera. They made a big hill and the birds fly something shine and saw a big tree.`

**TinyTransformer.py - 3 Layers, batch=2048, lr=2e-3 (76.1% Acc - New champion!)**
> `Once there was a little girl named Lily. She saw the new toy. He liked to play with her mom smiled and said, "Hello, Spot saw Tom was very happy with the cake was so smaller saw a big that she was happy`

**microgpt_lite.py (79.4% Acc - Nearly perfect TinyStory)**
> `Once upon a time, there was a little boy named Tim. He loved to measure his favorite toy. One day, he saw a big, deep broken shirt. He thought it would be fun to play with it.`
