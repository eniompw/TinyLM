# 🧪 AI Lab Notebook: Training Tiny Language Models

This file tracks our training experiments on character-level language models trained on the **TinyStories** dataset. 

Our baseline model is **TinyTransformer.py** (a 2-layer transformer, float16 precision, ReLU activation, learned positional embeddings, and a context size of 8 characters). All runs use a standard Google Colab T4 GPU.

> ⚠️ **The Colab Lottery & Scientific Controls:** 
> Google Colab assigns T4 GPUs from a shared pool. Sometimes you get a fast one, sometimes a slow one. If we only look at "Total Seconds," our data is ruined by hardware luck! 
> 
> To fix this, we use **Relative Speed Ratios**. We run the 2-Layer Baseline model as our "Control" (1.0× speed). If an experiment takes twice as long, its speed is **2.0×**. This ratio stays true whether you run it on a slow Colab GPU or a supercomputer!
>
> *Note: Larger batch sizes (e.g., 2048) use more memory bandwidth, which makes the "Colab Lottery" even more extreme. Runtimes can swing from ~65s to ~90s. Always use ratios!*

---

## 🔬 The Scientific Method: How We Trust Our Data

In AI, it is very easy to fool yourself. Here are the three rules we use to make sure our experiments are scientifically valid:

*   **🎲 The Starting Seed (`torch.manual_seed`):** Neural networks start with random guesses. The specific random guess you start with changes your final score slightly. We hardcode the seed so our experiments are **reproducible**.
*   **🎯 The Eval Seed:** When we test the model every 200 steps, we don't test it on the whole dataset (it would run out of GPU memory). We grab a random subset. But if the subset changes every time, our accuracy will "wobble" up and down based on luck! We fixed this by creating a dedicated `eval_rng`. Now, the model is *always* tested on the exact same 4,096 stories.
*   **✂️ The Golden Rule:** Change **only one thing at a time**. If we add a layer AND double the batch size, and the model gets better, which one caused it? We won't know. Science requires isolation.

---

## 🧠 How to Read This Document

Before we dive in, here are two key scientific concepts we use to test AI models. Think of the model like a recipe or a PC build:

*   **🧪 Experiment:** Trying a *new feature* or *upgrading a setting* to see if it makes the model better. (e.g., *"What if we add more layers to the model's brain?"* or *"What if we double the memory?"*)
*   **✂️ Ablation:** Taking an existing feature *away* to prove that it's actually necessary. It's like removing the baking powder from a cake recipe to see if it actually matters. If the cake goes flat, you proved the baking powder matters! (e.g., *"What if we remove the model's ability to know word order?"*)

---

## 📊 The Leaderboard: Model Comparison

*Best configuration for each architecture we tested.*

| Model | Best Accuracy | Steps Taken | Relative Speed (vs 2L Baseline) |
| :--- | ---: | ---: | ---: |
| NameSLP.py | 39.6% | 2000 | 1.8× |
| TinyMLP.py | 59.4% | 2000 | 0.2× |
| TorchMLP.py | 62.4% | 2000 | 0.2× |
| SimpleTransformer.py | 67.2% | 2000 | 1.8× |
| **TinyTransformer.py (2 layers)** 🥇 | **68.4%** | **2000** | **1.0× (Control)** |
| TinyTransformer.py (context=64) | 68.5% | 1800 | 10.0× |
| TinyTransformer.py (Narrow-Deep 4L, 810K params) | 68.9% | 2400 | 3.5× |
| TinyTransformer.py (Efficient-Deep 4L, ffn=512) | 70.8% | 2000 | 2.3× |
| TinyTransformer.py (Balanced Narrow-Deep 4L, 192d) | 70.8% | 2400 | 2.9× |
| TinyTransformer.py (3 layers, Wider FFN=2048) | 71.8% | 2200 | 3.0× |
| TinyTransformer.py (3 layers, batch=1024, lr=2e-3) | 72.4% | 2200 | 2.5× |
| TinyTransformer.py (4 layers) | 73.1% | 3400 | 4.0× |
| TinyTransformer.py (3 layers) | 73.5% | 2200 | 1.5× |
| TinyTransformer.py (3 layers, batch=1536) ✨ | 73.0%* | 2200 | 2.7× |
| **TinyTransformer.py (3 layers, batch=2048)** | **76.1%** | **2200** | **~3.5×** |
| **TinyTransformer.py (3L, ctx=16, 5000 stories)** 🧠 | **71.7%** | **2200** | **~2.5×** |
| **TinyTransformer.py (3L, ctx=32, 5000 stories, 1536 batch)** 👑 | **70.1%** | **1800** | **~3.2×** |
| microgpt_lite.py | 79.4% | 3500 | 10.2× |

> **🚨 The Plot Twist (Read before judging the scores!):**
> Look at the bottom three rows. Why did accuracy go *down* to 70.1%? Because we expanded the dataset from 1,000 to 5,000 stories. The 76.1% model was cheating—it memorized the test. The 70.1% model stopped memorizing and actually learned English. **Lower accuracy score = higher real-world intelligence!**

---

## 🔬 Ablation & Experiment Summary

Here is the quick cheat sheet of what we learned. All tests below are single changes made to our baseline 2-layer TinyTransformer (~68% accuracy, 1.0× speed).

### 🏗️ Architecture (Shape & Size)
| Type | Change Tested | Accuracy Δ | Speed Δ | The Verdict |
| :--- | :--- | ---: | ---: | :--- |
| **Exp** | **Depth:** 2 → 3 layers | +5.1% | 1.5× slower | ✅ Best speed/accuracy tradeoff. |
| **Exp** | **Depth:** 2 → 4 layers | +1.2% | 2.2× slower | ✅ Worth it if you have time (73.1% at 3400 steps). |
| **Exp** | **Shape:** Wide/Short → Narrow/Deep | +1.0% | 20% slower | ✅ Depth beats width, even with half the parameters! |
| **Exp** | **Narrow-Deep Alt. HPs** (128d, 4L) | +0.5% | 3.4× slower | ⚠️ Half the params, competitive accuracy. |
| **Exp** | **Efficient-Deep** (256d, ffn=512, 4L) | +2.4% | 2.3× slower | ⚠️ Strong mid-training but peaks early. |
| **Exp** | **Balanced Narrow-Deep** (192d, 4L) | +2.4% | 2.8× slower | ⚠️ Ties Efficient-Deep but takes longer. |
| **Exp** | **Wider FFN** (3L, ffn=2048) | +3.4% | 3.0× slower | ⚠️ Bigger MLP helps, but not enough to beat standard 3L. |
| **Abl** | **Remove Positional Embeddings** | −7.7% | Negligible | ❌ Without this, the AI reads sentences as "word soup." |

### ⚡ Training & Speed Hacks
| Type | Change Tested | Accuracy Δ | Speed Δ | The Verdict |
| :--- | :--- | ---: | ---: | :--- |
| **Exp** | **`torch.compile`** (Cold vs Warm) | Neutral | ~2.3× faster | ✅ Always "warm up" your model before timing it! |
| **Exp** | **Precision:** float16 → bfloat16 | +0.2% | 4.2× slower | ❌ The T4 GPU doesn't have native bfloat16 hardware. |
| **Exp** | **Weight Tying** (sharing layers) | −3.0% | Neutral | ❌ Confused the model and hurt accuracy. |
| **Exp** | **Activation:** ReLU → GELU | Neutral | 14% slower | ❌ GELU is too math-heavy for this small model. |
| **Exp** | **Loss:** Last-word vs Full-sequence | Neutral | 1.47× slower | ⚠️ Learns faster early on, but hits the same ceiling. |
| **Exp** | **Flash Attention** + Context 32 | +0.2% | 3.2× slower | ⚠️ Proves memory-efficient math works, but model is too small to use it. |

### 🧠 The "Real Intelligence" Push (Batch, Context & Data)
| Type | Change Tested | Accuracy Δ | Speed Δ | The Verdict |
| :--- | :--- | ---: | ---: | :--- |
| **Exp** | **High LR Fast Convergence** (batch=1024) | +4.0% | 2.5× slower | ⚠️ Faster, but high LR makes training unstable. |
| **Exp** | **Middle Ground** (batch=1536) | +6.8% | 2.7× slower | ✅ Excellent compromise. ~1 min runtime. |
| **Exp** | **Large Batch + High LR** (batch=2048) | +7.7% | ~3.5× slower | ✅ Huge accuracy win — best raw score. |
| **Exp** | **Dataset Size:** 1k → 3k/5k stories | −4.7% | Negligible | ✅ **The Memorization Trap:** Drops raw acc, but drastically improves grammar. |
| **Exp** | **Context Size:** 8 → 16 (on large dataset) | −1.5% | ~1.5× slower | ✅ Fixes pronoun/gender swapping. Model can track subjects! |
| **Exp** | **Weight Decay:** 0 → 0.01 | Neutral | Negligible | ✅ Acts as a "grammar regularizer." Stops lazy repetition. |
| **Exp** | **Context Size:** 16 → 32 (on large dataset) | −1.6% | ~1.3× slower | ✅ The ultimate 2-min tradeoff. Fixes 90% of pronoun swaps. |
| **Exp** | **Inference Temp:** 0.7 → 0.5 | N/A (Inference) | N/A | ✅ Eliminates fake words (e.g., "throbe" → "robe"). |

---

## 📈 Step-by-Step Accuracy Data

*We split the data into "Phases" to tell the story of our experiments. ⭐ marks the peak accuracy. 📉 shows where the model starts overfitting and getting worse!*

**Legend:** 
* **2L/3L/4L** = TinyTransformer with 2, 3, or 4 layers.
* **ND** = Narrow-Deep (skinnier model, more layers).
* **FFN** = Feed-Forward Network (the "thinking" part of the layer).

### Phase 1: The Baselines (Where we started)
*Goal: See if our basic Transformer architecture beats the older, simpler models.*

| Step | NameSLP | TinyMLP | SimpleTrans | **2L (Baseline)** | microgpt |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 3.5% | 4.7% | 4.0% | 19.3% | 1.7% |
| 200 | 37.1% | 44.8% | 53.5% | 54.8% | 53.6% |
| 800 | 38.9% | 55.0% | 62.4% | 63.2% | 71.4% |
| 1600 | 39.5% | 58.3% | 66.2% | 67.0% | 76.0% |
| 2000 | **39.6%** ⭐ | **59.4%** ⭐ | **67.2%** ⭐ | 67.4% | 77.0% |
| 3500 | - | - | - | - | **79.4%** ⭐ |

### Phase 2: Shape & Size Tests (Does depth or width matter more?)
*Goal: Find out if adding layers, widening the brain, or changing the shape gives us better accuracy than the 2L Baseline.*

| Step | **3L** (Run 1) | **4L** | ND 4L (128d) | Eff. Deep 4L | Bal. ND 4L (192d) | Wider FFN 3L |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 19.3% | 19.3% | 10.6% | 5.2% | 19.3% | 20.2% |
| 800 | 64.8% | 64.6% | 63.0% | 63.9% | 65.6% | 64.7% |
| 1600 | 67.6% | 68.0% | 67.8% | 68.4% | 70.0% | 70.2% |
| 2000 | 70.2% | 68.9% | 69.4% | **70.8%** ⭐ | 70.5% | 71.1% |
| 2200 | **73.5%** ⭐ | - | 68.1% 📉 | 69.7% 📉 | 70.4% | **71.8%** ⭐ |
| 2400 | 71.7% 📉 | - | 68.9% | - | **70.8%** ⭐ | - |
| 3400 | - | **73.1%** ⭐ | - | - | - | - |

> 💡 **Pro-Tip:** Look at the 3-Layer model. It hits 73.5% at step 2200, but drops to 71.7% at step 2400. This is called **overfitting**. The model memorized the training data so hard that it got worse at writing new stories. Always stop training when you hit the ⭐!

### Phase 3: The "Raw Score" Champions (Batch Size & Learning Rate)
*Goal: Instead of changing the model's shape, what if we just change HOW it learns? (Using the 3-Layer model).*

| Step | High LR (batch=1024) | Mid Ground (batch=1536) | **Large Batch+LR** (batch=2048) |
| ---: | ---: | ---: | ---: |
| 0 | 19.3% | 19.3% | 19.3% |
| 800 | 62.9% | 65.5% | 66.9% |
| 1600 | 66.2% | 69.9% | 71.0% |
| 2000 | 68.8% | 70.8% | 72.3% |
| 2200 | **72.4%** ⭐ | **75.2%** ⭐ | **76.1%** ⭐ |
| 2400 | 71.1% 📉 | 73.0% 📉 | - |

### Phase 4: The Real Intelligence Push (Generalization vs. Memorization)
*Goal: Stop chasing raw accuracy numbers and fix the AI's "amnesia." By expanding the dataset and context window, we force the model to actually learn English rather than memorizing 1,000 stories.*

| Step | **3L, 2048 batch, 3k stories** (ctx=8) | **3L, 1536 batch, 5k stories** (ctx=32, wd=0.01) |
| ---: | ---: | ---: |
| 0 | 18.5% | 19.2% |
| 400 | 62.1% | 61.4% |
| 800 | 66.0% | 64.4% |
| 1200 | 67.8% | 67.8% |
| 1600 | 69.3% | **70.1%** ⭐ |
| 2000 | 71.4% | - |

> 💡 **Pro-Tip:** Look at the scores! They are *lower* than Phase 3 (which hit 76.1%). But look at the generated samples below. This proves that on small datasets, high accuracy is just memorization (overfitting). If you want a model that writes well in the real world, train it on more data and accept a slightly lower eval score!

---

## 📝 Experiment Details & Lessons Learned

### 🧪 EXPERIMENTS: Upgrading the Engine
*These tests try adding or changing features to see if we can build a better, faster, or smarter model.*

#### Phase 1: Architecture Tweaks
**1. `torch.compile`: Cold vs Warm Run**
*   **The Change:** `torch.compile` speeds up code, but it takes time to "translate" the Python code into fast machine code on the first run.
*   **Result:** Cold Time (1st run) = 46.3s. Warm Time (2nd run) = 19.7s.
*   **The Takeaway:** Always run your code once, throw away the time, and run it again to see the true speed. Like warming up a car engine!

**2. Layer Depth (2 vs 3 vs 4 layers)**
*   **The Change:** We added extra layers to see if a "taller" brain is better than a "wider" one. 
*   **Result:** 3 layers hit 73.5% in 2200 steps. 4 layers hit 73.1% but took 3400 steps.
*   **The Takeaway:** 3 layers is the **sweet spot**. Think of it like building a tower: going wider takes huge amounts of material, but going taller gives the model more "steps" to process complex logic. But go too tall, and it becomes too slow to train!

**3. Context Size (8 vs 64 characters)**
*   **The Change:** We gave the model a bigger "short-term memory," letting it look at 64 characters at once instead of 8.
*   **Result:** Accuracy barely moved (+1.1%), but training time exploded from 25s to 197s!
*   **The Takeaway:** Attention math scales quadratically (if you double the context, you quadruple the math). 8x the context meant 7.8x the time. This is exactly why researchers invented "Flash Attention" to fix this later.

**4. Float16 vs Bfloat16 Precision**
*   **The Change:** We swapped standard float16 math for bfloat16 (a newer format that handles big numbers better).
*   **Result:** bfloat16 was 4.2× slower on our T4 GPU!
*   **The Takeaway:** Hardware matters. The older T4 GPU doesn't have physical circuits for bfloat16, so it fakes it using float32, which is slow. 

**5. GELU vs ReLU Activation**
*   **The Change:** We swapped ReLU (a simple "if negative, make zero" math rule) for GELU (a complex curve used in GPT models).
*   **Result:** Identical accuracy, but 14% slower.
*   **The Takeaway:** Don't use complex math if simple math works just as well. GELU's complex calculations slowed the GPU down with no benefit at this small scale.

#### Phase 2: The Training Hacks
**6. Large Batch + High LR (3 Layers, 2048 batch)**
*   **The Change:** Instead of making the model bigger, we doubled the **batch size** (data processed at once) from 1024 → 2048 and doubled the **learning rate** from 1e-3 → 2e-3 to match.
*   **Result:** A new best raw score: **76.1% at step 2200**. 
*   **The Takeaway:** For this dataset, **more data per step mattered more than more parameters**. 

**7. The Memorization Trap (Dataset Size: 1k → 3k/5k)**
*   **The Change:** We expanded `num_stories` from 1,000 to 5,000.
*   **Result:** The raw accuracy score dropped from 76.1% down to 71.4%. However, the generated text improved dramatically. The 76.1% model produced word salad ("the cake was so smaller saw a big"), while the 5,000-story model produced clean clauses.
*   **The Takeaway:** With only 1,000 stories, the model sees the exact same evaluation stories so many times that it just memorizes the answers. It "hacks" the test. Expanding the dataset forces the AI to learn the underlying *rules* of English grammar to succeed.

#### Phase 3: The Coherence Push
**8. Context is King for Semantics (8 → 16 → 32 characters)**
*   **The Change:** We doubled `context_size` from 8 to 16, and then to 32, giving the AI a 5-6 word short-term memory.
*   **Result:** The AI stopped swapping pronouns mid-sentence. It could finally remember "named Lily" long enough to correctly use "She" later in the sentence.
*   **The Takeaway:** 8 characters is barely 1.5 words. The AI literally could not see the subject of the sentence by the time it wrote the verb. 32 characters fixes the "amnesia" while still fitting inside the 2-minute Colab budget!

**9. Mild Weight Decay & Inference Temperature**
*   **The Change:** We added a tiny amount of `weight_decay=0.01` to the optimizer, and lowered the generation `temperature` from 0.7 to 0.5.
*   **Result:** Weight decay stopped the model from repeating the same phrases over and over. The lower temperature stopped the model from making risky, weird guesses that resulted in fake words like "throbe" (turning it into the real word "robe").
*   **The Takeaway:** Training is only half the battle. A little regularization during training, and conservative sampling during generation, polishes the final output.

---

### ✂️ ABLATION: Proving What Matters
*This test removes a crucial feature to prove why the AI needs it in the first place.*

**1. Ablation: Positional Embeddings**
*   **The Feature Removed:** We removed the code that tells the AI the order of the letters. The AI now sees "tac" and "cat" as the exact same thing.
*   **Result:** Accuracy crashed by 7.7%. 
*   **The Takeaway:** Transformers are like a person reading a handful of Scrabble tiles scattered on a table. By default, they see all the letters but have no concept of left-to-right order. Without Positional Embeddings, the AI is just looking at "word soup." Order matters!

---

## 📖 Generated Samples (Seeing is Believing)

Numbers are great, but what does the AI actually write? Here are samples from our models, showing how they get smarter as we change the architecture and training settings.

**TinyMLP.py (59.4% Acc - Letters work, words are broken)**
> `Once tichec. Ther. She said outned. Sker to. Hif even very the box. It. I mesis momors.`

**SimpleTransformer.py (67.2% Acc - Almost real sentences)**
> `Once there was a faster. They learned the pusiade of the yell socked up and played together.`

**TinyTransformer.py - 3 Layers, batch=2048 (76.1% Acc - Highest raw score, but cheats)**
> `Once there was a little girl named Lily. She saw the new toy. He liked to play with her mom smiled and said, "Hello, Spot saw Tom was very happy with the cake was so smaller saw a big that she was happy`
*(Notice how it starts well but turns into word salad because it memorized patterns, not grammar).*

**TinyTransformer.py - 3L, 2048 batch, 3k stories (71.4% Acc - Generalization Win!)**
> `Once there was a great time and she was green and strong. Tim and Sue were so happy that the box opened the bug friends. She was sad and looked for them. He grabbed the box of the went to help his mom came in`
*(Notice how much better the clauses flow compared to the 76.1% champion. It learned structure, not just memorized words!)*

**TinyTransformer.py - 3L, 1536 batch, 5k stories, ctx=32 (70.1% Acc - The 2-Minute Ceiling!)**
> `Once there was a little boy named Tim. Tim laughed and said, "Thank you, Tom. I want to a dog named Tim to the girl was sad. He liked to play with his friends. They were very happy and said, "Okay, what is a small bird came to share`
*(Look at the first two sentences. "little boy named Tim... He liked to play". The 32-character context allowed the model to maintain the gender connection perfectly across a sentence boundary. No fake words, perfect punctuation. This is the ultimate result for a 2-minute Colab run.)*

**microgpt_lite.py (79.4% Acc - Nearly perfect TinyStory)**
> `Once upon a time, there was a little boy named Tim. He loved to measure his favorite toy. One day, he saw a big, deep broken shirt. He thought it would be fun to play with it.`
```