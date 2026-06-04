# Language Model Accuracy Tracker

This file tracks training accuracy for language model experiments run on Google Colab.

## Runtime Environment

- Platform: Google Colab
- GPU: T4 GPU
- Backend: Python 3 (Google Compute Engine, GPU)
- System RAM usage: 8.5 / 12.7 GB
- GPU RAM usage: 12.2 / 15.0 GB
- Disk usage: 47.2 / 112.6 GB

## Accuracy Comparison

**Key:** TT = TinyTransformer.py, TTC = TinyTransformerClass.py

| Epoch | NameSLP.py | TinyMLP.py | TorchMLP.py | TT | TTC | microgpt_lite.py | LlamaLite |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 3.5% | 4.7% | 21.4% | 19.2% | 19.3% | 1.7% | 19.6% |
| 200 | 37.1% | 44.8% | 54.3% | 54.6% | 53.8% | 53.6% | 47.3% |
| 400 | 38.2% | 48.9% | 58.0% | 59.5% | 59.0% | 65.2% | 53.7% |
| 600 | 38.6% | 52.3% | 59.1% | 61.0% | 61.3% | 68.6% | 57.1% |
| 800 | 38.9% | 55.0% | 59.9% | 63.7% | 63.4% | 71.4% | 58.3% |
| 1000 | 39.1% | 56.4% | 60.8% | 66.3% | 65.4% | 71.9% | 60.9% |
| 1200 | 39.2% | 56.7% | 61.4% | 66.0% | 65.5% | 73.3% | 62.6% |
| 1400 | 39.4% | 58.2% | 60.8% | 67.4% | 66.5% | 74.6% | 63.0% |
| 1600 | 39.5% | 58.3% | 61.8% | 67.4% | 66.6% | 76.0% | 64.1% |
| 1800 | 39.5% | 59.2% | 61.1% | 67.9% | 67.3% | 75.9% | 66.4% |
| 2000 | 39.6% | 59.4% | 62.4% | 68.4% | 68.0% | 77.0% | 65.6% |
| 3500 | - | - | - | - | - | 79.4% | - |

## Summary Comparison

| Model | Best Accuracy | Epoch | Training Time |
|---|---:|---:|---:|
| NameSLP.py | 39.6% | 2000 | 35.1s |
| TinyMLP.py | 59.4% | 2000 | 3.9s |
| TorchMLP.py | 62.4% | 2000 | 3.6s |
| TinyTransformer.py | 68.4% | 2000 | 20.9s |
| TinyTransformer.py (`context_size=8`, prev run) | 67.4% | 1600 | 25.4s |
| TinyTransformer.py (`context_size=64`) | 68.5% | 1800 | 197.5s |
| TinyTransformerClass.py | 68.0% | 2000 | 42.5s |
| microgpt_lite.py | 79.4% | 3500 | 202.0s |
| LlamaLite (`context_size=32`, 1.59M params) | 66.4% | 1800 | 62.7s |

## Transformer Experiment Notes

- `5000` TinyStories often causes CUDA OOM/crash in this setup.
- Increasing transformer context window to `64` made training much slower, with only a small accuracy gain in this pair of runs.

### TinyTransformer Context Size Accuracy Comparison

| Epoch | `context_size=8` | `context_size=64` |
|---:|---:|---:|
| 0 | 19.8% | 19.0% |
| 200 | 55.7% | 55.5% |
| 400 | 59.4% | 58.4% |
| 600 | 60.7% | 61.3% |
| 800 | 62.7% | 64.4% |
| 1000 | 63.1% | 64.8% |
| 1200 | 66.3% | 64.4% |
| 1400 | 66.8% | 68.3% |
| 1600 | 67.4% | 67.6% |
| 1800 | 66.9% | 68.5% |
| 2000 | 67.0% | 68.4% |

Training time:

- `context_size=8`: `25.4s`
- `context_size=64`: `197.5s`

Quick comparison:

- Best accuracy: `67.4%` (`context_size=8`) vs `68.5%` (`context_size=64`) -> `+1.1` points.
- Training time: `25.4s` (`context_size=8`) vs `197.5s` (`context_size=64`) -> about `7.8x` slower at `context_size=64`.

## Generated Samples

### NameSLP.py

- emma
- osola
- riganna
- ahala
- horme
- rayly
- etannoye
- toraeyn
- alose
- gettel
- yandilon
- ceamira
- anleiph
- kafrin
- melia
- j

### TinyMLP.py

```text
Once tichec. Ther.
She said outned. Sker to. Hif even very the box. It. I mesis momors. He day.
"Se! smiled in outsy lows.
They played, it and said, "Yes, I wast, hure ats a creany five a bind. She saidy
```

### TorchMLP.py

```text
Once upon a time, to mak,""
The learry tried that her the corne but he saw two learned. She chess smal wife sell best couldn't my her and ran was a big for naughed loved clean withing. Mommy!"I will magin
```

### TinyTransformer.py

```text
Once there. She wise her bird was full and went out the thought it was too take a gift the big slid it, so he did not play and sun that day on, but they decided to go something for a moment finally
```

### TinyTransformerClass.py

```text
Once there was a little bird would help them to measure. They had so happy to the park with her bed the cat on, and he looked at the head to help. They are red back well. One day, a big blue.
```

### microgpt_lite.py

```text
Once upon a time, there was a little boy named Tim. He loved to measure his favorite toy. One day, he saw a big, deep broken shirt. He thought it would be fun to play with it.
```

### LlamaLite (`context_size=32`)

```text
Once there was a little boy named Tiny. The marked were very share hugged something and wanted to the walked to see a difffort to play outside. The little girl got red and got. Lily loved to play with his pretty
```
