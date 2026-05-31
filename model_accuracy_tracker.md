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

| Epoch | NameSLP.py | TinyMLP.py | TorchMLP.py | TinyTransformer.py |
|---:|---:|---:|---:|---:|
| 0 | 3.5% | 4.7% | 21.4% | 19.3% |
| 200 | 37.1% | 44.8% | 54.3% | 55.4% |
| 400 | 38.2% | 48.9% | 58.0% | 59.5% |
| 600 | 38.6% | 52.3% | 59.1% | 60.9% |
| 800 | 38.9% | 55.0% | 59.9% | 63.4% |
| 1000 | 39.1% | 56.4% | 60.8% | 65.9% |
| 1200 | 39.2% | 56.7% | 61.4% | 65.7% |
| 1400 | 39.4% | 58.2% | 60.8% | 67.0% |
| 1600 | 39.5% | 58.3% | 61.8% | 67.3% |
| 1800 | 39.5% | 59.2% | 61.1% | 68.1% |
| 2000 | 39.6% | 59.4% | 62.4% | 68.5% |

## Summary Comparison

| Model | Best Accuracy | Epoch | Training Time |
|---|---:|---:|---:|
| NameSLP.py | 39.6% | 2000 | 35.1s |
| TinyMLP.py | 59.4% | 2000 | 3.9s |
| TorchMLP.py | 62.4% | 2000 | 3.6s |
| TinyTransformer.py | 68.5% | 2000 | 29.2s |

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
Once there was a little bird would help his friends to play unnot started to find his friends while it!
The asked his mom park outside in the sun bag beak tree and got of the pick she could find the bird sa
```