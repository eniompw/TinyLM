# Language Model Accuracy Tracker

This file tracks training accuracy for language model experiments run on Google Colab.

## Runtime Environment

- Platform: Google Colab
- GPU: T4 GPU
- Backend: Python 3 (Google Compute Engine, GPU)
- System RAM usage: 8.5 / 12.7 GB
- GPU RAM usage: 12.2 / 15.0 GB
- Disk usage: 47.2 / 112.6 GB

## Run Log

### Model: NameSLP.py

| Epoch | Accuracy |
|---:|---:|
| 0 | 3.5% |
| 200 | 37.1% |
| 400 | 38.2% |
| 600 | 38.6% |
| 800 | 38.9% |
| 1000 | 39.1% |

Best recorded accuracy: 39.1% at epoch 1000.

Generated names sample:

- emma
- oroi
- atilah
- ria
- aidady
- mich
- olyian
- mirys
- phnaryia
- espe
- keyla
- azbellen
- carannse
- brtelyn
- iadnno
- kellane

### Model: TinyMLP.py

| Epoch | Accuracy |
|---:|---:|
| 0 | 4.7% |
| 200 | 44.8% |
| 400 | 48.9% |
| 600 | 52.3% |
| 800 | 55.0% |
| 1000 | 56.4% |
| 1200 | 56.7% |
| 1400 | 58.2% |
| 1600 | 58.3% |
| 1800 | 59.2% |
| 2000 | 59.4% |

Best recorded accuracy: 59.4% at epoch 2000.

Training time: 3.9s

Generated text sample:

```text
Once tichec. Ther.
She said outned. Sker to. Hif even very the box. It. I mesis momors. He day.
"Se! smiled in outsy lows.
They played, it and said, "Yes, I wast, hure ats a creany five a bind. She saidy
```

### Model: TorchMLP.py

| Epoch | Accuracy |
|---:|---:|
| 0 | 21.4% |
| 200 | 54.3% |
| 400 | 58.0% |
| 600 | 59.1% |
| 800 | 59.9% |
| 1000 | 60.8% |
| 1200 | 61.4% |
| 1400 | 60.8% |
| 1600 | 61.8% |
| 1800 | 61.1% |
| 2000 | 62.4% |

Best recorded accuracy: 62.4% at epoch 2000.

Training time: 3.6s

Generated text sample:

```text
Once upon a time, to mak,""
The learry tried that her the corne but he saw two learned. She chess smal wife sell best couldn't my her and ran was a big for naughed loved clean withing. Mommy!"I will magin
```

## Template For Next Models

Copy this section for each new model run:

### Model: <model_name>

- Date:
- Notes:

| Epoch | Accuracy |
|---:|---:|
| 0 |  |
| ... |  |

Best recorded accuracy:

Generated names sample:

- 