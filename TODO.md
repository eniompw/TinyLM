# TODO

- [ ] Normalize `context_size` vs `block_size` so the same context-window concept uses one name across files.
- [ ] Standardize dataset variable names for inputs and labels (`X`/`y`, `raw_in`/`raw_out`, `inputs`/`targets`) to one convention.
- [ ] Use one name for predicted class probabilities (`probs` / `probabilities` / `sample_probs`).
- [ ] Use one name for the running generation context (`ctx` / `context`).
- [ ] Review hardcoded context length `4` in `TorchMLP.py` and replace it with the shared context variable.

## Reference Map

- `context_size`: [NameSLP.py](NameSLP.py), [TinyTransformer.py](TinyTransformer.py)
- `block_size`: [TinyMLP.py](TinyMLP.py)
- hardcoded `4` for the context length: [TorchMLP.py](TorchMLP.py)
- `X`, `y`: [NameSLP.py](NameSLP.py)
- `raw_in`, `raw_out`: [TinyMLP.py](TinyMLP.py)
- `inputs`, `targets`: [TinyMLP.py](TinyMLP.py), [TorchMLP.py](TorchMLP.py), [TinyTransformer.py](TinyTransformer.py)
- `probs`: [NameSLP.py](NameSLP.py)
- `probabilities`: [TinyMLP.py](TinyMLP.py), [TorchMLP.py](TorchMLP.py), [TinyTransformer.py](TinyTransformer.py)
- `sample_probs`: [TinyMLP.py](TinyMLP.py)
- `ctx`: [NameSLP.py](NameSLP.py)
- `context`: [TinyMLP.py](TinyMLP.py), [TorchMLP.py](TorchMLP.py), [TinyTransformer.py](TinyTransformer.py)

## Potential Performance Changes

- If `microgpt_lite.py` is added later, try Flash Attention with `F.scaled_dot_product_attention`.
