from datasets import load_dataset
import itertools, warnings, time
import cupy as cp, numpy as np
warnings.filterwarnings('ignore')

def softmax(x):
    e = cp.exp(x - x.max(axis=1, keepdims=True))   # shift for numerical stability
    return e / e.sum(axis=1, keepdims=True)          # return probabilities

# --- Data ---
dataset = load_dataset('karpathy/tinystories-gpt4-clean', split='train', streaming=True)
text = ''.join(s['text'] for s in itertools.islice(dataset, 200))
vocab = sorted(set(text))
vocab_size = len(vocab)
char_to_idx = {c: i for i, c in enumerate(vocab)}
idx_to_char = {i: c for i, c in enumerate(vocab)}

context_size, emb_dim, hidden_size = 4, 256, 150     # optimal network dimensions
lr, batch_size = 0.5, 1024                           # training hyperparameters
data = [char_to_idx[c] for c in text]
inputs  = cp.array([data[i:i+context_size] for i in range(len(data)-context_size)])
targets = cp.array(data[context_size:])              # targets are next char
N = len(inputs)

# --- Model ---
cp.random.seed(42)
r  = lambda *s: (cp.random.randn(*s) * 0.1).astype(cp.float32)
C  = r(vocab_size, emb_dim)                          # token embeddings
W1 = r(context_size * emb_dim, hidden_size)          # hidden weights
W2 = r(hidden_size, vocab_size)                      # output weights

# --- Train ---
start = time.time()
for epoch in range(2001):
    idx = cp.random.randint(0, N, size=batch_size)
    X, Y = inputs[idx], targets[idx]                 # random mini-batch

    emb = C[X].reshape(batch_size, -1)               # flat embeddings
    h   = cp.maximum(0, emb @ W1)                    # hidden state: ReLU
    probs = softmax(h @ W2)                          # softmax probabilities

    dlogits = probs.copy()                           # CE loss gradient
    dlogits[cp.arange(batch_size), Y] -= 1
    dlogits /= batch_size

    dW2 = h.T @ dlogits                              # grad W2
    dh  = (dlogits @ W2.T) * (h > 0)                # backprop through ReLU
    dW1 = emb.T @ dh                                 # grad W1
    dC  = cp.zeros_like(C)
    cp.add.at(dC, X.ravel(), (dh @ W1.T).reshape(-1, emb_dim))  # grad embeddings

    for p, g in zip([C, W1, W2], [dC, dW1, dW2]):
        p -= lr * g                                  # SGD step

    if epoch % 200 == 0:
        h_all = cp.maximum(0, C[inputs].reshape(N, -1) @ W1)
        acc = cp.mean(softmax(h_all @ W2).argmax(1) == targets)
        print(f"Epoch {epoch:4d} | Acc: {acc:.1%}")

print(f"Training time: {time.time() - start:.1f}s")

# --- Generate ---
def generate(num_chars=200):
    ctx = list(data[:context_size])                  # seed context
    out = [idx_to_char[i] for i in ctx]
    for _ in range(num_chars):
        h = cp.maximum(0, C[cp.array([ctx])].reshape(1, -1) @ W1)
        p = cp.asnumpy(softmax(h @ W2)[0])           # get char probabilities
        ctx = ctx[1:] + [int(np.random.choice(vocab_size, p=p))]  # sample + slide
        out.append(idx_to_char[ctx[-1]])
    return ''.join(out)

print(generate())