from datasets import load_dataset
import itertools, warnings, time
import cupy as cp, numpy as np
warnings.filterwarnings('ignore')

def softmax(x):
    e = cp.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

# --- Data ---
dataset = load_dataset('karpathy/tinystories-gpt4-clean', split='train', streaming=True)
text = ''.join(s['text'] for s in itertools.islice(dataset, 200))

vocab = sorted(set(text))
vocab_size = len(vocab)
char_to_idx = {c: i for i, c in enumerate(vocab)}
idx_to_char = {i: c for i, c in enumerate(vocab)}

context_size = 3
data = [char_to_idx[c] for c in text]
inputs = cp.array([data[i:i+context_size] for i in range(len(data)-context_size)])
targets = cp.array(data[context_size:])
one_hot_targets = cp.eye(vocab_size, dtype=cp.float32)[targets]

# --- Model Architecture ---
cp.random.seed(42)
emb_dim, hidden_size, lr = 10, 100, 0.5
N = len(inputs)

# Clean initialization helper for float32 precision
init_randn = lambda *shape: (cp.random.randn(*shape) * 0.1).astype(cp.float32)

C  = init_randn(vocab_size, emb_dim)
W1 = init_randn(context_size * emb_dim, hidden_size)
b1 = cp.zeros((1, hidden_size), dtype=cp.float32)
W2 = init_randn(hidden_size, vocab_size)
b2 = cp.zeros((1, vocab_size), dtype=cp.float32)

start = time.time()

for epoch in range(2001):
    # --- Forward Pass ---
    emb_cat = C[inputs].reshape(N, -1)          # (N, context_size * emb_dim)
    h       = cp.tanh(emb_cat @ W1 + b1)        # (N, hidden_size)
    logits  = h @ W2 + b2                       # (N, vocab_size)
    probs   = softmax(logits)                   # (N, vocab_size)
    
    # --- Backward Pass ---
    dlogits = (probs - one_hot_targets) / N
    
    dW2     = h.T @ dlogits
    db2     = dlogits.sum(axis=0, keepdims=True)
    
    dh      = dlogits @ W2.T
    dh_pre  = dh * (1.0 - h**2)                 # Derivative of tanh
    
    dW1     = emb_cat.T @ dh_pre
    db1     = dh_pre.sum(axis=0, keepdims=True)
    
    # Re-shape the gradients to properly map the 10-dimensional arrays
    demb_cat = dh_pre @ W1.T
    demb     = demb_cat.reshape(N, context_size, emb_dim)
    dC       = cp.zeros_like(C)
    cp.add.at(dC, inputs.ravel(), demb.reshape(-1, emb_dim))
    
    # --- Update Weights ---
    for param, grad in zip([C, W1, b1, W2, b2], [dC, dW1, db1, dW2, db2]):
        param -= lr * grad
        
    if epoch % 200 == 0:
        print(f"Epoch {epoch:4d} | Acc: {cp.mean(probs.argmax(1) == targets):.1%}")

print(f"Training time: {time.time() - start:.1f}s")

# --- Generate ---
def generate(num_chars=200):
    ctx = list(data[:context_size])
    out = [idx_to_char[i] for i in ctx]
    for _ in range(num_chars):
        h = cp.tanh(C[cp.array([ctx])].reshape(1, -1) @ W1 + b1)
        probs = cp.asnumpy(softmax(h @ W2 + b2)[0])
        ctx = ctx[1:] + [int(np.random.choice(vocab_size, p=probs))]
        out.append(idx_to_char[ctx[-1]])
    return ''.join(out)

print(generate())