from datasets import load_dataset
import itertools, warnings, numpy as np
warnings.filterwarnings('ignore')

def softmax(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

# --- Data ---
dataset = load_dataset('karpathy/tinystories-gpt4-clean', split='train', streaming=True)
text    = ''.join(s['text'] for s in itertools.islice(dataset, 100))

vocab       = sorted(set(text))
vocab_size  = len(vocab)
char_to_idx = {c: i for i, c in enumerate(vocab)}
idx_to_char = {i: c for i, c in enumerate(vocab)}
data        = [char_to_idx[c] for c in text]

# build bigram training pairs: one-hot input → one-hot target
X = np.eye(vocab_size)[data[:-1]]    # current char, one-hot
y = np.eye(vocab_size)[data[1:]]    # next char, one-hot

# --- Model (identical structure to SLP) ---
np.random.seed(42)
W = np.random.randn(vocab_size, vocab_size) * 0.1
b = np.zeros((1, vocab_size))
learning_rate = 0.1

for epoch in range(1000):
    probs    = softmax(X @ W + b)
    dlogits  = (probs - y) / len(X)
    W       -= learning_rate * X.T @ dlogits
    b       -= learning_rate * dlogits.sum(0, keepdims=True)
    if epoch % 100 == 0:
        acc = np.mean(probs.argmax(1) == y.argmax(1))
        print(f"Epoch {epoch:4d} | Acc: {acc:.1%}")

# --- Generate ---
def generate(num_chars=200):
    idx = 0
    out = [idx_to_char[idx]]
    for _ in range(num_chars):
        x     = np.eye(vocab_size)[[idx]]
        probs = softmax(x @ W + b)[0]
        idx   = np.random.choice(vocab_size, p=probs)
        out.append(idx_to_char[idx])
    return ''.join(out)

print(generate())
