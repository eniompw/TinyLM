from datasets import load_dataset
import itertools, warnings, numpy as np
warnings.filterwarnings('ignore')

def softmax(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))   # subtract max for numerical stability
    return e / e.sum(axis=1, keepdims=True)

# --- Data ---
dataset = load_dataset('karpathy/tinystories-gpt4-clean', split='train', streaming=True)
text    = ''.join(s['text'] for s in itertools.islice(dataset, 20))  # first 20 stories

vocab       = sorted(set(text))                             # unique characters
vocab_size  = len(vocab)
char_to_idx = {c: i for i, c in enumerate(vocab)}          # char → integer
idx_to_char = {i: c for i, c in enumerate(vocab)}          # integer → char

inputs  = np.array([char_to_idx[c] for c in text[:-1]])    # current char as integer index
targets = np.array([char_to_idx[c] for c in text[1:]])     # next char as integer index

# --- Model ---
np.random.seed(42)
W = np.random.randn(vocab_size, vocab_size) * 0.1           # weights: vocab → vocab
b = np.zeros((1, vocab_size))
learning_rate = 0.1

for epoch in range(1000):
    logits  = W[inputs] + b                                 # row lookup replaces matmul
    probs   = softmax(logits)                               # forward pass
    dlogits = (probs - np.eye(vocab_size)[targets]) / len(inputs)  # cross-entropy + softmax gradient
    np.add.at(W, inputs, -(learning_rate * dlogits))        # sparse weight update
    b      -= learning_rate * dlogits.sum(0, keepdims=True) # update biases
    if epoch % 100 == 0:
        acc = np.mean(probs.argmax(1) == targets)
        print(f"Epoch {epoch:4d} | Acc: {acc:.1%}")

# --- Generate ---
def generate(num_chars=200):
    idx = 0
    out = [idx_to_char[idx]]
    for _ in range(num_chars):
        probs = softmax((W[idx] + b).reshape(1, -1))[0]    # predict next char probabilities
        idx   = np.random.choice(vocab_size, p=probs)      # sample from distribution
        out.append(idx_to_char[idx])
    return ''.join(out)

print(generate())