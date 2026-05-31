from datasets import load_dataset
import itertools, warnings, time, cupy as cp, numpy as np
warnings.filterwarnings('ignore')

def softmax(logits):                                            # converts raw network outputs to probabilities
    e = cp.exp(logits - logits.max(axis=1, keepdims=True))      # shift to prevent float overflow
    return e / e.sum(axis=1, keepdims=True)                     # normalize so all probabilities sum to 1

# --- Data & Tokenization ---
dataset = load_dataset('karpathy/tinystories-gpt4-clean', split='train', streaming=True)
text = ''.join(s['text'] for s in itertools.islice(dataset, 200))

vocab = sorted(set(text))                                       # ordered list of unique characters
vocab_size = len(vocab)                                         # total count of unique characters
char_to_id = {c: i for i, c in enumerate(vocab)}                # dictionary mapping char to integer id
encoded = [char_to_id[c] for c in text]                         # map entire text to integer sequence

# --- Dataset Prep ---
context_size = 4                                                # number of previous chars used to predict next
inputs  = cp.array([encoded[i:i+context_size] for i in range(len(encoded)-context_size)]) # sliding windows
targets = cp.array(encoded[context_size:])                                                # next char to predict
N = len(inputs)                                                 # total number of training examples

# --- Model ---
emb_dim, hidden_size = 256, 150                                 # embedding dimensions, hidden layer neurons
cp.random.seed(42); randn = lambda *s: (cp.random.randn(*s) * 0.1).astype(cp.float32) # seed & normal init helper
C  = randn(vocab_size, emb_dim)                                 # token embedding lookup matrix
W1 = randn(context_size * emb_dim, hidden_size)                 # weights mapping context to hidden state
W2 = randn(hidden_size, vocab_size)                             # weights mapping hidden state to logits

# --- Train ---
lr, batch_size = 0.5, 1024                                      # learning rate, number of samples per batch
start = time.time()                                             # track training duration
for epoch in range(2001):
    idx = cp.random.randint(0, N, size=batch_size)              # random array of indices for batch
    X, Y = inputs[idx], targets[idx]                            # fetch random mini-batch (inputs and labels)

    # Forward pass
    emb = C[X].reshape(batch_size, -1)                          # concatenate embeddings for the window
    h   = cp.maximum(0, emb @ W1)                               # apply ReLU non-linearity to hidden state
    probs = softmax(h @ W2)                                     # get probability distribution over vocab

    # Backward pass
    probs[cp.arange(batch_size), Y] -= 1                        # in-place CE gradient: (probs - 1) for true labels
    probs /= batch_size                                         # average loss over batch (probs is now dlogits)

    dW2, dh = h.T @ probs, (probs @ W2.T) * (h > 0)             # gradients for output weights & hidden state (ReLU backprop)
    dW1 = emb.T @ dh                                            # gradient for hidden weights

    dC = cp.zeros_like(C)                                       # gradient accumulator for embeddings
    cp.add.at(dC, X.ravel(), (dh @ W1.T).reshape(-1, emb_dim))  # safely accumulate grads for repeated tokens

    for p, g in zip((C, W1, W2), (dC, dW1, dW2)):
        p -= lr * g                                             # standard SGD parameter update

    if epoch % 200 == 0:
        logits = cp.maximum(0, C[inputs].reshape(N, -1) @ W1) @ W2  # full dataset forward pass
        preds = logits.argmax(1)                                    # mathematical shortcut: argmax directly on logits
        print(f"Epoch {epoch:4d} | Acc: {cp.mean(preds == targets):.1%}")

print(f"Training time: {time.time() - start:.1f}s")

# --- Generate ---
def generate(num_chars=200):
    ctx = list(encoded[:context_size])                          # start with true initial context from text
    out = [vocab[i] for i in ctx]                               # decode initial context to string
    for _ in range(num_chars):
        p = cp.asnumpy(softmax(cp.maximum(0, C[cp.array([ctx])].reshape(1, -1) @ W1) @ W2)[0]) # fused forward pass
        next_id = int(np.random.choice(vocab_size, p=p))        # randomly sample from predicted distribution
        ctx = ctx[1:] + [next_id]                               # slide context window forward by one token
        out.append(vocab[next_id])
    return ''.join(out)

print(generate())