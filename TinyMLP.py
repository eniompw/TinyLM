from datasets import load_dataset
import itertools, warnings, cupy as cp, numpy as np, time
warnings.filterwarnings('ignore')

def softmax(x):
    e = cp.exp(x - x.max(axis=1, keepdims=True))   # subtract max for numerical stability
    return e / e.sum(axis=1, keepdims=True)

# --- Data ---
dataset = load_dataset('karpathy/tinystories-gpt4-clean', split='train', streaming=True)
text    = ''.join(s['text'] for s in itertools.islice(dataset, 200))  # first 200 stories

vocab       = sorted(set(text))                                 # unique characters
vocab_size  = len(vocab)
char_to_idx = {c: i for i, c in enumerate(vocab)}               # char → integer
idx_to_char = {i: c for i, c in enumerate(vocab)}               # integer → char

context_size = 3                                                # context window
data    = [char_to_idx[c] for c in text]
inputs  = cp.array([data[i:i+context_size] for i in range(len(data)-context_size)])  # (N, context)
targets = cp.array(data[context_size:])                         # next char to predict

# --- Model Architecture (MLP) ---
cp.random.seed(42)
emb_dim = 10
hidden_size = 100

# 1. Embedding layer: maps vocab to dense vectors
C  = cp.random.randn(vocab_size, emb_dim) * 0.1 

# 2. Hidden layer: maps concatenated embeddings to hidden space
W1 = cp.random.randn(context_size * emb_dim, hidden_size) * 0.1
b1 = cp.zeros((1, hidden_size))

# 3. Output layer: maps hidden space back to vocabulary probabilities
W2 = cp.random.randn(hidden_size, vocab_size) * 0.1
b2 = cp.zeros((1, vocab_size))

# We use a slightly higher learning rate since the network is deeper
learning_rate = 0.5 

start = time.time()
N = len(inputs)
one_hot_targets = cp.eye(vocab_size)[targets]

for epoch in range(2001):
    # --- Forward Pass ---
    emb = C[inputs]                                # (N, context_size, emb_dim)
    emb_cat = emb.reshape(N, -1)                   # Flatten to (N, context_size * emb_dim)
    
    h_preact = emb_cat @ W1 + b1                   # Linear transformation
    h = cp.tanh(h_preact)                          # Non-linear activation (N, hidden_size)
    
    logits = h @ W2 + b2                           # Output projection (N, vocab_size)
    probs = softmax(logits)                        # (N, vocab_size)
    
    # --- Backward Pass ---
    dlogits = (probs - one_hot_targets) / N        # Cross-entropy + softmax gradient
    
    # Gradients for W2, b2
    dW2 = h.T @ dlogits
    db2 = dlogits.sum(axis=0, keepdims=True)
    
    # Gradients for W1, b1
    dh = dlogits @ W2.T
    dh_preact = dh * (1.0 - h**2)                  # Derivative of tanh
    dW1 = emb_cat.T @ dh_preact
    db1 = dh_preact.sum(axis=0, keepdims=True)
    
    # Gradients for Embedding Layer (C)
    demb_cat = dh_preact @ W1.T
    demb = demb_cat.reshape(N, context_size, emb_dim)
    dC = cp.zeros_like(C)
    for i in range(context_size):
        cp.add.at(dC, inputs[:, i], demb[:, i, :]) # Accumulate gradients for each context position
    
    # --- Weight Update ---
    C  -= learning_rate * dC
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    
    if epoch % 200 == 0:
        acc = cp.mean(probs.argmax(1) == targets)
        print(f"Epoch {epoch:4d} | Acc: {acc:.1%}")

print(f"Training time: {time.time() - start:.1f}s")

# --- Generate ---
def generate(num_chars=200):
    context = list(data[:context_size])                    # seed with first chars
    out = [idx_to_char[i] for i in context]
    for _ in range(num_chars):
        # Forward pass for single step
        emb = C[cp.array([context])]                       # Get embeddings
        emb_cat = emb.reshape(1, -1)                       # Flatten
        h = cp.tanh(emb_cat @ W1 + b1)                     # Hidden layer
        logits = h @ W2 + b2                               # Output layer
        
        probs = cp.asnumpy(softmax(logits)[0])             # move to CPU
        next_idx = int(np.random.choice(vocab_size, p=probs)) 
        context = context[1:] + [next_idx]                 # slide window
        out.append(idx_to_char[next_idx])
    return ''.join(out)

print(generate())