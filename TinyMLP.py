from datasets import load_dataset
import itertools, warnings, time
import cupy as cp, numpy as np
warnings.filterwarnings('ignore')

if cp.cuda.runtime.getDeviceCount() == 0:
    raise RuntimeError("TinyMLP.py assumes a CUDA GPU (for example, Colab T4). Enable GPU runtime and retry.")

def softmax(x):
    e = cp.exp(x - x.max(axis=1, keepdims=True))                # shift for numerical stability
    return e / e.sum(axis=1, keepdims=True)                     # return probabilities

# --- Data ---
dataset = load_dataset('karpathy/tinystories-gpt4-clean', split='train', streaming=True)
text = ''.join(s['text'] for s in itertools.islice(dataset, 200)) # get first 200 stories

vocab = sorted(set(text))                                       # unique chars
vocab_size = len(vocab)
char_to_idx = {c: i for i, c in enumerate(vocab)}               # encoder
idx_to_char = {i: c for i, c in enumerate(vocab)}               # decoder

context_size = 4                                                # optimal context window
data = [char_to_idx[c] for c in text]                           # encode data
inputs = cp.array([data[i:i+context_size] for i in range(len(data)-context_size)])
targets = cp.array(data[context_size:])                         # targets are next char
one_hot_targets = cp.eye(vocab_size, dtype=cp.float32)[targets] # shape (N, vocab_size)

# --- Model ---
cp.random.seed(42)                                              
emb_dim, hidden_size = 10, 150                                  # optimal network dimensions
lr, batch_size = 0.5, 1024                                      # training hyperparameters
N = len(inputs)                                                 # total samples

init_randn = lambda *shape: (cp.random.randn(*shape) * 0.1).astype(cp.float32)

C  = init_randn(vocab_size, emb_dim)                            # token embeddings
W1 = init_randn(context_size * emb_dim, hidden_size)            # hidden weights
b1 = cp.zeros((1, hidden_size), dtype=cp.float32)               # hidden biases
W2 = init_randn(hidden_size, vocab_size)                        # output weights
b2 = cp.zeros((1, vocab_size), dtype=cp.float32)                # output biases

start = time.time()                                             

for epoch in range(2001):
    # --- Mini-batching ---
    idx = cp.random.randint(0, N, size=batch_size)              # random sample indices
    X_batch = inputs[idx]                                       # slice inputs
    Y_batch = one_hot_targets[idx]                              # slice targets
    
    # --- Forward pass ---
    emb_cat = C[X_batch].reshape(batch_size, -1)                # flat embeddings: (B, block_size * emb_dim)
    h       = cp.maximum(0, emb_cat @ W1 + b1)                  # hidden state: ReLU non-linearity
    logits  = h @ W2 + b2                                       # unnormalized log probabilities
    probs   = softmax(logits)                                   # softmax probabilities
    
    # --- Backward pass ---
    dlogits = (probs - Y_batch) / batch_size                    # CE loss gradient 
    
    dW2     = h.T @ dlogits                                     # grad W2
    db2     = dlogits.sum(axis=0, keepdims=True)                # grad b2
    
    dh      = dlogits @ W2.T                                    # backprop into hidden
    dh_pre  = dh * (h > 0)                                      # backprop through ReLU
    
    dW1     = emb_cat.T @ dh_pre                                # grad W1
    db1     = dh_pre.sum(axis=0, keepdims=True)                 # grad b1
    
    demb_cat = dh_pre @ W1.T                                    # backprop into embeddings
    demb     = demb_cat.reshape(batch_size, context_size, emb_dim) # reshape back to (B, block, emb_dim)
    dC       = cp.zeros_like(C)                                 
    cp.add.at(dC, X_batch.ravel(), demb.reshape(-1, emb_dim))   # accumulate gradients into C
    
    # --- Update ---
    for param, grad in zip([C, W1, b1, W2, b2], [dC, dW1, db1, dW2, db2]): 
        param -= lr * grad                                      # SGD step
        
    if epoch % 200 == 0:                                        # full-dataset accuracy check
        test_h = cp.maximum(0, C[inputs].reshape(N, -1) @ W1 + b1)
        test_probs = softmax(test_h @ W2 + b2)
        print(f"Epoch {epoch:4d} | Acc: {cp.mean(test_probs.argmax(1) == targets):.1%}")

print(f"Training time: {time.time() - start:.1f}s")             

# --- Generate ---
def generate(num_chars=200):                                    # autoregressive generation
    ctx = list(data[:context_size])                             # seed context
    out = [idx_to_char[i] for i in ctx]                         
    for _ in range(num_chars):
        h = cp.maximum(0, C[cp.array([ctx])].reshape(1, -1) @ W1 + b1) 
        probs = cp.asnumpy(softmax(h @ W2 + b2)[0])             # get char probabilities
        ctx = ctx[1:] + [int(np.random.choice(vocab_size, p=probs))] # sample and slide window
        out.append(idx_to_char[ctx[-1]])                        
    return ''.join(out)                                         

print(generate())