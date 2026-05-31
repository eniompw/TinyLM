import time, torch
import torch.nn.functional as F
from tinystories_dataset import load_tinystories

# Automatically create all tensors on GPU if available, removing manual device boilerplate
torch.set_default_device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Data & Tokenization ---
inputs, targets, vocab, enc = load_tinystories(num_records=200, context_size=4)         # 4 is context_size (previous chars to predict next)
inputs, targets = torch.tensor(inputs), torch.tensor(targets)                           # convert to tensors

# --- Model ---
torch.manual_seed(42); randn = lambda *s: (torch.randn(*s) * 0.1).requires_grad_()      # seed & init helper (now requires_grad for autograd)
C  = randn(len(vocab), 256)                                                             # token embedding lookup matrix (256 is emb_dim)
W1 = randn(4 * 256, 150)                                                                # weights mapping context (4 * 256) to hidden_size (150)
W2 = randn(150, len(vocab))                                                             # weights mapping hidden state (150) to logits (vocab length)

# --- Train ---
lr, batch_size = 0.5, 1024                                                              # learning rate, number of samples per batch
start = time.time()                                                                     # track training duration
for epoch in range(2001):
    idx = torch.randint(0, len(inputs), (batch_size,))                                  # random batch indices (len(inputs) is total examples)
    X, Y = inputs[idx], targets[idx]                                                    # fetch random mini-batch (inputs and labels)

    # Forward pass
    emb = C[X].view(batch_size, -1)                                                     # concatenate embeddings for the window
    h   = torch.relu(emb @ W1)                                                          # apply ReLU non-linearity to hidden state
    loss = F.cross_entropy(h @ W2, Y)                                                   # computes softmax and cross-entropy loss automatically

    # Backward pass
    loss.backward()                                                                     # autograd automatically computes gradients for C, W1, W2

    with torch.no_grad():                                                               # disable gradient tracking for parameter updates
        for p in (C, W1, W2): 
            p -= lr * p.grad                                                            # standard SGD parameter update
            p.grad = None                                                               # zero out gradients for the next iteration

    if epoch % 200 == 0:
        with torch.no_grad():                                                           # disable tracking during evaluation
            preds = (torch.relu(C[inputs].view(len(inputs), -1) @ W1) @ W2).argmax(1)   # full dataset (len(inputs)) forward & argmax
            print(f"Epoch {epoch:4d} | Acc: {(preds == targets).float().mean():.1%}")

print(f"Training time: {time.time() - start:.1f}s")

# --- Generate ---
@torch.no_grad()                                                                        # disable autograd tracking during inference
def generate(num_chars=200, ctx=list(enc[:4])):                                         # start with true initial context (4 chars)
    out = [vocab[i] for i in ctx]                                                       # decode initial context to string
    for _ in range(num_chars):
        p = torch.softmax(torch.relu(C[torch.tensor([ctx])].view(1, -1) @ W1) @ W2, 1)  # fused forward pass
        next_id = torch.multinomial(p, 1).item()                                        # randomly sample from predicted distribution
        ctx, out = ctx[1:] + [next_id], out + [vocab[next_id]]                          # slide window forward and append string
    return ''.join(out)

print(generate())