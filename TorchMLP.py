import time, torch
import torch.nn as nn, torch.nn.functional as F
from tinystories_dataset import load_tinystories

# Automatically create all tensors on GPU if available, removing manual device boilerplate
torch.set_default_device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Data & Tokenization ---
inputs, targets, vocab, encoded_text = load_tinystories(num_stories=200, context_size=4)# 4 is block_size (previous chars to predict next)
inputs, targets = torch.tensor(inputs), torch.tensor(targets)                           # convert to tensors

# --- Model ---
torch.manual_seed(42)                                                                   # seed helper for reproducibility
embed = nn.Embedding(len(vocab), 256)                                                   # token embedding lookup layer (256 is embed_dim)
model = nn.Sequential(
    nn.Linear(4 * 256, 150), nn.ReLU(),                                                 # maps context (4 * 256) to hidden_dim (150) & applies ReLU
    nn.Linear(150, len(vocab))                                                          # maps hidden state (150) to logits (vocab length)
)
optimizer = torch.optim.SGD(list(embed.parameters()) + list(model.parameters()), lr=0.5)# optimizer replaces manual parameter updates

# --- Train ---
batch_size = 1024                                                                       # number of samples per batch
start = time.time()                                                                     # track training duration
for epoch in range(2001):
    idx = torch.randint(0, len(inputs), (batch_size,))                                  # random batch indices (len(inputs) is total examples)
    emb = embed(inputs[idx]).view(batch_size, -1)                                       # concatenate embeddings for the window
    loss = F.cross_entropy(model(emb), targets[idx])                                    # computes softmax and cross-entropy loss automatically
    optimizer.zero_grad(); loss.backward(); optimizer.step()                            # zero grads, backprop, SGD update

    if epoch % 200 == 0:
        with torch.no_grad():                                                           # disable tracking during evaluation
            preds = model(embed(inputs).view(len(inputs), -1)).argmax(1)                # full dataset (len(inputs)) forward & argmax
            print(f"Epoch {epoch:4d} | Loss: {loss:.4f} | Acc: {(preds == targets).float().mean():.1%}")

print(f"Training time: {time.time() - start:.1f}s")

# --- Generate ---
@torch.no_grad()                                                                        # disable autograd tracking during inference
def generate(num_chars=200, context=list(encoded_text[:4])):                            # start with true initial context (4 chars)
    out = [vocab[i] for i in context]                                                   # decode initial context to string
    for _ in range(num_chars):
        probs = torch.softmax(model(embed(torch.tensor([context])).view(1, -1)), 1)     # fused forward pass
        nxt = torch.multinomial(probs, 1).item()                                        # randomly sample from predicted distribution
        context, out = context[1:] + [nxt], out + [vocab[nxt]]                         # slide window forward and append string
    return ''.join(out)

print(generate())
