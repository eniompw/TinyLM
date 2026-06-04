import time, torch
import torch.nn as nn, torch.nn.functional as F
from tinystories_dataset import load_tinystories

# Automatically create all tensors on GPU if available, removing manual device boilerplate
torch.set_default_device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Data & Tokenization ---
input_ids, target_ids, idx_to_char, token_ids = load_tinystories(num_stories=200, context_size=4)# 4 is block_size (previous chars to predict next)
input_ids, target_ids = torch.tensor(input_ids), torch.tensor(target_ids)                        # convert to tensors

# --- Model ---
torch.manual_seed(42)                                                                             # seed helper for reproducibility
embedding = nn.Embedding(len(idx_to_char), 256)                                                  # token embedding lookup layer (256 is embed_dim)
mlp = nn.Sequential(
    nn.Linear(4 * 256, 150), nn.ReLU(),                                                          # maps context (4 * 256) to hidden_dim (150) & applies ReLU
    nn.Linear(150, len(idx_to_char))                                                             # maps hidden state (150) to logits (vocab length)
)
optimizer = torch.optim.SGD(list(embedding.parameters()) + list(mlp.parameters()), lr=0.5)       # optimizer replaces manual parameter updates

# --- Train ---
batch_size = 1024                                                                                 # number of samples per batch
start = time.time()                                                                               # track training duration
for step in range(2001):
    batch_idx = torch.randint(0, len(input_ids), (batch_size,))                                  # random batch indices (len(input_ids) is total examples)
    x = embedding(input_ids[batch_idx]).view(batch_size, -1)                                     # concatenate embeddings for the window
    loss = F.cross_entropy(mlp(x), target_ids[batch_idx])                                        # computes softmax and cross-entropy loss automatically
    optimizer.zero_grad(); loss.backward(); optimizer.step()                                      # zero grads, backprop, SGD update

    if step % 200 == 0:
        with torch.no_grad():                                                                     # disable tracking during evaluation
            pred_ids = mlp(embedding(input_ids).view(len(input_ids), -1)).argmax(1)              # full dataset forward & argmax
            print(f"Step {step:4d} | Loss: {loss:.4f} | Acc: {(pred_ids == target_ids).float().mean():.1%}")

print(f"Training time: {time.time() - start:.1f}s")

# --- Generate ---
@torch.no_grad()                                                                                  # disable autograd tracking during inference
def generate(num_chars=200, context_ids=list(token_ids[:4])):                                    # start with true initial context (4 chars)
    output_chars = [idx_to_char[i] for i in context_ids]                                         # decode initial context to string
    for _ in range(num_chars):
        next_token_probs = torch.softmax(mlp(embedding(torch.tensor([context_ids])).view(1, -1)), 1)  # fused forward pass
        next_token = torch.multinomial(next_token_probs, 1).item()                               # randomly sample from predicted distribution
        context_ids, output_chars = context_ids[1:] + [next_token], output_chars + [idx_to_char[next_token]]  # slide window forward and append string
    return ''.join(output_chars)

print(generate())
