import time, torch
import torch.nn as nn, torch.nn.functional as F
from tinystories_dataset import load_tinystories

# Automatically create all tensors on GPU if available, removing manual device boilerplate
torch.set_default_device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Hyperparameters ---
num_stories  = 5000                                                                               # number of stories loaded from TinyStories dataset
context_size = 16                                                                                 # number of previous tokens used to predict next
embed_dim    = 256                                                                                # token embedding dimension
hidden_dim   = 512                                                                                # MLP hidden layer dimension
batch_size   = 2048                                                                               # number of samples per training step
lr           = 1e-3                                                                               # learning rate
n_steps      = 8001                                                                               # total training steps
temp         = 1.0                                                                                # temperature for sampling during generation

# --- Data & Tokenization ---
input_ids, target_ids, idx_to_char, token_ids, vocab_size = load_tinystories(num_stories=num_stories, context_size=context_size)

# --- Eval subset (fixed seed to always measure same rows) ---
eval_rng = torch.Generator(device='cuda' if torch.cuda.is_available() else 'cpu')
eval_rng.manual_seed(0)
eval_idx = torch.randint(0, len(input_ids), (4096,), generator=eval_rng)                        # fixed 4096-sample subset eliminates OOM and wobble

# --- Model ---
torch.manual_seed(42)                                                                             # seed helper for reproducibility
embedding = nn.Embedding(vocab_size, embed_dim)                                                   # token embedding lookup layer
mlp       = nn.Sequential(
    nn.Linear(context_size * embed_dim, hidden_dim), nn.ReLU(),                                  # maps context to hidden_dim & applies ReLU
    nn.Linear(hidden_dim, vocab_size)                                                             # maps hidden state to logits (vocab length)
)

params    = list(embedding.parameters()) + list(mlp.parameters())
optimizer = torch.optim.AdamW(params, lr=lr)                                                     # AdamW optimizer

# --- Train ---
start = time.time()                                                                               # track training duration
for step in range(n_steps):
    batch_idx = torch.randint(0, len(input_ids), (batch_size,))                                  # random batch indices
    x = embedding(input_ids[batch_idx]).view(batch_size, -1)                                     # concatenate embeddings for the window
    loss = F.cross_entropy(mlp(x), target_ids[batch_idx])                                        # computes softmax and cross-entropy loss automatically
    optimizer.zero_grad(); loss.backward(); optimizer.step()                                      # zero grads, backprop, AdamW update

    if step % 200 == 0:
        with torch.no_grad():
            pred_ids = mlp(embedding(input_ids[eval_idx]).view(4096, -1)).argmax(1)              # fixed subset eval
            print(f"Step {step:4d} | Loss: {loss:.4f} | Acc: {(pred_ids == target_ids[eval_idx]).float().mean():.1%} | {time.time()-start:.1f}s")

print(f"Training time: {time.time() - start:.1f}s")

# --- Generate ---
@torch.no_grad()                                                                                  # disable autograd tracking during inference
def generate(num_chars=200, context_ids=list(token_ids[:context_size]), temp=temp):              # start with true initial context
    output_chars = [idx_to_char[i] for i in context_ids]                                         # decode initial context to string
    for _ in range(num_chars):
        x = embedding(torch.tensor([context_ids])).view(1, -1)                                   # embed current window
        next_token_probs = torch.softmax(mlp(x) / temp, 1)                                       # apply temp parameter to pick higher-confidence tokens
        next_token = torch.multinomial(next_token_probs, 1).item()                               # randomly sample from predicted distribution
        context_ids, output_chars = context_ids[1:] + [next_token], output_chars + [idx_to_char[next_token]] # slide window forward and append string
    return ''.join(output_chars)

print("\n--- Generated Story ---")
print(generate())
