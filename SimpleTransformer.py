import time, torch
import torch.nn as nn, torch.nn.functional as F
from tinystories_dataset import load_tinystories

# Automatically create all tensors on GPU if available, removing manual device boilerplate
torch.set_default_device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Hyperparameters ---
num_stories  = 5000                                                                               # number of stories loaded from TinyStories dataset
context_size = 32                                                                                 # number of previous tokens used to predict next
embed_dim    = 128                                                                                # token/positional embedding dimension (d_model)
n_heads      = 4                                                                                  # number of attention heads in each transformer layer
ffn_dim      = 256                                                                                # feed-forward network hidden dimension
n_layers     = 3                                                                                  # number of transformer encoder layers
batch_size   = 1536                                                                               # number of samples per training step
lr           = 2e-3                                                                               # learning rate
n_steps      = 1801                                                                               # total training steps
temp         = 0.5                                                                                # temperature for sampling during generation

# --- Data & Tokenization ---
input_ids, target_ids, idx_to_char, token_ids, _ = load_tinystories(num_stories=num_stories, context_size=context_size) # previous chars to predict next

# --- Model ---
torch.manual_seed(42)                                                                             # seed helper for reproducibility
tok_embed   = nn.Embedding(len(idx_to_char), embed_dim)                                           # token embedding lookup layer
pos_embed   = nn.Embedding(context_size, embed_dim)                                               # positional embedding for sequence order
transformer = torch.compile(nn.TransformerEncoder(nn.TransformerEncoderLayer(embed_dim, n_heads, ffn_dim, batch_first=True, dropout=0.), n_layers))
linear      = nn.Linear(embed_dim, len(idx_to_char))                                              # maps hidden state to logits (vocab length)

params    = list(tok_embed.parameters()) + list(pos_embed.parameters()) + list(transformer.parameters()) + list(linear.parameters())
optimizer = torch.optim.Adam(params, lr=lr)                                                       # optimizer replaces manual parameter updates
print(f"params: {sum(p.numel() for p in params):,}")                                              # total trainable parameter count

# --- Train ---
start = time.time()                                                                               # track training duration
for step in range(n_steps):
    batch_idx = torch.randint(0, len(input_ids), (batch_size,))                                   # random batch indices
    x = tok_embed(input_ids[batch_idx]) + pos_embed(torch.arange(context_size))                   # add token and positional embeddings
    loss = F.cross_entropy(linear(transformer(x)[:, -1, :]), target_ids[batch_idx])               # computes softmax and cross-entropy loss automatically
    optimizer.zero_grad(); loss.backward(); optimizer.step()                                      # zero grads, backprop, Adam update

    if step % 200 == 0:
        with torch.no_grad():                                                                     # disable tracking during evaluation
            eval_idx = torch.randint(0, len(input_ids), (4096,), generator=torch.Generator(device=input_ids.device).manual_seed(0)) # fixed seed eval sample to eliminate accuracy wobble
            pred_ids = linear(transformer(tok_embed(input_ids[eval_idx]) + pos_embed(torch.arange(context_size)))[:, -1, :]).argmax(1) # subsample forward & argmax
            print(f"Step {step:4d} | Loss: {loss:.4f} | Acc: {(pred_ids == target_ids[eval_idx]).float().mean():.1%} | {time.time()-start:.1f}s")

print(f"Training time: {time.time() - start:.1f}s")

# --- Generate ---
@torch.no_grad()                                                                                  # disable autograd tracking during inference
def generate(num_chars=200, context_ids=list(token_ids[:context_size]), temp=temp):               # start with true initial context
    output_chars = [idx_to_char[i] for i in context_ids]                                          # decode initial context to string
    for _ in range(num_chars):
        x = tok_embed(torch.tensor([context_ids])) + pos_embed(torch.arange(context_size))        # embed current window
        next_token_probs = torch.softmax(linear(transformer(x)[:, -1, :]) / temp, 1)              # apply temp parameter to pick higher-confidence tokens
        next_token = torch.multinomial(next_token_probs, 1).item()                                # randomly sample from predicted distribution
        context_ids, output_chars = context_ids[1:] + [next_token], output_chars + [idx_to_char[next_token]] # slide window forward and append string
    return ''.join(output_chars)

print("\n--- Generated Story ---")
print(generate())
