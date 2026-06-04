import time, torch
import torch.nn as nn, torch.nn.functional as F
from tinystories_dataset import load_tinystories

# Automatically create all tensors on GPU if available, removing manual device boilerplate
torch.set_default_device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Data & Tokenization ---
context_size = 8
input_ids, target_ids, idx_to_char, token_ids = load_tinystories(num_stories=1000, context_size=context_size) # previous chars to predict next
input_ids, target_ids = torch.tensor(input_ids), torch.tensor(target_ids)                        # convert to tensors

# --- Model ---
torch.manual_seed(42)                                                                             # seed helper for reproducibility
tok_embed = nn.Embedding(len(idx_to_char), 256)                                                  # token embedding lookup layer (256 is embed_dim)
pos_embed = nn.Embedding(context_size, 256)                                                      # positional embedding for sequence order
transformer = torch.compile(nn.TransformerEncoder(nn.TransformerEncoderLayer(256, 4, 1024, batch_first=True, dropout=0., norm_first=True), 2))
linear = nn.Linear(256, len(idx_to_char))                                                        # maps hidden state to logits (vocab length)

params = list(tok_embed.parameters()) + list(pos_embed.parameters()) + list(transformer.parameters()) + list(linear.parameters())
optimizer = torch.optim.AdamW(params, lr=1e-3, fused=True)                                       # optimizer replaces manual parameter updates
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 2000, eta_min=1e-4)            # smoothly decays learning rate

# --- Train ---
batch_size = 1024                                                                                 # number of samples per batch
start = time.time()                                                                               # track training duration
for step in range(2001):
    batch_idx = torch.randint(0, len(input_ids), (batch_size,))                                  # random batch indices (len(input_ids) is total examples)
    batch_x, batch_y = input_ids[batch_idx], target_ids[batch_idx]                               # fetch random mini-batch (inputs and labels)

    with torch.autocast('cuda', dtype=torch.float16):                                            # float16 mixed precision for speed
        x = tok_embed(batch_x) + pos_embed(torch.arange(context_size))                          # add token and positional embeddings
        loss = F.cross_entropy(linear(transformer(x)[:, -1, :]), batch_y)                        # computes softmax and cross-entropy loss automatically

    optimizer.zero_grad(); loss.backward(); optimizer.step(); scheduler.step()                   # zero grads, backprop, AdamW update, and lr decay

    if step % 200 == 0:
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):                       # disable tracking during evaluation
            eval_idx = torch.randint(0, len(input_ids), (4096,))                                 # subset evaluation to prevent GPU OOM
            x_eval = tok_embed(input_ids[eval_idx]) + pos_embed(torch.arange(context_size))     # embed dataset subset
            pred_ids = linear(transformer(x_eval)[:, -1, :]).argmax(1)                          # dataset subset forward & argmax
            print(f"Step {step:4d} | Loss: {loss:.4f} | Acc: {(pred_ids == target_ids[eval_idx]).float().mean():.1%}")

print(f"Training time: {time.time() - start:.1f}s")

# --- Generate ---
@torch.no_grad()                                                                                  # disable autograd tracking during inference
def generate(num_chars=200, context_ids=list(token_ids[:context_size])):                         # start with true initial context
    output_chars = [idx_to_char[i] for i in context_ids]                                         # decode initial context to string
    for _ in range(num_chars):
        x = tok_embed(torch.tensor([context_ids])) + pos_embed(torch.arange(context_size))      # embed current window
        next_token_probs = torch.softmax(linear(transformer(x)[:, -1, :]) / 0.7, 1)             # apply temp 0.7 to pick higher-confidence tokens
        next_token = torch.multinomial(next_token_probs, 1).item()                               # randomly sample from predicted distribution
        context_ids, output_chars = context_ids[1:] + [next_token], output_chars + [idx_to_char[next_token]] # slide window forward and append string
    return ''.join(output_chars)

print(generate())
