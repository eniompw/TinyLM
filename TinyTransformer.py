import time, torch
import torch.nn as nn, torch.nn.functional as F
from tinystories_dataset import load_tinystories

# Automatically create all tensors on GPU if available, removing manual device boilerplate
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_default_device(device)

# --- Hyperparameters (Max Coherence: Context=32, Temp=0.5) ---
context_size = 32                                                                                 # INCREASED: 16 -> 32. Fixes pronoun/gender swaps by keeping names in memory longer.
embed_dim    = 256                                                                                # token/positional embedding dimension (d_model)
n_heads      = 4                                                                                  # number of attention heads in each transformer layer
ffn_dim      = 1024                                                                               # feed-forward network hidden dimension
n_layers     = 3                                                                                  # 3 layers is the sweet spot for depth vs speed
batch_size   = 1536                                                                               # DECREASED: 2048 -> 1536 to offset the math cost of context=32 and stay under 2 mins.
lr           = 2e-3                                                                               # high learning rate to match the larger batch
n_steps      = 1800                                                                               # Adjusted to fit the 2-min budget with larger context.

# --- Data & Tokenization ---
input_ids, target_ids, idx_to_char, token_ids = load_tinystories(num_stories=5000, context_size=context_size)
input_ids, target_ids = torch.tensor(input_ids), torch.tensor(target_ids)

# --- Model ---
torch.manual_seed(0)
eval_rng = torch.Generator(device=device).manual_seed(0)
tok_embed = nn.Embedding(len(idx_to_char), embed_dim)
pos_embed = nn.Embedding(context_size, embed_dim)
transformer = torch.compile(nn.TransformerEncoder(nn.TransformerEncoderLayer(embed_dim, n_heads, ffn_dim, batch_first=True, dropout=0., norm_first=True), n_layers))
linear = nn.Linear(embed_dim, len(idx_to_char))

params = list(tok_embed.parameters()) + list(pos_embed.parameters()) + list(transformer.parameters()) + list(linear.parameters())
print(f"params: {sum(p.numel() for p in params):,}")

# weight_decay=0.01 prevents overfitting and encourages smooth grammar.
optimizer = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.95), weight_decay=0.01, fused=True)

# Scheduler synced to n_steps
scheduler, start = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_steps, eta_min=1e-4), time.time()

# --- Train ---
for step in range(n_steps):
    batch_idx = torch.randint(0, len(input_ids), (batch_size,))
    batch_x, batch_y = input_ids[batch_idx], target_ids[batch_idx]

    with torch.autocast('cuda', dtype=torch.float16):
        x = tok_embed(batch_x) + pos_embed(torch.arange(context_size))
        loss = F.cross_entropy(linear(transformer(x)[:, -1, :]), batch_y)

    optimizer.zero_grad(); loss.backward(); optimizer.step(); scheduler.step()

    if step % 200 == 0:
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
            eval_idx = torch.randint(0, len(input_ids), (4096,), generator=eval_rng)
            x_eval = tok_embed(input_ids[eval_idx]) + pos_embed(torch.arange(context_size))
            pred_ids = linear(transformer(x_eval)[:, -1, :]).argmax(1)
            print(f"Step {step:4d} | Loss: {loss:.4f} | Acc: {(pred_ids == target_ids[eval_idx]).float().mean():.1%} | {time.time()-start:.1f}s")

print(f"Training time: {time.time() - start:.1f}s")

# --- Generate ---
@torch.no_grad()
def generate(num_chars=200, context_ids=list(token_ids[:context_size])):
    output_chars = [idx_to_char[i] for i in context_ids]
    for _ in range(num_chars):
        x = tok_embed(torch.tensor([context_ids])) + pos_embed(torch.arange(context_size))
        # DECREASED temp 0.7 -> 0.5. Makes the model more confident, eliminating fake words like "throbe".
        next_token_probs = torch.softmax(linear(transformer(x)[:, -1, :]) / 0.5, 1)
        next_token = torch.multinomial(next_token_probs, 1).item()
        context_ids, output_chars = context_ids[1:] + [next_token], output_chars + [idx_to_char[next_token]]
    return ''.join(output_chars)

print("\n--- Generated Story ---")
print(generate())