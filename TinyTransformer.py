import time, torch
import torch.nn as nn, torch.nn.functional as F
from tinystories_dataset import load_tinystories

# Automatically create all tensors on GPU if available, removing manual device boilerplate
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_default_device(device)

# --- Hyperparameters (Max Coherence: Context=32, Temp=0.5) ---
num_stories  = 5000                                                                               # EXPANDED: 1000 -> 5000 stories for rich vocabulary and sentence variety
context_size = 32                                                                                 # INCREASED: 16 -> 32. Fixes pronoun/gender swaps by keeping names in memory longer.
embed_dim    = 256                                                                                # token/positional embedding dimension (d_model)
n_heads      = 4                                                                                  # number of attention heads in each transformer layer
ffn_dim      = 1024                                                                               # feed-forward network hidden dimension
n_layers     = 3                                                                                  # 3 layers is the sweet spot for depth vs speed
batch_size   = 1536                                                                               # DECREASED: 2048 -> 1536 to offset the math cost of context=32 and stay under 2 mins.
lr           = 2e-3                                                                               # high learning rate to match the larger batch
n_steps      = 1801                                                                               # Adjusted to fit the 2-min budget with larger context.
temp         = 0.5                                                                                # DECREASED: 0.7 -> 0.5. Makes the model more confident, eliminating fake words like "throbe".

# --- Data & Tokenization ---
input_ids, target_ids, idx_to_char, token_ids = load_tinystories(num_stories=num_stories, context_size=context_size) # previous chars to predict next
input_ids, target_ids = torch.tensor(input_ids), torch.tensor(target_ids)                         # convert to tensors

# --- Model ---
torch.manual_seed(0)                                                                              # seed helper for reproducibility
eval_rng = torch.Generator(device=device).manual_seed(0)                                          # FIXED: Dedicated GPU generator to eliminate accuracy noise!
tok_embed = nn.Embedding(len(idx_to_char), embed_dim)                                             # token embedding lookup layer
pos_embed = nn.Embedding(context_size, embed_dim)                                                 # positional embedding for sequence order
transformer = torch.compile(nn.TransformerEncoder(nn.TransformerEncoderLayer(embed_dim, n_heads, ffn_dim, batch_first=True, dropout=0., norm_first=True), n_layers))
linear = nn.Linear(embed_dim, len(idx_to_char))                                                   # maps hidden state to logits (vocab length)

params = list(tok_embed.parameters()) + list(pos_embed.parameters()) + list(transformer.parameters()) + list(linear.parameters())
print(f"params: {sum(p.numel() for p in params):,}")

# weight_decay=0.01 prevents overfitting and encourages smooth grammar.
optimizer = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.95), weight_decay=0.01, fused=True)    # optimizer replaces manual parameter updates

# Scheduler synced to n_steps
scheduler, start = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_steps, eta_min=1e-4), time.time() # smoothly decays learning rate

# --- Train ---
for step in range(n_steps):
    batch_idx = torch.randint(0, len(input_ids), (batch_size,))                                   # random batch indices (len(input_ids) is total examples)
    batch_x, batch_y = input_ids[batch_idx], target_ids[batch_idx]                                # fetch random mini-batch (inputs and labels)

    with torch.autocast('cuda', dtype=torch.float16):                                             # float16 mixed precision for speed
        x = tok_embed(batch_x) + pos_embed(torch.arange(context_size))                            # add token and positional embeddings
        loss = F.cross_entropy(linear(transformer(x)[:, -1, :]), batch_y)                         # computes softmax and cross-entropy loss automatically

    optimizer.zero_grad(); loss.backward(); optimizer.step(); scheduler.step()                    # zero grads, backprop, AdamW update, and lr decay

    if step % 200 == 0:
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):                        # disable tracking during evaluation
            eval_idx = torch.randint(0, len(input_ids), (4096,), generator=eval_rng)              # FIXED eval subset to prevent accuracy wobble
            x_eval = tok_embed(input_ids[eval_idx]) + pos_embed(torch.arange(context_size))       # embed dataset subset
            pred_ids = linear(transformer(x_eval)[:, -1, :]).argmax(1)                            # dataset subset forward & argmax
            print(f"Step {step:4d} | Loss: {loss:.4f} | Acc: {(pred_ids == target_ids[eval_idx]).float().mean():.1%} | {time.time()-start:.1f}s")

print(f"Training time: {time.time() - start:.1f}s")

# --- Generate ---
@torch.no_grad()                                                                                  # disable autograd tracking during inference
def generate(num_chars=200, context_ids=list(token_ids[:context_size]), temp=temp):               # start with true initial context
    output_chars = [idx_to_char[i] for i in context_ids]                                          # decode initial context to string
    for _ in range(num_chars):
        x = tok_embed(torch.tensor([context_ids])) + pos_embed(torch.arange(context_size))        # embed current window
        # Makes the model more confident, eliminating fake words like "throbe".
        next_token_probs = torch.softmax(linear(transformer(x)[:, -1, :]) / temp, 1)              # apply temp parameter to pick higher-confidence tokens
        next_token = torch.multinomial(next_token_probs, 1).item()                                # randomly sample from predicted distribution
        context_ids, output_chars = context_ids[1:] + [next_token], output_chars + [idx_to_char[next_token]] # slide window forward and append string
    return ''.join(output_chars)

print("\n--- Generated Story ---")
print(generate())
