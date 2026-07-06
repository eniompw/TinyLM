import time, torch
import torch.nn as nn, torch.nn.functional as F
from tinystories_dataset import load_tinystories_bpe

# Automatically create all tensors on GPU if available, removing manual device boilerplate
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_default_device(device)

# --- Hyperparameters ---
num_stories  = 5000                                                                               # stories to train on — large enough to avoid memorisation
context_size = 32                                                                                 # BPE tokens per context window (~20-25 words, vs ~5 words for char-level)
embed_dim    = 256                                                                                # token/positional embedding dimension (d_model)
n_heads      = 4                                                                                  # number of attention heads in each transformer layer
ffn_dim      = 1024                                                                               # feed-forward network hidden dimension
n_layers     = 3                                                                                  # 3 layers is the sweet spot for depth vs speed
batch_size   = 2048                                                                               # large batch + high LR is the key driver of fast convergence
lr           = 2e-3                                                                               # high learning rate to match the large batch
n_steps      = 1001                                                                               # tuned to ~116s: maximises steps within the 2-min budget
temp         = 0.7                                                                                # sampling temperature — higher = more creative, lower = more accurate
softcap      = 15.0                                                                               # Gemma 2 logit softcapping: bounds extreme logits, prevents divergence

# --- Data & Tokenization ---
input_ids, target_ids, tokenizer, encoded, vocab_size = load_tinystories_bpe(num_stories, context_size) # custom BPE vocab=4000 trained on TinyStories corpus

# --- Model ---
torch.manual_seed(0)                                                                              # seed for reproducibility
eval_rng    = torch.Generator(device=device).manual_seed(0)                                       # dedicated GPU generator to eliminate accuracy wobble
tok_embed   = nn.Embedding(vocab_size, embed_dim)                                                 # token embedding lookup layer
pos_embed   = nn.Embedding(context_size, embed_dim)                                               # positional embedding for sequence order
transformer = torch.compile(nn.TransformerEncoder(                                                # torch.compile fuses ops for ~1.2x speedup after one-time ~10s compile tax
    nn.TransformerEncoderLayer(embed_dim, n_heads, ffn_dim, batch_first=True, dropout=0., norm_first=True), # norm_first=True is Pre-LN: stabler gradients than Post-LN
    n_layers))
linear = nn.Linear(embed_dim, vocab_size)                                                         # maps hidden state to logits (vocab length)

params = list(tok_embed.parameters()) + list(pos_embed.parameters()) + list(transformer.parameters()) + list(linear.parameters())
print(f"params: {sum(p.numel() for p in params):,}")

# weight_decay=0.01 prevents overfitting and encourages smooth grammar
optimizer = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.95), weight_decay=0.01, fused=True)   # fused=True merges optimizer kernels for faster GPU updates
scheduler, start = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_steps, eta_min=1e-4), time.time() # smoothly decays learning rate to eta_min

# --- Warmup compile ---
with torch.autocast('cuda', dtype=torch.float16):                                                 # triggers torch.compile graph build before the training timer starts
    _ = linear(transformer(tok_embed(input_ids[:1]) + pos_embed(torch.arange(context_size))))

# --- Train ---
for step in range(n_steps):
    batch_idx = torch.randint(0, len(input_ids), (batch_size,))                                   # random batch indices
    batch_x, batch_y = input_ids[batch_idx], target_ids[batch_idx]                                # fetch random mini-batch (inputs and labels)

    with torch.autocast('cuda', dtype=torch.float16):                                             # float16 mixed precision halves memory bandwidth for major speedup
        x      = tok_embed(batch_x) + pos_embed(torch.arange(context_size))                       # add token and positional embeddings
        logits = softcap * torch.tanh(linear(transformer(x)[:, -1, :]) / softcap)                 # forward pass with logit softcapping to bound extreme values
        loss   = F.cross_entropy(logits, batch_y)                                                  # computes softmax and cross-entropy loss automatically

    optimizer.zero_grad(); loss.backward(); optimizer.step(); scheduler.step()                    # zero grads, backprop, AdamW update, lr decay

    if step % 100 == 0:
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):                        # disable gradient tracking during evaluation
            eval_idx    = torch.randint(0, len(input_ids), (4096,), generator=eval_rng)           # fixed eval subset to prevent accuracy wobble
            x_eval      = tok_embed(input_ids[eval_idx]) + pos_embed(torch.arange(context_size))  # embed eval subset
            logits_eval = softcap * torch.tanh(linear(transformer(x_eval)[:, -1, :]) / softcap)   # eval forward pass with softcapping
            pred_ids    = logits_eval.argmax(1)                                                    # greedy prediction
            print(f"Step {step:4d} | Loss: {loss:.4f} | Acc: {(pred_ids == target_ids[eval_idx]).float().mean():.1%} | {time.time()-start:.1f}s")

print(f"Training time: {time.time() - start:.1f}s")

# --- Generate ---
@torch.no_grad()                                                                                   # disable autograd tracking during inference
def generate(num_tokens=100, context_ids=list(encoded[:context_size]), temp=temp):                # start with true initial context
    output = list(context_ids)
    for _ in range(num_tokens):
        x      = tok_embed(torch.tensor([context_ids])) + pos_embed(torch.arange(context_size))   # embed current window
        logits = softcap * torch.tanh(linear(transformer(x)[:, -1, :]) / softcap)                 # forward pass with softcapping
        next_token = torch.multinomial(torch.softmax(logits / temp, 1), 1).item()                 # sample from temperature-scaled distribution
        context_ids, output = context_ids[1:] + [next_token], output + [next_token]               # slide window forward and append token
    return tokenizer.decode(output)                                                                # decode BPE token ids back to clean text

print("\n--- Generated Story ---")
print(generate())