import time, torch
import torch.nn as nn, torch.nn.functional as F
from tinystories_dataset import load_tinystories_bpe

torch.set_default_device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Hyperparameters ---
num_stories, context_size, vocab_size = 5000, 32, 4000
embed_dim, n_heads, ffn_dim, n_layers = 128, 4, 256, 3
batch_size, lr, n_steps, temp = 1536, 2e-3, 1801, 0.5

# --- Data ---
input_ids, target_ids, tokenizer, token_ids, vocab_size = load_tinystories_bpe(num_stories=num_stories, context_size=context_size, vocab_size=vocab_size)

# --- Model ---
torch.manual_seed(0)
eval_idx = torch.randint(0, len(input_ids), (4096,), generator=torch.Generator(device=input_ids.device).manual_seed(0))

tok_embed = nn.Embedding(vocab_size, embed_dim)
pos_embed = nn.Embedding(context_size, embed_dim)
transformer = torch.compile(nn.TransformerEncoder(nn.TransformerEncoderLayer(embed_dim, n_heads, ffn_dim, batch_first=True, dropout=0., norm_first=True), n_layers))
linear = nn.Linear(embed_dim, vocab_size)

params = list(tok_embed.parameters()) + list(pos_embed.parameters()) + list(transformer.parameters()) + list(linear.parameters())
optimizer = torch.optim.AdamW(params, lr=lr)
print(f"params: {sum(p.numel() for p in params):,}")

# --- Train ---
start = time.time()
for step in range(n_steps):
    idx = torch.randint(0, len(input_ids), (batch_size,))
    x = tok_embed(input_ids[idx]) + pos_embed(torch.arange(context_size))
    loss = F.cross_entropy(linear(transformer(x)[:, -1]), target_ids[idx])
    optimizer.zero_grad(); loss.backward(); optimizer.step()

    if step % 200 == 0:
        with torch.no_grad():
            pred = linear(transformer(tok_embed(input_ids[eval_idx]) + pos_embed(torch.arange(context_size)))[:, -1]).argmax(1)
            print(f"Step {step:4d} | Loss: {loss:.4f} | Acc: {(pred == target_ids[eval_idx]).float().mean():.1%} | {time.time()-start:.1f}s")

print(f"Training time: {time.time() - start:.1f}s")

# --- Generate ---
@torch.no_grad()
def generate(num_tokens=100, context_ids=list(token_ids[:context_size]), temp=temp):
    for _ in range(num_tokens):
        x = tok_embed(torch.tensor([context_ids])) + pos_embed(torch.arange(context_size))
        next_token = torch.multinomial(torch.softmax(linear(transformer(x)[:, -1]) / temp, 1), 1).item()
        context_ids = context_ids[1:] + [next_token]
    return tokenizer.decode(context_ids)

print("\n--- Generated Story ---")
print(generate())