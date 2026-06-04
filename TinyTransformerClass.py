import time, torch
import torch.nn as nn, torch.nn.functional as F
from tinystories_dataset import load_tinystories

# Automatically create all tensors on GPU if available, removing manual device boilerplate
torch.set_default_device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Hyperparameters ---
context_size, embed_dim          = 8, 256                                                         # context window size; embedding dimension (d_model)
n_heads, ffn_dim, n_layers       = 4, 1024, 2                                                     # attention heads; FFN hidden dim; transformer layers
batch_size, lr, n_steps          = 1024, 1e-3, 2001                                               # samples per step; learning rate; total training steps

# --- Data & Tokenization ---
input_ids, target_ids, idx_to_char, token_ids = load_tinystories(num_stories=1000, context_size=context_size) # previous chars to predict next
input_ids, target_ids = torch.tensor(input_ids), torch.tensor(target_ids)                        # convert to tensors

# --- Model ---
torch.manual_seed(42)                                                                             # seed helper for reproducibility

class TinyTransformer(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.tok_embed = nn.Embedding(vocab_size, embed_dim)                                      # token embedding lookup layer
        self.pos_embed = nn.Embedding(context_size, embed_dim)                                    # positional embedding for sequence order
        self.transformer = torch.compile(nn.TransformerEncoder(nn.TransformerEncoderLayer(embed_dim, n_heads, ffn_dim, batch_first=True, dropout=0., norm_first=True), n_layers))
        self.linear = nn.Linear(embed_dim, vocab_size)                                            # maps hidden state to logits (vocab length)

    def forward(self, x):
        x = self.tok_embed(x) + self.pos_embed(torch.arange(x.shape[1]))                        # add token and positional embeddings
        return self.linear(self.transformer(x)[:, -1, :])                                        # transformer + project last token to logits

model = TinyTransformer(len(idx_to_char))
print(f"params: {sum(p.numel() for p in model.parameters()):,}")                                 # print total parameter count
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), fused=True)         # optimizer replaces manual parameter updates
scheduler, start = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_steps, eta_min=1e-4), time.time() # smoothly decays learning rate

# --- Train ---
def get_batch():
    batch_idx = torch.randint(0, len(input_ids), (batch_size,))                                  # random batch indices (len(input_ids) is total examples)
    return input_ids[batch_idx], target_ids[batch_idx]                                           # fetch random mini-batch (inputs and labels)

for step in range(n_steps):
    batch_x, batch_y = get_batch()

    with torch.autocast('cuda', dtype=torch.float16):                                            # float16 mixed precision for speed
        loss = F.cross_entropy(model(batch_x), batch_y)                                          # computes softmax and cross-entropy loss automatically

    optimizer.zero_grad(); loss.backward(); optimizer.step(); scheduler.step()                   # zero grads, backprop, AdamW update, and lr decay

    if step % 200 == 0:
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):                       # disable tracking during evaluation
            eval_idx = torch.randint(0, len(input_ids), (4096,))                                 # subset evaluation to prevent GPU OOM
            pred_ids = model(input_ids[eval_idx]).argmax(1)                                      # dataset subset forward & argmax
            print(f"Step {step:4d} | Loss: {loss:.4f} | Acc: {(pred_ids == target_ids[eval_idx]).float().mean():.1%} | {time.time()-start:.1f}s")

print(f"Training time: {time.time() - start:.1f}s")

# --- Generate ---
@torch.no_grad()                                                                                  # disable autograd tracking during inference
def generate(num_chars=200, context_ids=list(token_ids[:context_size])):                        # start with true initial context
    output_chars = [idx_to_char[i] for i in context_ids]                                         # decode initial context to string
    for _ in range(num_chars):
        next_token_probs = torch.softmax(model(torch.tensor([context_ids])) / 0.7, 1)           # apply temp 0.7 to pick higher-confidence tokens
        next_token = torch.multinomial(next_token_probs, 1).item()                               # randomly sample from predicted distribution
        context_ids, output_chars = context_ids[1:] + [next_token], output_chars + [idx_to_char[next_token]] # slide window forward and append string
    return ''.join(output_chars)

print(generate())
