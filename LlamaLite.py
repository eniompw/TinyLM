import time, torch
import torch.nn as nn, torch.nn.functional as F
from tinystories_dataset import load_tinystories

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_device(device)
print(f"device: {device}" + (f" | {torch.cuda.get_device_name(0)}" if device.type == 'cuda' else ''))

# --- Data & Tokenization ---
context_size = 256
input_ids, target_ids, idx_to_char, token_ids = load_tinystories(num_stories=1000, context_size=context_size) # previous chars to predict next
input_ids, target_ids = torch.tensor(input_ids), torch.tensor(target_ids)                        # convert to tensors

# --- Model Definitions ---
torch.manual_seed(42)                                                                             # seed helper for reproducibility

class RMSNorm(nn.Module):
    def forward(self, x): return x * (x.pow(2).mean(-1, keepdim=True) + 1e-5).rsqrt()             # normalise without mean subtraction

def apply_rope(x, cos, sin):
    x0, x1 = x.float().unflatten(-1, (-1, 2)).unbind(-1)                                          # split last dim into (head_dim//2, 2) pairs
    cos, sin = cos.view(1, -1, 1, cos.shape[-1]), sin.view(1, -1, 1, sin.shape[-1])
    return torch.stack([x0*cos - x1*sin, x0*sin + x1*cos], -1).flatten(-2).to(x.dtype)            # rotate pairs to encode relative position

class ModernBlock(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.n_head, self.head_dim = n_head, n_embd // n_head
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)                                   # fused Q, K, V projection
        self.wo = nn.Linear(n_embd, n_embd, bias=False)                                           # output projection
        self.mlp = nn.Sequential(nn.Linear(n_embd, 4 * n_embd, bias=False), nn.SiLU(), nn.Linear(4 * n_embd, n_embd, bias=False)) # MLP block
        self.norm1, self.norm2 = RMSNorm(), RMSNorm()                                             # pre-norm layers

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        r = x; x = self.norm1(x)                                                                  # save residual, pre-norm attention input
        q, k, v = self.c_attn(x).split(C, dim=2)                                                  # project and split into Q, K, V
        q, k, v = [t.view(B, T, self.n_head, self.head_dim) for t in (q, k, v)]                   # reshape for multi-head attention
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)                                   # inject positional info into q and k
        x = F.scaled_dot_product_attention(q.transpose(1,2), k.transpose(1,2), v.transpose(1,2), is_causal=True).transpose(1,2).reshape(B, T, C)
        x = self.wo(x) + r                                                                        # flash attention + output projection + residual
        return self.mlp(self.norm2(x)) + x                                                        # pre-norm MLP + residual

class LlamaLite(nn.Module):
    def __init__(self, vocab_size, n_layer=2, n_embd=256, n_head=4):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, n_embd)                                               # token embedding lookup layer
        self.blocks = nn.ModuleList([ModernBlock(n_embd, n_head) for _ in range(n_layer)])        # stack of transformer blocks
        self.norm = RMSNorm()                                                                     # final layer norm
        for p in self.parameters():
            if p.dim() > 1: nn.init.normal_(p, mean=0.0, std=0.02)                                # init weights matching tiny scale
        t, f = torch.arange(context_size).float(), 1.0 / (10000.0 ** (torch.arange(0, n_embd//n_head, 2).float() / (n_embd//n_head)))
        self.register_buffer('rope_cos', torch.outer(t, f).cos()), self.register_buffer('rope_sin', torch.outer(t, f).sin()) # RoPE tables

    def forward(self, tokens):
        x, cos, sin = self.wte(tokens), self.rope_cos[:tokens.shape[1]], self.rope_sin[:tokens.shape[1]] # embeddings and sliced rope tables
        for b in self.blocks: x = b(x, cos, sin)                                                  # pass through transformer blocks
        return F.linear(self.norm(x), self.wte.weight)                                            # weight-tied lm_head over full sequence

# --- Setup & Train ---
model = torch.compile(LlamaLite(len(idx_to_char)))                                                # fuse GPU kernels for ~2x speedup
print(f"params: {sum(p.numel() for p in model.parameters()):,}")
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95), fused=True)         # native param extraction, fused AdamW
scheduler, scaler, start = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 2000, 1e-4), torch.amp.GradScaler('cuda'), time.time()
print("model ready")

for step in range(2001):
    batch_idx = torch.randint(0, len(input_ids), (1024,))                                         # random batch indices (batch_size=1024)
    batch_x, batch_y = input_ids[batch_idx], target_ids[batch_idx]                                # fetch random mini-batch

    with torch.autocast('cuda', dtype=torch.float16):                                             # float16 mixed precision for speed
        loss = F.cross_entropy(model(batch_x)[:, -1, :], batch_y)                                 # take last token logits and compute loss

    optimizer.zero_grad(); scaler.scale(loss).backward(); scaler.unscale_(optimizer)              # zero grads, backprop, unscale for clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)                                       # prevent loss spikes from large gradients
    scaler.step(optimizer); scaler.update(); scheduler.step()                                     # AdamW update, scaler update, and lr decay

    if step % 200 == 0:
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):                        # disable tracking during evaluation
            eval_idx = torch.randint(0, len(input_ids), (4096,))                                  # subset evaluation to prevent GPU OOM
            pred_ids = model(input_ids[eval_idx])[:, -1, :].argmax(1)                             # dataset subset forward & argmax
            print(f"Step {step:4d} | Loss: {loss:.4f} | Acc: {(pred_ids == target_ids[eval_idx]).float().mean():.1%} | {time.time()-start:.1f}s")

print(f"Training time: {time.time() - start:.1f}s")

# --- Generate ---
@torch.no_grad()                                                                                   # disable autograd tracking during inference
def generate(num_chars=200, ctx=list(token_ids[:context_size])):                                 # start with true initial context
    out = [idx_to_char[i] for i in ctx]                                                           # decode initial context to string
    for _ in range(num_chars):
        next_t = torch.multinomial(F.softmax(model(torch.tensor([ctx[-context_size:]]))[0, -1, :] / 0.7, -1), 1).item() # predict next token
        ctx = ctx[1:] + [next_t]; out.append(idx_to_char[next_t])                                 # slide window and append char
    return ''.join(out)

print(generate())
