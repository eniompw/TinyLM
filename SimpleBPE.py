import time, torch                                                          # Timing + tensors/GPU tools
import torch.nn as nn, torch.nn.functional as F                              # Neural layers + cross-entropy loss
# from tinystories_dataset import load_tinystories_bpe                          # TinyStories custom-BPE loader

torch.set_default_device('cuda' if torch.cuda.is_available() else 'cpu')     # Use GPU when available

# --- Hyperparameters ---
num_stories, context_size, vocab_size = 5000, 32, 4000                       # Data size, BPE memory, tokenizer size
embed_dim, n_heads, ffn_dim, n_layers = 128, 4, 256, 3                        # Transformer shape
batch_size, lr, n_steps, temp = 2048, 2e-3, 1801, 0.5                         # AMP-best training + generation settings

# --- Data ---
input_ids, target_ids, tokenizer, token_ids, vocab_size = load_tinystories_bpe(  # Build BPE training contexts
    num_stories=num_stories, context_size=context_size, vocab_size=vocab_size)

# --- Model ---
torch.manual_seed(42)                                                         # Reproducible weights and batches
eval_idx = torch.randint(0, len(input_ids), (4096,),                         # Fixed evaluation subset
    generator=torch.Generator(device=input_ids.device).manual_seed(42))

tok_embed = nn.Embedding(vocab_size, embed_dim)                              # BPE token IDs -> vectors
pos_embed = nn.Embedding(context_size, embed_dim)                            # Token positions -> vectors
transformer = torch.compile(nn.TransformerEncoder(                           # Compile attention stack for speed
    nn.TransformerEncoderLayer(embed_dim, n_heads, ffn_dim, batch_first=True, dropout=0., norm_first=True), n_layers))
linear = nn.Linear(embed_dim, vocab_size)                                    # Vectors -> next-token scores

params = list(tok_embed.parameters()) + list(pos_embed.parameters()) + list(transformer.parameters()) + list(linear.parameters())  # All trainable weights
optimizer = torch.optim.AdamW(params, lr=lr, fused=True)                     # Fast fused CUDA AdamW
scaler = torch.amp.GradScaler('cuda')                                        # Protect float16 gradients from underflow
print(f"params: {sum(p.numel() for p in params):,}")                         # Confirm model size

# --- Train ---
start = time.time()                                                          # Start warm-compiled training timer
for step in range(n_steps):                                                  # Repeat random-batch updates
    idx = torch.randint(0, len(input_ids), (batch_size,))                    # Sample training contexts

    with torch.autocast('cuda', torch.float16):                              # Run suitable GPU operations in float16
        x = tok_embed(input_ids[idx]) + pos_embed(torch.arange(context_size))  # Token + position information
        loss = F.cross_entropy(linear(transformer(x)[:, -1]), target_ids[idx])  # Predict final next token

    optimizer.zero_grad(); scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()  # Clear, scale gradients, update, adjust scale

    if step % 200 == 0:                                                      # Log learning every 200 updates
        with torch.no_grad(), torch.autocast('cuda', torch.float16):         # Evaluation: no gradients, float16
            x = tok_embed(input_ids[eval_idx]) + pos_embed(torch.arange(context_size))  # Embed fixed eval contexts
            pred = linear(transformer(x)[:, -1]).argmax(1)                   # Choose most likely next token
            print(f"Step {step:4d} | Loss: {loss:.4f} | Acc: {(pred == target_ids[eval_idx]).float().mean():.1%} | {time.time()-start:.1f}s")  # Report progress

print(f"Training time: {time.time() - start:.1f}s")                          # Total warm training time

# --- Generate ---
@torch.no_grad()                                                             # Disable gradients for generation
def generate(num_tokens=100, context_ids=list(token_ids[:context_size]), temp=temp):  # Continue initial BPE context
    for _ in range(num_tokens):                                              # Generate one token at a time
        with torch.autocast('cuda', torch.float16):                          # Use fast float16 inference
            x = tok_embed(torch.tensor([context_ids])) + pos_embed(torch.arange(context_size))  # Embed current context
            next_token = torch.multinomial(torch.softmax(linear(transformer(x)[:, -1]) / temp, 1), 1).item()  # Sample next token
        context_ids = context_ids[1:] + [next_token]                         # Slide context window forward
    return tokenizer.decode(context_ids)                                     # Convert BPE IDs back to text

print("\n--- Generated Story ---")                                           # Output label
print(generate())                                                            # Generate and display a story