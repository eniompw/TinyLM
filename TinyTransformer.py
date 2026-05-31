import time, torch
import torch.nn as nn, torch.nn.functional as F
#from tinystories_dataset import load_tinystories

# Automatically create all tensors on GPU if available, removing manual device boilerplate
torch.set_default_device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Data & Tokenization ---
context_size = 8                                                                        # number of previous chars to use as context for predicting the next char   
inputs, targets, vocab, encoded_text = load_tinystories(num_stories=1000, context_size=context_size) # previous chars to predict next
inputs, targets = torch.tensor(inputs), torch.tensor(targets)                           # convert to tensors

# --- Model ---
torch.manual_seed(42)                                                                   # seed helper for reproducibility
embed = nn.Embedding(len(vocab), 256)                                                   # token embedding lookup layer (256 is embed_dim)
pos_embed = nn.Embedding(context_size, 256)                                             # positional embedding for sequence order
transformer = torch.compile(nn.TransformerEncoder(nn.TransformerEncoderLayer(256, 4, 1024, batch_first=True, dropout=0., norm_first=True), 2))
model = nn.Linear(256, len(vocab))                                                      # maps hidden state to logits (vocab length)

params = list(embed.parameters()) + list(pos_embed.parameters()) + list(transformer.parameters()) + list(model.parameters())
optimizer = torch.optim.AdamW(params, lr=1e-3, fused=True)                              # optimizer replaces manual parameter updates
scaler = torch.amp.GradScaler('cuda')                                                   # scaler prevents float16 underflow
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 2000, eta_min=1e-4)   # smoothly decays learning rate

# --- Train ---
batch_size = 1024                                                                       # number of samples per batch
start = time.time()                                                                     # track training duration
for epoch in range(2001):
    batch_indices = torch.randint(0, len(inputs), (batch_size,))                        # random batch indices (len(inputs) is total examples)
    batch_inputs, batch_targets = inputs[batch_indices], targets[batch_indices]         # fetch random mini-batch (inputs and labels)

    # Forward pass
    with torch.autocast('cuda', dtype=torch.float16):                                   # float16 mixed precision for speed
        emb = embed(batch_inputs) + pos_embed(torch.arange(context_size))               # add token and positional embeddings
        logits = model(transformer(emb)[:, -1, :])                                      # forward pass through transformer & linear layer
        loss = F.cross_entropy(logits, batch_targets)                                   # computes softmax and cross-entropy loss automatically

    # Backward pass
    optimizer.zero_grad(set_to_none=True)                                               # zero out gradients for the next iteration
    scaler.scale(loss).backward()                                                       # autograd automatically computes scaled gradients
    scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(params, 1.0)             # unscale and clip to prevent loss spikes
    scaler.step(optimizer); scaler.update(); scheduler.step()                           # parameter update, scaler update, and lr decay

    if epoch % 200 == 0:
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):              # disable tracking during evaluation
            eval_idx = torch.randint(0, len(inputs), (4096,))                           # subset evaluation to prevent GPU OOM
            emb_eval = embed(inputs[eval_idx]) + pos_embed(torch.arange(context_size))  # embed dataset subset
            preds = model(transformer(emb_eval)[:, -1, :]).argmax(1)                    # dataset subset forward & argmax
            print(f"Epoch {epoch:4d} | Acc: {(preds == targets[eval_idx]).float().mean():.1%}")

print(f"Training time: {time.time() - start:.1f}s")

# --- Generate ---
@torch.no_grad()                                                                        # disable autograd tracking during inference
def generate(num_chars=200, context=list(encoded_text[:context_size])):                 # start with true initial context
    generated_text = [vocab[i] for i in context]                                        # decode initial context to string
    for _ in range(num_chars):
        emb = embed(torch.tensor([context])) + pos_embed(torch.arange(context_size))    # embed current window
        probabilities = torch.softmax(model(transformer(emb)[:, -1, :]) / 0.7, 1)       # apply temp 0.7 to pick higher-confidence tokens
        
        next_id = torch.multinomial(probabilities, 1).item()                            # randomly sample from predicted distribution
        context, generated_text = context[1:] + [next_id], generated_text + [vocab[next_id]] # slide window forward and append string
    return ''.join(generated_text)

print(generate())