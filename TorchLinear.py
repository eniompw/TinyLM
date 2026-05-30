from datasets import load_dataset
import itertools, warnings
import torch, torch.nn as nn, torch.nn.functional as F
warnings.filterwarnings('ignore')

# --- Data ---
dataset = load_dataset('karpathy/tinystories-gpt4-clean', split='train', streaming=True)
text    = '\n'.join(s['text'] for s in itertools.islice(dataset, 5000))  # first 5000 stories

# --- Tokenization ---
vocab       = sorted(set(text))                             # unique characters
vocab_size  = len(vocab)
char_to_idx = {c: i for i, c in enumerate(vocab)}          # char → integer
idx_to_char = {i: c for i, c in enumerate(vocab)}          # integer → char
data        = torch.tensor([char_to_idx[c] for c in text], dtype=torch.long).cuda()

# --- Config ---
block_size = 16    # characters of context fed to the model
embed_dim  = 32    # embedding vector size per character
hidden_dim = 256   # neurons per hidden layer
batch_size = 256   # training examples per step
input_dim  = block_size * embed_dim  # flattened input to MLP (512)

# --- Model ---
embed = nn.Embedding(vocab_size, embed_dim).cuda()  # learns a vector per character
pos   = nn.Embedding(block_size, embed_dim).cuda()  # learns a vector per position
model = nn.Sequential(
    nn.Linear(input_dim, hidden_dim), nn.ReLU(),   # hidden layer 1
    nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),  # hidden layer 2
    nn.Linear(hidden_dim, vocab_size)              # output: score for each character
).cuda()

def forward(tokens):
    x = embed(tokens) + pos(torch.arange(block_size).cuda())  # embed + position
    return model(x.view(tokens.shape[0], -1))                 # flatten → MLP → logits

# --- Train ---
optimizer = torch.optim.AdamW(
    list(model.parameters()) + list(embed.parameters()) + list(pos.parameters()), lr=1e-3
)

for epoch in range(3000):
    offsets = torch.randint(0, len(data) - block_size - 1, (batch_size,)).cuda()  # random positions
    inputs  = data[offsets.unsqueeze(1) + torch.arange(block_size).cuda()]        # (batch, block)
    targets = data[offsets + block_size]                                           # next character

    loss = F.cross_entropy(forward(inputs), targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 500 == 0:
        print(f"epoch {epoch:4d} | loss {loss.item():.3f}")

# --- Generate ---
def generate(start_idx=0, num_chars=300):
    context = data[start_idx : start_idx + block_size].unsqueeze(0)  # seed context
    output  = [idx_to_char[i.item()] for i in context[0]]
    for _ in range(num_chars):
        next_idx = torch.multinomial(F.softmax(forward(context), dim=-1), 1).item()  # sample next char
        output.append(idx_to_char[next_idx])
        context  = torch.cat([context[:, 1:], torch.tensor([[next_idx]]).cuda()], dim=1)  # slide window
    return ''.join(output)

print(generate())
