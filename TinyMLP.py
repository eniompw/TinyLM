import time, cupy as cp, numpy as np
from tinystories_dataset import load_tinystories

def softmax(logits):                                            # converts raw network outputs to probabilities
    exponentials = cp.exp(logits - logits.max(axis=1, keepdims=True)) # shift to prevent float overflow
    return exponentials / exponentials.sum(axis=1, keepdims=True)     # normalize so all probabilities sum to 1

# --- Data & Tokenization ---
block_size = 4                                                  # number of previous chars used to predict next
raw_in, raw_out, vocab, encoded_text = load_tinystories(num_stories=200, context_size=block_size)
inputs, targets = cp.array(raw_in), cp.array(raw_out)           # convert to CuPy arrays for GPU slicing

vocab_size = len(vocab)                                         # total count of unique characters
num_examples = len(inputs)                                      # total number of training examples

# --- Model ---
emb_dim, hidden_size = 256, 150                                 # embedding dimensions, hidden layer neurons
cp.random.seed(42); randn = lambda *s: (cp.random.randn(*s) * 0.1).astype(cp.float32) # seed & normal init helper
embed_matrix   = randn(vocab_size, emb_dim)                     # token embedding lookup matrix
weights_hidden = randn(block_size * emb_dim, hidden_size)       # weights mapping context to hidden state
weights_output = randn(hidden_size, vocab_size)                 # weights mapping hidden state to logits

# --- Train ---
lr, batch_size = 0.5, 1024                                      # learning rate, number of samples per batch
start = time.time()                                             # track training duration
for epoch in range(2001):
    batch_indices = cp.random.randint(0, num_examples, size=batch_size) # random array of indices for batch
    batch_inputs, batch_targets = inputs[batch_indices], targets[batch_indices] # fetch random mini-batch

    # Forward pass
    context_embeds = embed_matrix[batch_inputs].reshape(batch_size, -1) # concatenate embeddings for the window
    hidden_state   = cp.maximum(0, context_embeds @ weights_hidden)     # apply ReLU non-linearity to hidden state
    probabilities  = softmax(hidden_state @ weights_output)             # get probability distribution over vocab

    # Backward pass
    probabilities[cp.arange(batch_size), batch_targets] -= 1            # in-place CE gradient: (probs - 1) for true labels
    probabilities /= batch_size                                         # average loss over batch (probs is now dlogits)

    grad_weights_output = hidden_state.T @ probabilities                # gradient for output weights
    grad_hidden_state   = (probabilities @ weights_output.T) * (hidden_state > 0) # gradient for hidden state (ReLU backprop)
    grad_weights_hidden = context_embeds.T @ grad_hidden_state          # gradient for hidden weights

    grad_embed_matrix = cp.zeros_like(embed_matrix)                     # gradient accumulator for embeddings
    cp.add.at(grad_embed_matrix, batch_inputs.ravel(), (grad_hidden_state @ weights_hidden.T).reshape(-1, emb_dim))

    for param, grad in zip((embed_matrix, weights_hidden, weights_output), (grad_embed_matrix, grad_weights_hidden, grad_weights_output)):
        param -= lr * grad                                              # standard SGD parameter update

    if epoch % 200 == 0:
        logits = cp.maximum(0, embed_matrix[inputs].reshape(num_examples, -1) @ weights_hidden) @ weights_output
        preds = logits.argmax(1)                                        # mathematical shortcut: argmax directly on logits
        print(f"Epoch {epoch:4d} | Acc: {cp.mean(preds == targets):.1%}")

print(f"Training time: {time.time() - start:.1f}s")

# --- Generate ---
def generate(num_chars=200):
    context = list(encoded_text[:block_size])                   # start with true initial context from text
    generated_text = [vocab[i] for i in context]                # decode initial context to string
    for _ in range(num_chars):
        sample_probs = cp.asnumpy(softmax(cp.maximum(0, embed_matrix[cp.array([context])].reshape(1, -1) @ weights_hidden) @ weights_output)[0])
        next_id = int(np.random.choice(vocab_size, p=sample_probs)) # randomly sample from predicted distribution
        context = context[1:] + [next_id]                       # slide context window forward by one token
        generated_text.append(vocab[next_id])
    return ''.join(generated_text)

print(generate())