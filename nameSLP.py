from names_dataset import load_names
import numpy as np

def softmax(x):
    exp_x = np.exp(x - x.max(axis=1, keepdims=True))            # subtract max for numerical stability
    return exp_x / exp_x.sum(axis=1, keepdims=True)             # normalize so all probabilities sum to 1

# --- Dataset ---
context_size = 6                                                # increased to 6 for better SLP accuracy
X, y, vocab = load_names(context_size=context_size)             # load dataset features, targets, and vocabulary
vocab_size = len(vocab)                                         # total count of unique characters
y_one_hot = np.eye(vocab_size)[y]                               # one-hot encode labels

# --- Model ---
np.random.seed(42)
W = np.random.randn(context_size * vocab_size, vocab_size)*0.1  # weights: (6 * vocab_size) inputs → vocab_size classes
b = np.zeros((1, vocab_size))                                   # biases for each class
learning_rate = 5.0                                             # step size for gradient descent

# --- Train ---
for epoch in range(1001):
    probs = softmax(X @ W + b)                                  # forward pass
    dlogits = (probs - y_one_hot) / len(X)                      # cross-entropy + softmax gradient
    W -= learning_rate * X.T @ dlogits                          # update weights
    b -= learning_rate * dlogits.sum(0, keepdims=True)          # update biases
    
    if epoch % 200 == 0:
        acc = np.mean(probs.argmax(1) == y)                     # calculate exact match accuracy
        print(f"Epoch {epoch:4d} | Acc: {acc:.1%}")

# --- Generate ---
ctx = X[0].reshape(context_size, vocab_size).argmax(1).tolist() # extract first 6 chars from X[0] to use as seed
out = [vocab[i] for i in ctx]                                   # initialize output string list

for _ in range(100):
    x_gen = np.eye(vocab_size)[ctx].reshape(1, -1)              # one-hot encode and flatten the current context
    p = softmax(x_gen @ W + b)[0]                               # forward pass to get probability distribution
    next_id = int(np.random.choice(vocab_size, p=p))            # randomly sample from predicted distribution
    
    ctx = ctx[1:] + [next_id]                                   # slide context window forward by dropping oldest char
    out.append(vocab[next_id])                                  # append decoded character to output

print("\nGenerated names:")
print(''.join(out))