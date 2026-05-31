from datasets import load_dataset
import itertools

def load_tinystories(num_stories=200, context_size=4):
    """
    Fetches the TinyStories dataset and prepares it for a character-level language model.
    """
    # Fetch data
    dataset = load_dataset('karpathy/tinystories-gpt4-clean', split='train', streaming=True)
    text = ''.join(s['text'] for s in itertools.islice(dataset, num_stories))

    # Build vocabulary and tokenize
    vocab = sorted(set(text))                                       # ordered list of unique characters
    char_to_id = {c: i for i, c in enumerate(vocab)}                # dictionary mapping char to integer id
    encoded = [char_to_id[c] for c in text]                         # map entire text to integer sequence

    # If context_size=1, skip building windows to save RAM (training loop handles slicing).
    if context_size == 1:
        return [], [], vocab, encoded

    # Create sliding windows for inputs and targets using standard Python lists
    inputs = [encoded[i:i+context_size] for i in range(len(encoded)-context_size)] # sliding windows
    targets = encoded[context_size:]                                               # next char to predict

    return inputs, targets, vocab, encoded