from datasets import load_dataset
import itertools
import cupy as cp

def load_tinystories(num_records=200, context_size=4):
    """
    Fetches the TinyStories dataset and prepares it for a character-level language model.
    
    Returns:
        inputs (cp.ndarray): Sliding window context arrays of shape (N, context_size)
        targets (cp.ndarray): Target next-character arrays of shape (N,)
        vocab (list): The list of unique characters (vocabulary)
        encoded (list): The full encoded text as a list of integer IDs
    """
    # Fetch data
    dataset = load_dataset('karpathy/tinystories-gpt4-clean', split='train', streaming=True)
    text = ''.join(s['text'] for s in itertools.islice(dataset, num_records))

    # Build vocabulary and tokenize
    vocab = sorted(set(text))                                       
    char_to_id = {c: i for i, c in enumerate(vocab)}                
    encoded = [char_to_id[c] for c in text]                         

    # Create sliding windows for inputs and targets
    inputs = cp.array([encoded[i:i+context_size] for i in range(len(encoded)-context_size)])
    targets = cp.array(encoded[context_size:])                                                

    return inputs, targets, vocab, encoded