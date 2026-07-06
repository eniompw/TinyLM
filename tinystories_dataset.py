import torch, warnings, itertools
from datasets import load_dataset
warnings.filterwarnings('ignore')

def load_tinystories(num_stories=200, context_size=4):
    """
    Fetches TinyStories and prepares it for a character-level language model.
    Returns: input_ids, target_ids, vocab (list), encoded (list), vocab_size (int)
    """
    dataset = load_dataset('karpathy/tinystories-gpt4-clean', split='train', streaming=True)
    text = ''.join(s['text'] for s in itertools.islice(dataset, num_stories))

    vocab = sorted(set(text))                                        # ordered list of unique characters
    char_to_id = {c: i for i, c in enumerate(vocab)}                 # char → integer id
    encoded = [char_to_id[c] for c in text]                          # full text as integer sequence

    if context_size == 1:
        return [], [], vocab, encoded, len(vocab)

    inputs  = [encoded[i:i+context_size] for i in range(len(encoded)-context_size)]
    targets = encoded[context_size:]

    return torch.tensor(inputs), torch.tensor(targets), vocab, encoded, len(vocab)


def load_tinystories_bpe(num_stories=5000, context_size=32, vocab_size=4000):
    """
    Fetches TinyStories and prepares it for a BPE subword language model.
    Returns: input_ids, target_ids, tokenizer, encoded (list), vocab_size (int)
    """
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import ByteLevel
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder

    dataset = load_dataset('karpathy/tinystories-gpt4-clean', split='train', streaming=True)
    text = ''.join(s['text'] for s in itertools.islice(dataset, num_stories))

    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel()
    tokenizer.decoder = ByteLevelDecoder()
    tokenizer.train_from_iterator([text], BpeTrainer(vocab_size=vocab_size, special_tokens=["<unk>"]))

    encoded = tokenizer.encode(text).ids
    inputs  = [encoded[i:i+context_size] for i in range(len(encoded)-context_size)]
    targets = encoded[context_size:]

    return torch.tensor(inputs), torch.tensor(targets), tokenizer, encoded, tokenizer.get_vocab_size()