import urllib.request
import numpy as np

def load_karpathy_names(context_size=4, limit=10000):
    url = "https://raw.githubusercontent.com/karpathy/makemore/master/names.txt"
    text = urllib.request.urlopen(url).read().decode('utf-8')   # fetch names dataset directly from github
    text = text[:limit]                                         # slice to prevent memory overflow in pure numpy

    vocab = sorted(set(text))                                   # ordered list of unique characters
    char_to_id = {c: i for i, c in enumerate(vocab)}            # dictionary mapping char to integer id
    encoded = [char_to_id[c] for c in text]                     # map entire text to integer sequence

    X_seq = np.array([encoded[i:i+context_size] for i in range(len(encoded)-context_size)]) 
    y = np.array(encoded[context_size:])                        # next char to predict
    
    vocab_size = len(vocab)                                     # total count of unique characters
    X = np.eye(vocab_size)[X_seq].reshape(len(X_seq), -1)       # one-hot encode and flatten to 2D array
    
    return X, y, vocab                                          # return features, labels, and vocabulary