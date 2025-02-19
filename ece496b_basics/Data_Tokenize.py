import numpy as np
from ece496b_basics.BPE_Tokenizer import BPE_Tokenizer

# Read raw text file
with open("../data/TinyStoriesV2-GPT4-train.txt", "r", encoding="utf-8") as f:
    text = f.read()
vocab_filename = "Tiny_bpe_vocab1.pkl"
merges_filename = "Tiny_bpe_merges1.pkl"

tokenizer = BPE_Tokenizer.from_files(vocab_filename, merges_filename)
token_ids = tokenizer.encode(text)  
# NumPy array
train_data = np.array(token_ids, dtype=np.uint16)

# Save as .npy file
np.save("Tiny_train.npy", train_data)


# print(tokenizer.decode(token_ids))


