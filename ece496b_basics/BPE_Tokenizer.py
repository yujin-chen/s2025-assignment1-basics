import json
from typing import Dict, List, Tuple, Iterable, Iterator
import os, json, pickle

    
class BPE_Tokenizer:
    def __init__(self, vocab: Dict[int, bytes], merges: List[Tuple[bytes, bytes]], special_byte_tokens: List[str] | None = None):
        self.vocab = vocab
        self.merges = merges
        self.special_byte_tokens = special_byte_tokens or []
        
        #reverse vocab mapping
        self.token2id = {b: tid for tid, b in vocab.items()}
        
        # Assign ranks to merges
        self.merges_ranks = {pair: i for i, pair in enumerate(merges)}
        
        # Add special byte_tokens to vocab if not already present
        next_id = max(self.vocab.keys(), default=-1) + 1
        for token in self.special_byte_tokens:
            token_bytes = token.encode("utf-8")
            if token_bytes not in self.token2id:
                self.vocab[next_id] = token_bytes
                self.token2id[token_bytes] = next_id
                next_id += 1
        
        # Sort special byte_tokens by length 
        self._specials_sorted = sorted(self.special_byte_tokens, key=len, reverse=True)
    
    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_byte_tokens=None):
        vocab = load_vocab(vocab_filepath)
        merges = load_merges(merges_filepath)
        
        # Optionally add special byte tokens if not present.
        if special_byte_tokens:
            existing_byte_tokens = set(vocab.values())
            for token in special_byte_tokens:
                token_bytes = token.encode("utf-8")
                if token_bytes not in existing_byte_tokens:
                    vocab[len(vocab)] = token_bytes  
        
        return cls(vocab, merges, special_byte_tokens)

    
    def encode(self, text: str) -> List[int]:
        # Pre-tokenization 
        i, n, chunks = 0, len(text), []
        while i < n:
            for special_tok in self._specials_sorted:
                if text.startswith(special_tok, i):
                    chunks.append(special_tok)
                    i += len(special_tok)
                    break
            else:
                start = i
                while i < n and not any(text.startswith(special_tok, i) for special_tok in self._specials_sorted):
                    if text[i] == "\n":
                        if start < i:
                            chunks.append(text[start:i])
                        chunks.append("\n")
                        i += 1
                        start = i
                    else:
                        i += 1
                if start < i:
                    chunks.append(text[start:i])

        token_ids = []
        
        # Convert each chunk into UTF-8 bytes and merge
        for chunk in chunks:
            if chunk in self.special_byte_tokens:
                token_ids.append(self.token2id[chunk.encode("utf-8")])
            else:
                byte_tokens = [bytes([b]) for b in chunk.encode("utf-8")]
                
                while True:
                    adjacent_pairs = list(zip(byte_tokens, byte_tokens[1:]))
                    best_rank, min_pair_idx = None, None
                    
                    for idx, pair in enumerate(adjacent_pairs):
                        if pair in self.merges_ranks:
                            rank_priority = self.merges_ranks[pair]
                            if best_rank is None or rank_priority < best_rank:
                                best_rank, min_pair_idx = rank_priority, idx
                    
                    if min_pair_idx is None:
                        break
                    
                    left, right = byte_tokens[min_pair_idx], byte_tokens[min_pair_idx + 1]
                    byte_tokens = byte_tokens[:min_pair_idx] + [left + right] + byte_tokens[min_pair_idx + 2:]
                
                token_ids.extend([self.token2id[tok] for tok in byte_tokens])
        
        return token_ids
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for chunk in iterable:
            yield from self.encode(chunk)
    
    def decode(self, ids: List[int]) -> str:
        return b"".join(self.vocab[tid] for tid in ids).decode("utf-8", errors="replace")
    
#Load vocab by file extension    
def load_vocab(vocab_filepath: str) -> Dict[int, bytes]:
        _, ext = os.path.splitext(vocab_filepath)
        if ext == ".json":
            with open(vocab_filepath, "r", encoding="utf-8") as f:
                json_data = json.load(f)
            # Assume JSON format maps token (str) -> index (int)
            vocab = {int(idx): token.encode("utf-8") for token, idx in json_data.items()}
            return vocab
        elif ext == ".pkl":
            with open(vocab_filepath, "rb") as f:
                vocab = pickle.load(f)
            return vocab
        else:
            raise ValueError(f"Unsupported vocab file extension: {ext}")

#Load merges by file extension
def load_merges(merges_filepath: str) -> List[Tuple[bytes, bytes]]:
    """Loads a merges file in a universal way based on file extension."""
    _, ext = os.path.splitext(merges_filepath)
    if ext in [".txt", ".merges"]:
        merges = []
        with open(merges_filepath, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(maxsplit=1)
                if len(parts) == 2:
                    merges.append((parts[0].encode("utf-8"), parts[1].encode("utf-8")))
                else:
                    raise ValueError(f"Invalid merge format: {line}")
        return merges
    elif ext == ".pkl":
        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)
        return merges
    else:
        raise ValueError(f"Unsupported merges file extension: {ext}")
    
#Test Tokenizer
if __name__ == "__main__":
    vocab_filename = "TinyStories_bpe_vocab.pkl"
    merges_filename = "TinyStories_bpe_merges.pkl"
    tokenizer = BPE_Tokenizer.from_files(vocab_filename, merges_filename)
    
    # Test encoding and decoding.
    sample_text = "Tom and Lily were playing with their toys in the living room. They liked to build towers and bridges with their blocks and cars. Tom was very proud of his tall tower. He wanted to make it even taller, so he reached for more blocks.Lily was watching him carefully. She noticed that the tower was starting to wobble. She tried to warn Tom, but it was too late. The tower collapsed, and the blocks scattered everywhere. Tom was disappointed, but Lily gave him a hug and said, \"Don't worry, Tom. We can build an even bigger tower next time.\""
    token_ids = tokenizer.encode(sample_text)
    print("Original text:", sample_text)
    print("Encoded token IDs:", token_ids)
    
    decoded_text = tokenizer.decode(token_ids)
    print("Decoded text:", decoded_text)
