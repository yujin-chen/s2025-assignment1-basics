import regex as re
from collections import defaultdict
from typing import List, Tuple, Dict

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def train_bpe(input_path: str, vocab_size: int, special_tokens: List[str]) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()

    #Pre-tokenization
    special_pattern = "|".join(map(re.escape, special_tokens))
    combined_pat = (
        re.compile(f"({special_pattern})|({PAT})", re.UNICODE | re.DOTALL)
        if special_tokens
        else re.compile(PAT, re.UNICODE | re.DOTALL)
    )

    #Create frequency table
    frequency_table = defaultdict(int)
    specials = set(special_tokens)
    for match in combined_pat.finditer(text):
        token = next(group for group in match.groups() if group is not None)
        if token in specials:
            continue
        byte_seq = tuple(bytes([b]) for b in token.encode('utf-8'))
        frequency_table[byte_seq] += 1

    #Initialize vocabulary and merges
    vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    current_id = 256
    for token in special_tokens:
        vocab[current_id] = token.encode('utf-8')
        current_id += 1

    merges: List[Tuple[bytes, bytes]] = []

    #Build initial pair counts
    pair_counts = defaultdict(int)
    for seq, count in frequency_table.items():
        for i in range(len(seq) - 1):
            pair_counts[(seq[i], seq[i+1])] += count

    #Merge until reach vocab size or no more pairs to merge
    while len(vocab) < vocab_size and pair_counts:
        # Select the most frequent pair
        best_pair = max(pair_counts.keys(), key=lambda pair: (pair_counts[pair], isinstance(pair, tuple), pair))
        a, b = best_pair
        merged = a + b
        merges.append(best_pair)

        #Add merged token to vocabulary
        vocab[current_id] = merged
        current_id += 1

        #Update frequency_table and pair_counts incrementally
        new_entries = defaultdict(int)
        for seq in list(frequency_table.keys()):
            count = frequency_table[seq]
            new_seq = []
            i = 0

            while i < len(seq):
                if i < len(seq) - 1 and seq[i] == a and seq[i + 1] == b:
                    #Handle adjacent pairs
                    prev = seq[i - 1] if i > 0 else None
                    next_token = seq[i + 2] if i + 2 < len(seq) else None

                    #Decrement old pairs
                    if prev is not None:
                        old_pair = (prev, a)
                        pair_counts[old_pair] -= count
                        if pair_counts[old_pair] == 0:
                            del pair_counts[old_pair]

                    old_pair = (a, b)
                    pair_counts[old_pair] -= count
                    if pair_counts[old_pair] == 0:
                        del pair_counts[old_pair]

                    if next_token is not None:
                        old_pair = (b, next_token)
                        pair_counts[old_pair] -= count
                        if pair_counts[old_pair] == 0:
                            del pair_counts[old_pair]

                    #Add new pairs
                    new_token = merged
                    if prev is not None:
                        new_pair = (prev, new_token)
                        pair_counts[new_pair] += count

                    if next_token is not None:
                        new_pair = (new_token, next_token)
                        pair_counts[new_pair] += count

                    new_seq.append(merged)
                    i += 2
                else:
                    new_seq.append(seq[i])
                    i += 1

            #Replace old sequences with their merged versions in the frequency table
            #Add new merged sequences to new_entries
            new_seq_tuple = tuple(new_seq)
            if new_seq_tuple != seq:
                new_entries[new_seq_tuple] += count
                del frequency_table[seq]

        #Update frequency_table with all merged sequences in new_entries
        for seq, count in new_entries.items():
            frequency_table[seq] += count

    return vocab, merges