import regex as re
import pickle
import time
import tracemalloc
import os
from collections import defaultdict, Counter
from typing import List, Tuple, Dict
import psutil
import mmap

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

#Create Frequency Table
def build_frequency_table(file_path: str, pattern_str: str, specials: set, chunk_size: int =64 * 10**6) -> Counter:
    frequency_table = Counter()
    combined_pat = re.compile(pattern_str, re.UNICODE | re.DOTALL)
    finditer = combined_pat.finditer  # Cache method lookup

    with open(file_path, 'r', encoding='utf-8') as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            while True:
                chunk = mm.read(chunk_size)
                if not chunk:  # End of file
                    break
                chunk_decoded = chunk.decode('utf-8', errors='ignore')  # Avoid decoding failures
                for match in finditer(chunk_decoded):
                    token = match.group(0)
                    if token not in specials:
                        frequency_table[token.encode('utf-8')] += 1

    return frequency_table
#Compute pair counts
def compute_local_pair_counts(tokens: List[bytes], seq_freq: int) -> Counter:
    counts = Counter()
    for i in range(len(tokens) - 1):
        counts[(tokens[i], tokens[i+1])] += seq_freq
    return counts

def train_bpe(input_path: str, vocab_size: int, special_tokens: List[str]) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    start_time = time.time()
    
    #regex pattern.
    special_pattern = "|".join(map(re.escape, special_tokens)) if special_tokens else ""
    pattern_str = f"({special_pattern})|({PAT})" if special_tokens else PAT
    specials = set(special_tokens)
    
    #Build frequency table.
    t0 = time.perf_counter()
    frequency_table = build_frequency_table(input_path, pattern_str, specials)  # Returns Counter()
    frequency_table_time = time.perf_counter() - t0
    print(f"Frequency Initialization Time: {frequency_table_time:.4f} seconds")
    
    #Initialize vocabulary.
    t0 = time.perf_counter()
    vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    current_id = 256
    for token in special_tokens:
        vocab[current_id] = token.encode('utf-8')
        current_id += 1
    merges: List[Tuple[bytes, bytes]] = []
    vocab_init_time = time.perf_counter() - t0
    print(f"Vocabulary Initialization Time: {vocab_init_time:.4f} seconds")
    
    #Convert frequency_table into a list of sequences with their frequency.
    t0 = time.perf_counter()
    all_sequences = []
    for seq_token, freq in frequency_table.items():
        byte_splits = [seq_token[i:i+1] for i in range(len(seq_token))]
        all_sequences.append({
            'tokens': byte_splits,
            'freq': freq
        })
    #Build per-sequence pair counts and global pair counts.
    sequence_pair_counts = {}
    global_pair_counts = Counter()
    
    for seq_id, entry in enumerate(all_sequences):
        tokens = entry['tokens']
        seq_freq = entry['freq']
        local_counts = compute_local_pair_counts(tokens, seq_freq)
        sequence_pair_counts[seq_id] = local_counts
        for pair, count in local_counts.items():
            global_pair_counts[pair] += count 
    
    t_pair = time.perf_counter() - t0
    print(f"Pair Counts Initialization Time: {t_pair:.4f} seconds")
    
    #BPE Merge Loop 
    t0 = time.perf_counter()
    while len(vocab) < vocab_size and global_pair_counts:
        best_pair = max(global_pair_counts.keys(), key=lambda pair: (global_pair_counts[pair], pair))
        (a, b) = best_pair
        merges.append((a, b))
        merged = a + b 
        vocab[current_id] = merged
        current_id += 1
        
        # Identify sequences affected by the best pair.
        affected_seq_ids = [seq_id for seq_id, counts in sequence_pair_counts.items() if best_pair in counts]
        for seq_id in affected_seq_ids:
            entry = all_sequences[seq_id]
            tokens = entry['tokens']
            seq_freq = entry['freq']
            old_counts = sequence_pair_counts[seq_id]
            
            # Merge all occurrences of (a, b) in this sequence.
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == a and tokens[i+1] == b:
                    new_tokens.append(merged)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            entry['tokens'] = new_tokens
            
            # Recompute pair counts for this sequence.
            new_counts = compute_local_pair_counts(new_tokens, seq_freq)
            sequence_pair_counts[seq_id] = new_counts
            
            #subtract old counts and add new counts.
            for pair, count in old_counts.items():
                global_pair_counts[pair] -= count
                if global_pair_counts[pair] <= 0:
                    del global_pair_counts[pair]
            for pair, count in new_counts.items():
                global_pair_counts[pair] += count
                
    merge_time = time.perf_counter() - t0
    total_time = time.time() - start_time
    print(f"BPE Merging Time (Local Update): {merge_time:.4f} seconds")
    print(f"Total Training Time: {total_time:.2f} seconds")
    
    longest_token = max(vocab.values(), key=len).decode("utf-8", errors="ignore")
    print(f"Longest Token: {longest_token}")
 
    return vocab, merges


if __name__ == "__main__":
    input_file = "../data/owt_train.txt"
    vocab_size = 32000
    special_tokens = ["<|endoftext|>"]
    
   # Start memory tracking
    tracemalloc.start()
    process = psutil.Process(os.getpid())
    start_tracemalloc, _ = tracemalloc.get_traced_memory()
    start_rss = process.memory_info().rss / (1024 * 1024)  # Convert to MB
    vocab, merges = train_bpe(input_file, vocab_size, special_tokens)
    end_tracemalloc, peak_tracemalloc = tracemalloc.get_traced_memory()
    end_rss = process.memory_info().rss / (1024 * 1024)  # Convert to MB
    tracemalloc.stop()
    print(f"Python Memory Usage (Tracked by tracemalloc): {peak_tracemalloc / (1024 * 1024):.2f} MB")
    print(f"Total Process Memory Usage (RSS): {end_rss:.2f} MB")

    with open("Owt_bpe_vocab100.pkl", "wb") as f:
        pickle.dump(vocab, f)
    with open("Owt_bpe_merges100.pkl", "wb") as f:
        pickle.dump(merges, f)