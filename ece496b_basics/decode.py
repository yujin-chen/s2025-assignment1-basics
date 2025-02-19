import torch
from Softmax import softmax
from BPE_Tokenizer import BPE_Tokenizer
import tiktoken

def decode(model, prompt_tokens, max_new_tokens, temperature, top_p, device):
    model.eval()
    generated = prompt_tokens[:]  
    context = torch.tensor([generated], dtype=torch.long, device=device)
    context_length = model.context_length  

    # Load the tokenizer with the special token registered
    Tiny_Tokenizer = BPE_Tokenizer.from_files(
        "Tiny_bpe_vocab1.pkl",
        "Tiny_bpe_merges1.pkl",
        special_byte_tokens=["<|endoftext|>"]
    )
    tik_encode = tiktoken.get_encoding("gpt2")

    for _ in range(max_new_tokens):
        if context.size(1) > context_length:
            context = context[:, -context_length:]

        with torch.no_grad():
            logits = model(context)
        next_token_logits = logits[0, -1, :]

        if temperature > 0.0:
            next_token_logits /= temperature

        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
        cumulative_probs = torch.cumsum(softmax(sorted_logits, dim=-1), dim=-1)
        cutoff_index = (cumulative_probs > top_p).nonzero(as_tuple=True)[0]
        if len(cutoff_index) > 0:
            sorted_logits = sorted_logits[:cutoff_index[0] + 1]
            sorted_indices = sorted_indices[:cutoff_index[0] + 1]
        probs = softmax(sorted_logits, dim=-1)

        next_token_id = sorted_indices[torch.multinomial(probs, num_samples=1)].item()
        generated.append(next_token_id)

        # Decode the entire generated sequence 
        # decoded_generated = Tiny_Tokenizer.decode(generated)
        # if "<|endoftext|>" in decoded_generated:
        #     final_output = decoded_generated.split("<|endoftext|>")[0]
        #     return Tiny_Tokenizer.encode(final_output) 
        # # Decode the entire generated sequence 
        decoded_generated = tik_encode.decode(generated)
        if "<|endoftext|>" in decoded_generated:
            final_output = decoded_generated.split("<|endoftext|>")[0]
            return tik_encode.encode(final_output) 

        context = torch.tensor([generated], dtype=torch.long, device=device)

    return generated
