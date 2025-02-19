import argparse
import torch
import tiktoken
from Transformer_LM import TransformerLM
from checkpoint import load_checkpoint
from decode import decode  
from BPE_Tokenizer import BPE_Tokenizer
from AdamW import AdamW

def main():
    # Instantiate your tokenizers
    # Tiny_Tokenizer = BPE_Tokenizer.from_files(
    #     "Tiny_bpe_vocab1.pkl",
    #     "Tiny_bpe_merges1.pkl",
    #     ["<|endoftext|>"]
    # )
    tik_encode = tiktoken.get_encoding("gpt2")


    parser = argparse.ArgumentParser(description="Generate text from a trained model.")
    parser.add_argument("--checkpoint_path", type=str, default="my_checkpoint.pt",
                        help="Path to the checkpoint file saved by Training.py")
    parser.add_argument("--vocab_size", type=int, required=True)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--d_ff", type=int, default=2048)
    parser.add_argument("--attn_pdrop", type=float, default=0.1)
    parser.add_argument("--residual_pdrop", type=float, default=0.1)

    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # Decoding params
    parser.add_argument(
        "--prompt",
        type=str,
        default="Once upon a time, in a warm and sunny place, there was a big pit. "
                "A little boy named Tom liked to play near the pit. One day, Tom lost his red ball. "
                "He was very sad.",
        help="Raw text prompt to be tokenized."
    )
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--end_token_id", type=int, help="Stop if this token is generated")

    args = parser.parse_args()

    # Assign default after parsing, but only if user did not provide a value
    if args.end_token_id is None:
        # args.end_token_id = Tiny_Tokenizer.encode("<|endoftext|>")
        args.end_token_id = tik_encode.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})

    #Initiate model structure
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        attn_pdrop=args.attn_pdrop,
        residual_pdrop=args.residual_pdrop,
        weights={}
    ).to(args.device)

    # Load from checkpoint
    optimizer = AdamW(model.parameters(), lr=1e-4)

    start_iter = load_checkpoint(args.checkpoint_path, model, optimizer)
    print(f"Loaded checkpoint from iteration {start_iter}")

    model.eval()

    #Convert the user prompt to token IDs
    if args.prompt.strip():
        # prompt_tokens = Tiny_Tokenizer.encode(args.prompt.strip())
        prompt_tokens = tik_encode.encode(args.prompt.strip(), allowed_special={"<|endoftext|>"})
    else:   
        prompt_tokens = []

    #Decode
    out_tokens = decode(
        model,
        prompt_tokens,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        device=args.device
    )

    print("Generated tokens:", out_tokens)
    # print("Generated text:", Tiny_Tokenizer.decode(out_tokens))
    print("Generated text:", tik_encode.decode(out_tokens))


if __name__ == "__main__":
    main()
