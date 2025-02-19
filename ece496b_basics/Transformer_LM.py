import torch
from torch.nn import Module, ModuleList, Embedding, Dropout, Linear
from ece496b_basics import Transformer_Block, RMSNorm


class TransformerLM(Module):
    def __init__(self, vocab_size, context_length, d_model, num_layers, num_heads, d_ff, attn_pdrop, residual_pdrop, weights):
        super().__init__()
        self.context_length = context_length
        self.residual_pdrop = residual_pdrop

        self.token_embeddings = Embedding(vocab_size, d_model)
        self.position_embeddings = Embedding(context_length, d_model)
        self.dropout = Dropout(residual_pdrop)

        # Initialize Transformer Blocks
        self.layers = ModuleList([
            Transformer_Block.TransformerBlock(
                d_model, num_heads, d_ff, attn_pdrop, residual_pdrop,
                {key.replace(f"layers.{i}.", ""): val for key, val in weights.items() if key.startswith(f"layers.{i}.")}
            ) for i in range(num_layers)
        ])

        self.ln_final = RMSNorm.RMSNorm(
            d_model, eps=1e-5, weights={"weight": weights.get("ln_final.weight", torch.ones(d_model))}
        )
        self.lm_head = Linear(d_model, vocab_size, bias=False)

        self.load_weights(weights)

    def load_weights(self, weights):
        """Load pre-trained weights into the model."""
        def copy_weight(param, key):
            if key in weights:
                param.data.copy_(weights[key])

        copy_weight(self.token_embeddings.weight, 'token_embeddings.weight')
        copy_weight(self.position_embeddings.weight, 'position_embeddings.weight')

        for i, layer in enumerate(self.layers):
            prefix = f'layers.{i}.'

            # Attention weights
            copy_weight(layer.attn.Wq.weight, f'{prefix}attn.q_proj.weight')
            copy_weight(layer.attn.Wk.weight, f'{prefix}attn.k_proj.weight')
            copy_weight(layer.attn.Wv.weight, f'{prefix}attn.v_proj.weight')
            copy_weight(layer.attn.Wo.weight, f'{prefix}attn.output_proj.weight')

            # Normalization & Feedforward weights
            copy_weight(layer.norm1.weight, f'{prefix}ln1.weight')
            copy_weight(layer.ffn.fc1.weight, f'{prefix}ffn.w1.weight')
            copy_weight(layer.ffn.fc2.weight, f'{prefix}ffn.w2.weight')

        copy_weight(self.ln_final.weight, "ln_final.weight")
        copy_weight(self.lm_head.weight, "lm_head.weight")

    def forward(self, in_indices):
        """Forward pass for Transformer LM."""
        batch_size, seq_len = in_indices.size()
        if seq_len > self.context_length:
            raise ValueError(f"Sequence length {seq_len} exceeds context_length {self.context_length}")

        token_emb = self.token_embeddings(in_indices)
        positions = torch.arange(seq_len, device=in_indices.device).unsqueeze(0).expand(batch_size, seq_len)
        pos_emb = self.position_embeddings(positions)

        x = self.dropout(token_emb + pos_emb)

        for layer in self.layers:
            x = layer(x)

        return self.lm_head(self.ln_final(x))
