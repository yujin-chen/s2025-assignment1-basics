import torch
from torch.nn import Module, Linear
from ece496b_basics import Scaled_Dot_Product_Attention

class MultiHeadSelfAttention(Module):
    def __init__(self, d_model, num_heads, attn_pdrop=None, weights=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads  

        assert self.d_k * num_heads == d_model, "d_model must be divisible by num_heads"
        assert self.d_v * num_heads == d_model, "d_model must be divisible by num_heads"

        self.Wq = Linear(d_model, d_model, bias=False)
        self.Wk = Linear(d_model, d_model, bias=False)
        self.Wv = Linear(d_model, d_model, bias=False)
        self.Wo = Linear(d_model, d_model, bias=False)

        self.attention = Scaled_Dot_Product_Attention.ScaledDotProductAttention(attn_pdrop)

        if weights is not None:
            # Collect only available weights
            q_heads = torch.cat(
                [weights[f"q_heads.{i}.weight"] for i in range(num_heads) if f"q_heads.{i}.weight" in weights],
                dim=0
            ) if any(f"q_heads.{i}.weight" in weights for i in range(num_heads)) else None

            k_heads = torch.cat(
                [weights[f"k_heads.{i}.weight"] for i in range(num_heads) if f"k_heads.{i}.weight" in weights],
                dim=0
            ) if any(f"k_heads.{i}.weight" in weights for i in range(num_heads)) else None

            v_heads = torch.cat(
                [weights[f"v_heads.{i}.weight"] for i in range(num_heads) if f"v_heads.{i}.weight" in weights],
                dim=0
            ) if any(f"v_heads.{i}.weight" in weights for i in range(num_heads)) else None

            output_proj = weights.get("output_proj.weight", None)

            # Assign weights only if they exist
            if q_heads is not None:
                self.Wq.weight.data = q_heads
            if k_heads is not None:
                self.Wk.weight.data = k_heads
            if v_heads is not None:
                self.Wv.weight.data = v_heads
            if output_proj is not None:
                self.Wo.weight.data = output_proj

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()

        # Project Q, K, V
        q = self.Wq(x)
        k = self.Wk(x)
        v = self.Wv(x)

        q = q.view(batch_size, seq_len, self.num_heads, self.d_k).permute(0, 2, 1, 3)
        k = k.view(batch_size, seq_len, self.num_heads, self.d_k).permute(0, 2, 1, 3)
        v = v.view(batch_size, seq_len, self.num_heads, self.d_v).permute(0, 2, 1, 3)

        # Create causal mask
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device),
            diagonal=1
        ).unsqueeze(0) 

        # Apply attention
        attn_output = self.attention(q, k, v, mask=causal_mask)

        # Reshape back
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, self.d_model)

        # Output projection
        return self.Wo(attn_output)
