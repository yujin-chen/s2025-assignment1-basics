from torch.nn import Module, Dropout
from ece496b_basics import RMSNorm, positionwise_feedforward, Multihead_Self_Attention

class TransformerBlock(Module):
    def __init__(self, d_model, num_heads, d_ff, attn_pdrop, residual_pdrop, weights=None):
        super().__init__()

        ln1_w = None
        ln2_w = None
        attn_weights = {}
        ffn_weights = {}

        if weights is not None:
            ln1_w = weights.get("ln1.weight", None)
            ln2_w = weights.get("ln2.weight", None)
            
            # For attention
            attn_weights = {
                key.replace("attn.", ""): val
                for key, val in weights.items()
                if key.startswith("attn.")
            }
            # For feed-forward
            ffn_weights = {
                key.replace("ffn.", ""): val
                for key, val in weights.items()
                if key.startswith("ffn.")
            }

        self.norm1 = RMSNorm.RMSNorm(
            d_model, 
            eps=1e-5,
            weights={"weight": ln1_w} if ln1_w is not None else None
        )

        self.attn = Multihead_Self_Attention.MultiHeadSelfAttention(
            d_model,
            num_heads,
            attn_pdrop,
            attn_weights
        )
        self.dropout1 = Dropout(residual_pdrop)

        self.norm2 = RMSNorm.RMSNorm(
            d_model,
            eps=1e-5,
            weights={"weight": ln2_w} if ln2_w is not None else None
        )
        self.ffn = positionwise_feedforward.PositionwiseFeedForward(
            d_model,
            d_ff,
            ffn_weights  
        )
        self.dropout2 = Dropout(residual_pdrop)

    def forward(self, x, mask=None):
            x = x + self.dropout1(self.attn(self.norm1(x), mask))
            x = x + self.dropout2(self.ffn(self.norm2(x)))
            return x
    #Ablation study revert norm
    # def forward(self, x, mask=None):
    #     # Compute self-attention on the raw input x, add residual, then apply RMSNorm.
    #     z = self.norm1(x + self.dropout1(self.attn(x, mask)))
    #     # Compute feed-forward on z, add residual, then apply RMSNorm.
    #     y = self.norm2(z + self.dropout2(self.ffn(z)))
    #     return y
    #Ablation no norm
    # def forward(self, x, mask=None):
    #     x = x + self.dropout1(self.attn(x, mask))
    #     x = x + self.dropout2(self.ffn(x))
    #     return x
    #ablation study parallel
    # def forward(self, x, mask=None):
    #     y = x + self.dropout1(self.attn(self.norm1(x), mask)) + self.dropout2(self.ffn(self.norm2(x)))
    #     return y
