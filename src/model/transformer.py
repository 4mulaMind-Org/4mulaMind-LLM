import torch.nn as nn

from src.model.attention import MultiHeadAttention
from src.model.feed_forward import FeedForward


class TransformerBlock(nn.Module):

    def __init__(
        self,
        embedding_dim,
        num_heads,
        hidden_dim
    ):
        super().__init__()

        # ====================================
        # Layer Normalization
        # ====================================

        self.layer_norm_1 = nn.LayerNorm(
            embedding_dim
        )

        self.layer_norm_2 = nn.LayerNorm(
            embedding_dim
        )

        # ====================================
        # Self Attention
        # ====================================

        self.attention = MultiHeadAttention(
            embedding_dim,
            num_heads
        )

        # ====================================
        # Feed Forward
        # ====================================

        self.feed_forward = FeedForward(
            embedding_dim,
            hidden_dim
        )

    def forward(self, x):

        # ====================================
        # Pre-LN Attention
        # ====================================

        normalized_x = self.layer_norm_1(
            x
        )

        attention_output = self.attention(
            normalized_x
        )

        # Residual connection

        x = x + attention_output

        # ====================================
        # Pre-LN Feed Forward
        # ====================================

        normalized_x = self.layer_norm_2(
            x
        )

        feed_forward_output = (
            self.feed_forward(
                normalized_x
            )
        )

        # Residual connection

        x = x + feed_forward_output

        return x