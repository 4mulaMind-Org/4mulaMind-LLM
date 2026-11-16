import torch
import torch.nn as nn

from src.model.transformer import TransformerBlock


class FourmulaMindLM(nn.Module):

    def __init__(
        self,
        vocab_size,
        block_size,
        embedding_dim=64,
        num_heads=4,
        hidden_dim=256,
        num_layers=4
    ):
        super().__init__()

        self.block_size = block_size

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # ====================================
        # Token Embedding
        # ====================================

        self.token_embedding = nn.Embedding(
            vocab_size,
            embedding_dim
        )

        # ====================================
        # Position Embedding
        # ====================================

        self.position_embedding = nn.Embedding(
            block_size,
            embedding_dim
        )

        # ====================================
        # Transformer Layers
        # ====================================

        self.transformer_layers = nn.ModuleList(

            [
                TransformerBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    hidden_dim=hidden_dim
                )

                for _ in range(num_layers)
            ]
        )

        # ====================================
        # Final Layer Normalization
        # ====================================

        self.final_layer_norm = nn.LayerNorm(
            embedding_dim
        )

        # ====================================
        # Language Model Head
        # ====================================

        self.lm_head = nn.Linear(
            embedding_dim,
            vocab_size
        )

    # ====================================
    # Forward
    # ====================================

    def forward(self, x):

        batch_size, sequence_length = x.shape

        # ====================================
        # Token Embeddings
        # ====================================

        token_embeddings = self.token_embedding(x)

        # ====================================
        # Position IDs
        # ====================================

        positions = torch.arange(
            sequence_length,
            device=x.device
        )

        # ====================================
        # Position Embeddings
        # ====================================

        position_embeddings = self.position_embedding(
            positions
        )

        # ====================================
        # Combine Embeddings
        # ====================================

        x = (
            token_embeddings
            + position_embeddings
        )

        # ====================================
        # Transformer Layers
        # ====================================

        for layer in self.transformer_layers:

            x = layer(x)

        # ====================================
        # Final Normalization
        # ====================================

        x = self.final_layer_norm(x)

        # ====================================
        # Language Model Head
        # ====================================

        logits = self.lm_head(x)

        return logits

