import torch
import torch.nn as nn


class FourmulaMindModel(nn.Module):

    def __init__(
        self,
        vocab_size,
        block_size=8,
        embedding_dim=32
    ):
        super().__init__()

        # Token embedding
        self.token_embedding = nn.Embedding(
            vocab_size,
            embedding_dim
        )

        # Position embedding
        self.position_embedding = nn.Embedding(
            block_size,
            embedding_dim
        )

    def forward(self, x):

        batch_size, sequence_length = x.shape

        # Token vectors
        token_embeddings = self.token_embedding(x)

        # Positions: 0, 1, 2, ..., sequence_length-1
        positions = torch.arange(
            sequence_length,
            device=x.device
        )

        # Position vectors
        position_embeddings = self.position_embedding(
            positions
        )

        # Combine token + position information
        embeddings = (
            token_embeddings
            + position_embeddings
        )

        return embeddings