import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):

    def __init__(
        self,
        embedding_dim
    ):
        super().__init__()

        self.query = nn.Linear(
            embedding_dim,
            embedding_dim,
            bias=False
        )

        self.key = nn.Linear(
            embedding_dim,
            embedding_dim,
            bias=False
        )

        self.value = nn.Linear(
            embedding_dim,
            embedding_dim,
            bias=False
        )

    def forward(self, x):

        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        scores = (
            Q @ K.transpose(-2, -1)
        )

        scores = scores / math.sqrt(
            x.size(-1)
        )

        sequence_length = x.size(1)

        mask = torch.tril(
            torch.ones(
                sequence_length,
                sequence_length,
                device=x.device,
                dtype=torch.bool
            )
        )

        scores = scores.masked_fill(
            ~mask,
            float("-inf")
        )

        attention_weights = F.softmax(
            scores,
            dim=-1
        )

        output = (
            attention_weights @ V
        )

        return output


class MultiHeadAttention(nn.Module):

    def __init__(
        self,
        embedding_dim,
        num_heads
    ):
        super().__init__()

        assert embedding_dim % num_heads == 0

        self.num_heads = num_heads

        self.head_dim = (
            embedding_dim // num_heads
        )

        # ====================================
        # QKV Projection
        # ====================================

        self.query = nn.Linear(
            embedding_dim,
            embedding_dim,
            bias=False
        )

        self.key = nn.Linear(
            embedding_dim,
            embedding_dim,
            bias=False
        )

        self.value = nn.Linear(
            embedding_dim,
            embedding_dim,
            bias=False
        )

        # ====================================
        # Output Projection
        # ====================================

        self.output_projection = nn.Linear(
            embedding_dim,
            embedding_dim
        )

    def forward(self, x):

        (
            batch_size,
            sequence_length,
            embedding_dim
        ) = x.shape

        # ====================================
        # Q K V
        # ====================================

        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # ====================================
        # Split Heads
        # ====================================

        Q = Q.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        K = K.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        V = V.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        # ====================================
        # Attention Scores
        # ====================================

        scores = (
            Q @ K.transpose(-2, -1)
        )

        scores = scores / math.sqrt(
            self.head_dim
        )

        # ====================================
        # Causal Mask
        # ====================================

        causal_mask = torch.tril(
            torch.ones(
                sequence_length,
                sequence_length,
                device=x.device,
                dtype=torch.bool
            )
        )

        scores = scores.masked_fill(
            ~causal_mask,
            float("-inf")
        )

        # ====================================
        # Softmax
        # ====================================

        weights = F.softmax(
            scores,
            dim=-1
        )

        # ====================================
        # Weighted Values
        # ====================================

        output = (
            weights @ V
        )

        # ====================================
        # Merge Heads
        # ====================================

        output = output.transpose(
            1,
            2
        ).contiguous()

        output = output.view(
            batch_size,
            sequence_length,
            embedding_dim
        )

        # ====================================
        # Output Projection
        # ====================================

        output = self.output_projection(
            output
        )

        return output