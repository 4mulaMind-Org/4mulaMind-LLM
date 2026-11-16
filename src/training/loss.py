import torch
import torch.nn as nn


class LanguageModelLoss:

    def __init__(self):
        self.loss_function = nn.CrossEntropyLoss()

    def __call__(self, logits, targets):

        batch_size, sequence_length, vocab_size = logits.shape

        # [B, T, V]
        #      ↓
        # [B*T, V]

        logits = logits.view(
            batch_size * sequence_length,
            vocab_size
        )

        # [B, T]
        #   ↓
        # [B*T]

        targets = targets.view(
            batch_size * sequence_length
        )

        loss = self.loss_function(
            logits,
            targets
        )

        return loss