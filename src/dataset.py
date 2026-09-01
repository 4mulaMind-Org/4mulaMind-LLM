import torch


class TextDataset:

    def __init__(
        self,
        tokens,
        block_size=8
    ):
        self.tokens = tokens
        self.block_size = block_size

    def __len__(self):
        return len(self.tokens) - self.block_size

    def __getitem__(self, index):

        x = self.tokens[
            index:index + self.block_size
        ]

        y = self.tokens[
            index + 1:index + self.block_size + 1
        ]

        return (
            torch.tensor(x, dtype=torch.long),
            torch.tensor(y, dtype=torch.long)
        )