import json
import os
from array import array
from bisect import bisect_right
from collections import OrderedDict

import torch
from torch.utils.data import Dataset


class PacketDataset(Dataset):
    """
    Packet-backed dataset for next-token prediction.

    Each packet contains token IDs stored as uint16.

    By default, samples use:
        block_size = 128
        stride = 128

    This avoids the huge overlap produced by using
    stride=1 sliding windows.
    """

    def __init__(
        self,
        packet_dir,
        block_size=128,
        stride=None,
        cache_size=16
    ):
        super().__init__()

        self.packet_dir = packet_dir
        self.block_size = block_size

        # --------------------------------
        # Non-overlapping input windows
        # --------------------------------

        if stride is None:
            stride = block_size

        if stride <= 0:
            raise ValueError(
                "stride must be greater than zero."
            )

        self.stride = stride

        self.cache_size = cache_size

        # --------------------------------
        # Manifest
        # --------------------------------

        self.manifest_path = os.path.join(
            packet_dir,
            "manifest.json"
        )

        if not os.path.exists(
            self.manifest_path
        ):
            raise FileNotFoundError(
                f"Manifest not found: "
                f"{self.manifest_path}"
            )

        with open(
            self.manifest_path,
            "r",
            encoding="utf-8"
        ) as file:

            self.manifest = json.load(file)

        # --------------------------------
        # Packet metadata
        # --------------------------------

        self.packet_info = self.manifest.get(
            "packets",
            []
        )

        if not self.packet_info:
            raise RuntimeError(
                f"No packets found in "
                f"{self.manifest_path}"
            )

        # --------------------------------
        # Cumulative sample counts
        # --------------------------------

        self.cumulative_samples = []

        total_samples = 0

        for packet in self.packet_info:

            token_count = int(
                packet["token_count"]
            )

            # Need block_size + 1 tokens:
            #
            # X = first block_size tokens
            # Y = next block_size tokens
            #
            available = (
                token_count
                - block_size
            )

            if available > 0:

                packet_samples = (
                    available
                    + self.stride
                    - 1
                ) // self.stride

            else:

                packet_samples = 0

            total_samples += packet_samples

            self.cumulative_samples.append(
                total_samples
            )

        self.total_samples = total_samples

        if self.total_samples <= 0:

            raise RuntimeError(
                "Packets do not contain enough "
                "tokens for the selected block size."
            )

        # --------------------------------
        # LRU cache
        # --------------------------------

        self.cache = OrderedDict()

    # ====================================
    # Length
    # ====================================

    def __len__(self):

        return self.total_samples

    # ====================================
    # Load Packet
    # ====================================

    def _load_packet(
        self,
        packet_index
    ):

        # Cache hit
        if packet_index in self.cache:

            tokens = self.cache.pop(
                packet_index
            )

            self.cache[packet_index] = (
                tokens
            )

            return tokens

        info = self.packet_info[
            packet_index
        ]

        filename = info["file"]

        packet_path = os.path.join(
            self.packet_dir,
            filename
        )

        if not os.path.exists(
            packet_path
        ):
            raise FileNotFoundError(
                f"Packet not found: "
                f"{packet_path}"
            )

        expected_count = int(
            info["token_count"]
        )

        tokens = array("H")

        with open(
            packet_path,
            "rb"
        ) as file:

            tokens.fromfile(
                file,
                expected_count
            )

        if len(tokens) != expected_count:

            raise RuntimeError(
                f"Packet size mismatch: "
                f"{packet_path}. "
                f"Expected {expected_count}, "
                f"got {len(tokens)}."
            )

        self.cache[packet_index] = tokens

        while len(self.cache) > self.cache_size:

            self.cache.popitem(
                last=False
            )

        return tokens

    # ====================================
    # Locate Sample
    # ====================================

    def _locate(
        self,
        index
    ):

        if index < 0:
            index += self.total_samples

        if (
            index < 0
            or index >= self.total_samples
        ):
            raise IndexError(
                "Dataset index out of range."
            )

        packet_index = bisect_right(
            self.cumulative_samples,
            index
        )

        previous_total = 0

        if packet_index > 0:

            previous_total = (
                self.cumulative_samples[
                    packet_index - 1
                ]
            )

        local_sample_index = (
            index
            - previous_total
        )

        local_start = (
            local_sample_index
            * self.stride
        )

        return (
            packet_index,
            local_start
        )

    # ====================================
    # Get Item
    # ====================================

    def __getitem__(
        self,
        index
    ):

        (
            packet_index,
            start
        ) = self._locate(index)

        tokens = self._load_packet(
            packet_index
        )

        end = (
            start
            + self.block_size
            + 1
        )

        sequence = tokens[
            start:end
        ]

        # --------------------------------
        # Do not cross packet boundaries
        # --------------------------------

        if len(sequence) != (
            self.block_size + 1
        ):
            raise RuntimeError(
                "Insufficient tokens for "
                "training sample."
            )

        x = torch.tensor(
            sequence[:-1],
            dtype=torch.long
        )

        y = torch.tensor(
            sequence[1:],
            dtype=torch.long
        )

        return x, y

    # ====================================
    # Information
    # ====================================

    def info(self):

        return {
            "packet_dir": self.packet_dir,
            "packet_count": len(
                self.packet_info
            ),
            "total_samples": self.total_samples,
            "block_size": self.block_size,
            "stride": self.stride,
            "packet_size": self.manifest.get(
                "packet_size"
            ),
            "vocab_size": self.manifest.get(
                "vocab_size"
            ),
            "documents": self.manifest.get(
                "documents"
            ),
            "tokens": self.manifest.get(
                "tokens"
            )
        }