from collections import Counter
import heapq
import json


class Tokenizer:

    # ====================================
    # Special Tokens
    # ====================================

    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    BOS_TOKEN = "<BOS>"
    EOS_TOKEN = "<EOS>"

    # ====================================
    # Initialization
    # ====================================

    def __init__(self, text):

        # --------------------------------
        # Special tokens
        # --------------------------------

        self.pad_token = self.PAD_TOKEN
        self.unk_token = self.UNK_TOKEN
        self.bos_token = self.BOS_TOKEN
        self.eos_token = self.EOS_TOKEN

        # --------------------------------
        # Character vocabulary
        # --------------------------------

        self.chars = sorted(
            list(
                set(text)
            )
        )

        # --------------------------------
        # Remove special token strings
        # --------------------------------

        self.chars = [
            char
            for char in self.chars
            if char not in {
                self.PAD_TOKEN,
                self.UNK_TOKEN,
                self.BOS_TOKEN,
                self.EOS_TOKEN
            }
        ]

        # --------------------------------
        # Character mappings
        # --------------------------------

        self.char_to_id = {}

        self.id_to_char = {}

        # --------------------------------
        # Initialize special tokens
        # --------------------------------

        self._initialize_special_tokens()

        # --------------------------------
        # Add characters
        # --------------------------------

        for char in self.chars:

            self._add_character(
                char
            )

        # --------------------------------
        # BPE vocabulary
        # --------------------------------

        self.token_to_id = {}

        self.id_to_token = {}

        # --------------------------------
        # Copy special tokens
        # --------------------------------

        for token in [
            self.pad_token,
            self.unk_token,
            self.bos_token,
            self.eos_token
        ]:

            token_id = (
                self.char_to_id[token]
            )

            self.token_to_id[token] = (
                token_id
            )

            self.id_to_token[token_id] = (
                token
            )

        # --------------------------------
        # Copy character tokens
        # --------------------------------

        for char in self.chars:

            token_id = (
                self.char_to_id[char]
            )

            self.token_to_id[char] = (
                token_id
            )

            self.id_to_token[token_id] = (
                char
            )

        # --------------------------------
        # BPE merge rules
        # --------------------------------

        self.merges = []

        # --------------------------------
        # Vocabulary size
        # --------------------------------

        self.vocab_size = (
            len(self.token_to_id)
        )

    # ====================================
    # Initialize Special Tokens
    # ====================================

    def _initialize_special_tokens(self):

        special_tokens = [
            self.pad_token,
            self.unk_token,
            self.bos_token,
            self.eos_token
        ]

        for token_id, token in enumerate(
            special_tokens
        ):

            self.char_to_id[token] = (
                token_id
            )

            self.id_to_char[token_id] = (
                token
            )

    # ====================================
    # Add Character
    # ====================================

    def _add_character(self, char):

        if char in self.char_to_id:

            return self.char_to_id[
                char
            ]

        token_id = len(
            self.char_to_id
        )

        self.char_to_id[char] = (
            token_id
        )

        self.id_to_char[token_id] = (
            char
        )

        return token_id

    # ====================================
    # Special Token IDs
    # ====================================

    @property
    def pad_token_id(self):

        return self.token_to_id[
            self.pad_token
        ]

    @property
    def unk_token_id(self):

        return self.token_to_id[
            self.unk_token
        ]

    @property
    def bos_token_id(self):

        return self.token_to_id[
            self.bos_token
        ]

    @property
    def eos_token_id(self):

        return self.token_to_id[
            self.eos_token
        ]

    # ====================================
    # Character Encode
    # ====================================

    def encode_chars(self, text):

        token_ids = []

        for char in text:

            if char in self.char_to_id:

                token_ids.append(
                    self.char_to_id[char]
                )

            else:

                token_ids.append(
                    self.unk_token_id
                )

        return token_ids

    # ====================================
    # Character Decode
    # ====================================

    def decode_chars(self, token_ids):

        result = []

        for token_id in token_ids:

            if token_id not in self.id_to_char:

                continue

            token = self.id_to_char[
                token_id
            ]

            # --------------------------------
            # Skip special tokens
            # --------------------------------

            if token in {
                self.pad_token,
                self.bos_token,
                self.eos_token
            }:

                continue

            # --------------------------------
            # Unknown token
            # --------------------------------

            if token == self.unk_token:

                result.append("?")

            else:

                result.append(token)

        return "".join(result)

    # ====================================
    # Count Adjacent Pairs
    # ====================================

    def count_pairs(self, token_ids):

        pairs = Counter()

        for i in range(
            len(token_ids) - 1
        ):

            pair = (
                token_ids[i],
                token_ids[i + 1]
            )

            pairs[pair] += 1

        return pairs

    # ====================================
    # Merge Pair
    # ====================================

    def merge_pair(
        self,
        token_ids,
        pair,
        new_token_id
    ):

        merged = []

        i = 0

        while i < len(token_ids):

            # --------------------------------
            # Check current + next token
            # --------------------------------

            if (
                i < len(token_ids) - 1
                and token_ids[i]
                == pair[0]
                and token_ids[i + 1]
                == pair[1]
            ):

                merged.append(
                    new_token_id
                )

                i += 2

            else:

                merged.append(
                    token_ids[i]
                )

                i += 1

        return merged

    # ====================================
    # Add BPE Token
    # ====================================

    def add_token(self, token):

        # --------------------------------
        # Token already exists
        # --------------------------------

        if token in self.token_to_id:

            return self.token_to_id[
                token
            ]

        # --------------------------------
        # New token ID
        # --------------------------------

        token_id = self.vocab_size

        self.token_to_id[token] = (
            token_id
        )

        self.id_to_token[token_id] = (
            token
        )

        # --------------------------------
        # Keep character mapping synced
        # --------------------------------

        self.id_to_char[token_id] = (
            token
        )

        self.vocab_size += 1

        return token_id

    # ====================================
    # Train BPE
    # ====================================

    def train_bpe(
        self,
        text,
        num_merges=50,
        max_vocab_size=128
    ):

        print()
        print("================================")
        print("Training BPE")
        print("================================")

        # --------------------------------
        # Start with character IDs
        # --------------------------------

        token_ids = self.encode_chars(
            text
        )

        # --------------------------------
        # Reset merge rules
        # --------------------------------

        self.merges = []

        # --------------------------------
        # Perform merges
        # --------------------------------

        for merge_number in range(
            num_merges
        ):

            # --------------------------------
            # Vocabulary limit
            # --------------------------------

            if (
                max_vocab_size is not None
                and self.vocab_size
                >= max_vocab_size
            ):

                print(
                    "\nMaximum vocabulary size reached."
                )

                break

            # --------------------------------
            # Count pairs
            # --------------------------------

            pair_counts = self.count_pairs(
                token_ids
            )

            if not pair_counts:

                break

            # --------------------------------
            # Find most frequent pair
            # --------------------------------

            best_pair, frequency = (
                pair_counts.most_common(1)[0]
            )

            # --------------------------------
            # Make sure tokens exist
            # --------------------------------

            if (
                best_pair[0]
                not in self.id_to_token
                or
                best_pair[1]
                not in self.id_to_token
            ):

                break

            token_a = self.id_to_token[
                best_pair[0]
            ]

            token_b = self.id_to_token[
                best_pair[1]
            ]

            # --------------------------------
            # Never merge special tokens
            # --------------------------------

            special_tokens = {
                self.pad_token,
                self.unk_token,
                self.bos_token,
                self.eos_token
            }

            if (
                token_a in special_tokens
                or
                token_b in special_tokens
            ):

                pair_counts.pop(
                    best_pair,
                    None
                )

                if not pair_counts:

                    break

                continue

            # --------------------------------
            # Create new token
            # --------------------------------

            new_token = (
                token_a + token_b
            )

            # --------------------------------
            # Add token
            # --------------------------------

            new_token_id = self.add_token(
                new_token
            )

            # --------------------------------
            # Save merge rule
            # --------------------------------

            self.merges.append(
                (
                    best_pair[0],
                    best_pair[1]
                )
            )

            # --------------------------------
            # Display
            # --------------------------------

            print(
                f"Merge {merge_number + 1:04d}: "
                f"'{token_a}' + '{token_b}' "
                f"-> '{new_token}' "
                f"| frequency={frequency}"
            )

            # --------------------------------
            # Apply merge
            # --------------------------------

            token_ids = self.merge_pair(
                token_ids,
                best_pair,
                new_token_id
            )

        print()

        print(
            f"Final vocabulary size: "
            f"{self.vocab_size}"
        )

        print(
            f"Total merges: "
            f"{len(self.merges)}"
        )

        return token_ids

    # ====================================
    # BPE Encode
    # ====================================

    def encode_bpe(
        self,
        text,
        add_bos=False,
        add_eos=False
    ):
        """
        Encode text using learned BPE rules.

        A priority queue stores currently available
        merge candidates. This avoids scanning the
        entire token sequence for every merge rule.
        """

        # --------------------------------
        # Character tokenization
        # --------------------------------

        token_ids = self.encode_chars(
            text
        )

        # --------------------------------
        # Empty text
        # --------------------------------

        if not token_ids:

            result = []

            if add_bos:

                result.append(
                    self.bos_token_id
                )

            if add_eos:

                result.append(
                    self.eos_token_id
                )

            return result

        # --------------------------------
        # Merge rank
        #
        # Earlier merge = lower rank
        # --------------------------------

        merge_rank = {
            pair: rank
            for rank, pair in enumerate(
                self.merges
            )
        }

        # --------------------------------
        # Pair -> merged token ID
        # --------------------------------

        pair_to_new_token = {}

        for pair in self.merges:

            token_a = self.id_to_token.get(
                pair[0]
            )

            token_b = self.id_to_token.get(
                pair[1]
            )

            if (
                token_a is None
                or token_b is None
            ):

                continue

            merged_token = (
                token_a + token_b
            )

            new_token_id = (
                self.token_to_id.get(
                    merged_token
                )
            )

            if new_token_id is None:

                continue

            pair_to_new_token[pair] = (
                new_token_id
            )

        # --------------------------------
        # Linked-list structure
        # --------------------------------

        n = len(token_ids)

        previous = [
            i - 1
            for i in range(n)
        ]

        next_node = [
            i + 1
            for i in range(n)
        ]

        next_node[-1] = -1

        active = [
            True
            for _ in range(n)
        ]

        # --------------------------------
        # Priority queue
        #
        # (merge rank, left index)
        # --------------------------------

        heap = []

        # --------------------------------
        # Initial pair candidates
        # --------------------------------

        for i in range(n - 1):

            pair = (
                token_ids[i],
                token_ids[i + 1]
            )

            rank = merge_rank.get(
                pair
            )

            if rank is not None:

                heapq.heappush(
                    heap,
                    (
                        rank,
                        i
                    )
                )

        # --------------------------------
        # Apply merges
        # --------------------------------

        while heap:

            rank, left = (
                heapq.heappop(heap)
            )

            # --------------------------------
            # Ignore invalid node
            # --------------------------------

            if (
                left < 0
                or left >= n
                or not active[left]
            ):

                continue

            # --------------------------------
            # Find right node
            # --------------------------------

            right = next_node[left]

            if (
                right == -1
                or not active[right]
            ):

                continue

            # --------------------------------
            # Current pair
            # --------------------------------

            current_pair = (
                token_ids[left],
                token_ids[right]
            )

            current_rank = (
                merge_rank.get(
                    current_pair
                )
            )

            # --------------------------------
            # Ignore stale heap entry
            # --------------------------------

            if current_rank != rank:

                continue

            # --------------------------------
            # Find merged token
            # --------------------------------

            new_token_id = (
                pair_to_new_token.get(
                    current_pair
                )
            )

            if new_token_id is None:

                continue

            # --------------------------------
            # Merge left + right
            # --------------------------------

            token_ids[left] = (
                new_token_id
            )

            # --------------------------------
            # Remove right node
            # --------------------------------

            right_next = (
                next_node[right]
            )

            next_node[left] = (
                right_next
            )

            if right_next != -1:

                previous[right_next] = (
                    left
                )

            active[right] = False

            # --------------------------------
            # Check left neighbor
            # --------------------------------

            left_previous = (
                previous[left]
            )

            if left_previous != -1:

                pair = (
                    token_ids[
                        left_previous
                    ],
                    token_ids[left]
                )

                pair_rank = (
                    merge_rank.get(
                        pair
                    )
                )

                if pair_rank is not None:

                    heapq.heappush(
                        heap,
                        (
                            pair_rank,
                            left_previous
                        )
                    )

            # --------------------------------
            # Check right neighbor
            # --------------------------------

            new_right = (
                next_node[left]
            )

            if new_right != -1:

                pair = (
                    token_ids[left],
                    token_ids[new_right]
                )

                pair_rank = (
                    merge_rank.get(
                        pair
                    )
                )

                if pair_rank is not None:

                    heapq.heappush(
                        heap,
                        (
                            pair_rank,
                            left
                        )
                    )

        # --------------------------------
        # Reconstruct final sequence
        # --------------------------------

        result = []

        index = 0

        while index != -1:

            if active[index]:

                result.append(
                    token_ids[index]
                )

            index = next_node[index]

        # --------------------------------
        # Add BOS
        # --------------------------------

        if add_bos:

            result.insert(
                0,
                self.bos_token_id
            )

        # --------------------------------
        # Add EOS
        # --------------------------------

        if add_eos:

            result.append(
                self.eos_token_id
            )

        return result

    # ====================================
    # BPE Decode
    # ====================================

    def decode_bpe(
        self,
        token_ids,
        skip_special_tokens=True
    ):

        result = []

        for token_id in token_ids:

            if token_id not in self.id_to_token:

                continue

            token = self.id_to_token[
                token_id
            ]

            # --------------------------------
            # Special tokens
            # --------------------------------

            if token in {
                self.pad_token,
                self.bos_token,
                self.eos_token
            }:

                if skip_special_tokens:

                    continue

            # --------------------------------
            # Unknown token
            # --------------------------------

            if token == self.unk_token:

                result.append("?")

            else:

                result.append(token)

        return "".join(result)

    # ====================================
    # Generic Encode
    # ====================================

    def encode(
        self,
        text,
        add_bos=False,
        add_eos=False
    ):

        return self.encode_bpe(
            text,
            add_bos=add_bos,
            add_eos=add_eos
        )

    # ====================================
    # Generic Decode
    # ====================================

    def decode(
        self,
        token_ids,
        skip_special_tokens=True
    ):

        return self.decode_bpe(
            token_ids,
            skip_special_tokens=
            skip_special_tokens
        )

    # ====================================
    # Save
    # ====================================

    def save(self, path):

        data = {

            "version": 2,

            "special_tokens": {

                "pad_token": self.pad_token,

                "unk_token": self.unk_token,

                "bos_token": self.bos_token,

                "eos_token": self.eos_token

            },

            "chars": self.chars,

            "char_to_id": self.char_to_id,

            "id_to_char": {

                str(key): value

                for key, value

                in self.id_to_char.items()

            },

            "token_to_id": self.token_to_id,

            "id_to_token": {

                str(key): value

                for key, value

                in self.id_to_token.items()

            },

            "merges": [

                list(pair)

                for pair in self.merges

            ],

            "vocab_size": self.vocab_size

        }

        with open(
            path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                data,
                file,
                indent=2,
                ensure_ascii=False
            )

        print(
            f"Tokenizer saved to: {path}"
        )

    # ====================================
    # Load
    # ====================================

    @classmethod
    def load(
        cls,
        path
    ):

        with open(
            path,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(file)

        # --------------------------------
        # Create object without __init__
        # --------------------------------

        tokenizer = cls.__new__(
            cls
        )

        # --------------------------------
        # Special tokens
        # --------------------------------

        special_tokens = data.get(
            "special_tokens",
            {}
        )

        tokenizer.pad_token = (
            special_tokens.get(
                "pad_token",
                cls.PAD_TOKEN
            )
        )

        tokenizer.unk_token = (
            special_tokens.get(
                "unk_token",
                cls.UNK_TOKEN
            )
        )

        tokenizer.bos_token = (
            special_tokens.get(
                "bos_token",
                cls.BOS_TOKEN
            )
        )

        tokenizer.eos_token = (
            special_tokens.get(
                "eos_token",
                cls.EOS_TOKEN
            )
        )

        # --------------------------------
        # Characters
        # --------------------------------

        tokenizer.chars = data.get(
            "chars",
            []
        )

        # --------------------------------
        # Character mappings
        # --------------------------------

        tokenizer.char_to_id = {

            key: int(value)

            for key, value

            in data[
                "char_to_id"
            ].items()

        }

        tokenizer.id_to_char = {

            int(key): value

            for key, value

            in data[
                "id_to_char"
            ].items()

        }

        # --------------------------------
        # BPE mappings
        # --------------------------------

        tokenizer.token_to_id = {

            key: int(value)

            for key, value

            in data[
                "token_to_id"
            ].items()

        }

        tokenizer.id_to_token = {

            int(key): value

            for key, value

            in data[
                "id_to_token"
            ].items()

        }

        # --------------------------------
        # Merge rules
        # --------------------------------

        tokenizer.merges = [

            (
                int(pair[0]),
                int(pair[1])
            )

            for pair in data.get(
                "merges",
                []
            )

        ]

        # --------------------------------
        # Vocabulary size
        # --------------------------------

        tokenizer.vocab_size = int(
            data[
                "vocab_size"
            ]
        )

        # --------------------------------
        # Ensure special tokens exist
        # --------------------------------

        required_special_tokens = [

            tokenizer.pad_token,

            tokenizer.unk_token,

            tokenizer.bos_token,

            tokenizer.eos_token

        ]

        for token in required_special_tokens:

            if token not in tokenizer.token_to_id:

                token_id = (
                    tokenizer.vocab_size
                )

                tokenizer.token_to_id[
                    token
                ] = token_id

                tokenizer.id_to_token[
                    token_id
                ] = token

                tokenizer.char_to_id[
                    token
                ] = token_id

                tokenizer.id_to_char[
                    token_id
                ] = token

                tokenizer.vocab_size += 1

        return tokenizer

    # ====================================
    # Length
    # ====================================

    def __len__(self):

        return self.vocab_size