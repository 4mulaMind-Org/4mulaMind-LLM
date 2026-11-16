import json
import os
from array import array

from src.tokenizer import Tokenizer


# ============================================================
# Paths
# ============================================================

INPUT_JSONL = (
    "data/run4_train.jsonl"
)

TOKENIZER_PATH = (
    "tokenizers/4mulamind_bpe.json"
)

OUTPUT_DIR = (
    "data/packets/run4_train"
)


# ============================================================
# Configuration
# ============================================================

PACKET_SIZE = 16_384

MAX_UINT16 = 65_535


# ============================================================
# Read JSONL
# ============================================================

def load_texts(path):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        for line_number, line in enumerate(
            file,
            start=1
        ):

            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)

            except json.JSONDecodeError as error:

                print(
                    f"Skipping invalid JSON "
                    f"line {line_number}: {error}"
                )

                continue

            text = record.get(
                "text",
                ""
            )

            if not isinstance(
                text,
                str
            ):
                continue

            text = text.strip()

            if text:
                yield text


# ============================================================
# Write Packet
# ============================================================

def write_packet(
    tokens,
    packet_id
):

    filename = (
        f"packet_{packet_id:06d}.bin"
    )

    path = os.path.join(
        OUTPUT_DIR,
        filename
    )

    data = array(
        "H",
        tokens
    )

    with open(
        path,
        "wb"
    ) as file:

        data.tofile(file)

    return filename, len(tokens)


# ============================================================
# Build Packets
# ============================================================

def build_packets(tokenizer):

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    packet_tokens = []

    packet_id = 0

    total_documents = 0
    total_tokens = 0
    total_packets = 0

    packet_metadata = []

    eos_id = tokenizer.eos_token_id

    print()
    print("================================")
    print("Building Run 4 Packets")
    print("================================")

    print()
    print("Input:")
    print(INPUT_JSONL)

    print()
    print("Output:")
    print(OUTPUT_DIR)

    for text in load_texts(
        INPUT_JSONL
    ):

        total_documents += 1

        tokens = tokenizer.encode(
            text
        )

        # --------------------------------
        # Validate token IDs
        # --------------------------------

        for token_id in tokens:

            if not (
                0 <= token_id <= MAX_UINT16
            ):

                raise ValueError(
                    f"Token ID {token_id} "
                    f"cannot be stored in uint16."
                )

        # --------------------------------
        # Document boundary
        # --------------------------------

        tokens.append(
            eos_id
        )

        position = 0

        while position < len(tokens):

            remaining = (
                PACKET_SIZE
                - len(packet_tokens)
            )

            piece = tokens[
                position:
                position + remaining
            ]

            packet_tokens.extend(
                piece
            )

            position += len(piece)

            total_tokens += len(piece)

            # --------------------------------
            # Packet full
            # --------------------------------

            if len(packet_tokens) == PACKET_SIZE:

                (
                    filename,
                    count
                ) = write_packet(
                    packet_tokens,
                    packet_id
                )

                packet_metadata.append(
                    {
                        "packet_id": packet_id,
                        "file": filename,
                        "token_count": count
                    }
                )

                print(
                    f"Packet "
                    f"{packet_id:06d}: "
                    f"{count} tokens"
                )

                packet_id += 1
                total_packets += 1

                packet_tokens = []

    # --------------------------------
    # Final partial packet
    # --------------------------------

    if packet_tokens:

        (
            filename,
            count
        ) = write_packet(
            packet_tokens,
            packet_id
        )

        packet_metadata.append(
            {
                "packet_id": packet_id,
                "file": filename,
                "token_count": count
            }
        )

        print(
            f"Packet "
            f"{packet_id:06d}: "
            f"{count} tokens"
        )

        total_packets += 1

    # --------------------------------
    # Manifest
    # --------------------------------

    manifest = {
        "version": 1,
        "tokenizer": TOKENIZER_PATH,
        "packet_size": PACKET_SIZE,
        "vocab_size": tokenizer.vocab_size,
        "eos_token_id": eos_id,
        "documents": total_documents,
        "tokens": total_tokens,
        "packets": packet_metadata
    }

    manifest_path = os.path.join(
        OUTPUT_DIR,
        "manifest.json"
    )

    with open(
        manifest_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            manifest,
            file,
            indent=2
        )

    print()
    print("================================")
    print("Run 4 Packetization Complete")
    print("================================")

    print()
    print("Documents:", total_documents)

    print("Tokens:", total_tokens)

    print("Packets:", total_packets)

    print("Packet size:", PACKET_SIZE)

    print("Manifest:", manifest_path)


# ============================================================
# Main
# ============================================================

def main():

    print()
    print("================================")
    print("4mulaMind Run 4")
    print("Packet Builder")
    print("================================")

    # --------------------------------
    # Load existing tokenizer
    # --------------------------------

    tokenizer = Tokenizer.load(
        TOKENIZER_PATH
    )

    print()
    print(
        "Vocabulary:",
        tokenizer.vocab_size
    )

    print(
        "BPE merges:",
        len(tokenizer.merges)
    )

    if tokenizer.vocab_size > MAX_UINT16:

        raise ValueError(
            "Vocabulary exceeds uint16 limit."
        )

    build_packets(
        tokenizer
    )


if __name__ == "__main__":

    main()
