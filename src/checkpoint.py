import torch


# ============================================================
# Save Checkpoint
# ============================================================

def save_checkpoint(
    model,
    path,
    step=None,
    best_val_loss=None,
    config=None
):
    """
    Save model weights together with useful
    training metadata.
    """

    checkpoint = {
        "model_state_dict": model.state_dict(),
    }

    if step is not None:
        checkpoint["step"] = step

    if best_val_loss is not None:
        checkpoint["best_val_loss"] = best_val_loss

    if config is not None:
        checkpoint.update(config)

    torch.save(
        checkpoint,
        path
    )

    print(
        f"Model saved to: {path}"
    )


# ============================================================
# Load Checkpoint
# ============================================================

def load_checkpoint(
    model,
    path
):
    """
    Load model weights from either:

    1. A wrapped checkpoint containing
       'model_state_dict'

    2. A raw PyTorch state_dict
    """

    checkpoint = torch.load(
        path,
        map_location="cpu"
    )

    # --------------------------------
    # Wrapped checkpoint
    # --------------------------------

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):

        state_dict = (
            checkpoint["model_state_dict"]
        )

        print(
            f"Model loaded from: {path}"
        )

        # --------------------------------
        # Show checkpoint metadata
        # --------------------------------

        if "step" in checkpoint:

            print(
                "Checkpoint step:",
                checkpoint["step"]
            )

        if "best_val_loss" in checkpoint:

            print(
                "Best validation loss:",
                checkpoint["best_val_loss"]
            )

        for key in [
            "vocab_size",
            "block_size",
            "embedding_dim",
            "num_heads",
            "hidden_dim",
            "num_layers"
        ]:

            if key in checkpoint:

                print(
                    f"{key}:",
                    checkpoint[key]
                )

    # --------------------------------
    # Raw state_dict
    # --------------------------------

    else:

        state_dict = checkpoint

        print(
            f"Raw model weights loaded from: {path}"
        )

    # --------------------------------
    # Load weights
    # --------------------------------

    model.load_state_dict(
        state_dict
    )

    return model


# ============================================================
# Load Checkpoint Metadata
# ============================================================

def get_checkpoint_metadata(path):
    """
    Return checkpoint metadata without
    loading weights into a model.
    """

    checkpoint = torch.load(
        path,
        map_location="cpu"
    )

    if not isinstance(
        checkpoint,
        dict
    ):

        return {}

    metadata = {}

    for key in [
        "step",
        "best_val_loss",
        "vocab_size",
        "block_size",
        "embedding_dim",
        "num_heads",
        "hidden_dim",
        "num_layers"
    ]:

        if key in checkpoint:

            metadata[key] = checkpoint[key]

    return metadata