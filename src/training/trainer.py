import torch


class Trainer:

    def __init__(
        self,
        model,
        dataset,
        validation_dataset,
        loss_function,
        learning_rate=3e-4,
        batch_size=16,
        warmup_steps=100,
        max_steps=5000,
        patience=5,
        checkpoint_path="checkpoints/model.pt"
    ):

        self.model = model

        self.dataset = dataset

        self.validation_dataset = (
            validation_dataset
        )

        self.loss_function = loss_function

        self.batch_size = batch_size

        self.learning_rate = learning_rate

        self.warmup_steps = warmup_steps

        self.max_steps = max_steps

        # ====================================
        # Early Stopping
        # ====================================

        self.patience = patience

        # ====================================
        # Optimizer
        # ====================================

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.95),
            weight_decay=0.01
        )

    # ====================================
    # Learning Rate
    # ====================================

    def get_learning_rate(self, step):

        # ====================================
        # Warmup
        # ====================================

        if step < self.warmup_steps:

            return self.learning_rate * (
                (step + 1)
                / self.warmup_steps
            )

        # ====================================
        # Progress after warmup
        # ====================================

        progress = (
            step - self.warmup_steps
        ) / max(
            1,
            self.max_steps - self.warmup_steps
        )

        progress = min(
            max(progress, 0.0),
            1.0
        )

        # ====================================
        # Cosine Decay
        # ====================================

        min_lr = (
            self.learning_rate * 0.1
        )

        cosine_decay = (
            0.5
            * (
                1.0
                + torch.cos(
                    torch.tensor(
                        progress
                        * torch.pi
                    )
                )
            ).item()
        )

        return (
            min_lr
            + (
                self.learning_rate
                - min_lr
            )
            * cosine_decay
        )

    # ====================================
    # Create Batch
    # ====================================

    def get_batch(
        self,
        dataset
    ):

        dataset_size = len(
            dataset
        )

        indices = torch.randint(
            0,
            dataset_size,
            (self.batch_size,)
        ).tolist()

        inputs = []
        targets = []

        for index in indices:

            x, y = dataset[index]

            inputs.append(x)
            targets.append(y)

        x = torch.stack(
            inputs
        )

        targets = torch.stack(
            targets
        )

        return x, targets

    # ====================================
    # Validation
    # ====================================

    def evaluate(self):

        self.model.eval()

        total_loss = 0.0

        with torch.no_grad():

            dataset_size = len(
                self.validation_dataset
            )

            if dataset_size == 0:

                self.model.train()

                return float("inf")

            num_batches = min(
                50,
                (
                    dataset_size
                    + self.batch_size
                    - 1
                )
                // self.batch_size
            )

            for batch_index in range(
                num_batches
            ):

                start = (
                    batch_index
                    * self.batch_size
                )

                end = min(
                    start + self.batch_size,
                    dataset_size
                )

                inputs = []
                targets = []

                for index in range(
                    start,
                    end
                ):

                    x, y = (
                        self.validation_dataset[
                            index
                        ]
                    )

                    inputs.append(x)
                    targets.append(y)

                x = torch.stack(
                    inputs
                )

                targets = torch.stack(
                    targets
                )

                logits = self.model(x)

                loss = self.loss_function(
                    logits,
                    targets
                )

                total_loss += loss.item()

        self.model.train()

        return (
            total_loss
            / num_batches
        )

    # ====================================
    # Training
    # ====================================

    def train(
        self,
        steps=5000
    ):

        self.model.train()

        best_val_loss = float("inf")

        patience_counter = 0

        # ====================================
        # Training Loop
        # ====================================

        for step in range(steps):

            # ====================================
            # Learning Rate
            # ====================================

            learning_rate = (
                self.get_learning_rate(
                    step
                )
            )

            for param_group in (
                self.optimizer.param_groups
            ):

                param_group["lr"] = (
                    learning_rate
                )

            # ====================================
            # Batch
            # ====================================

            x, targets = self.get_batch(
                self.dataset
            )

            # ====================================
            # Forward
            # ====================================

            logits = self.model(x)

            # ====================================
            # Loss
            # ====================================

            loss = self.loss_function(
                logits,
                targets
            )

            # ====================================
            # Backpropagation
            # ====================================

            self.optimizer.zero_grad(
                set_to_none=True
            )

            loss.backward()

            # ====================================
            # Gradient Clipping
            # ====================================

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=1.0
            )

            # ====================================
            # Optimizer Step
            # ====================================

            self.optimizer.step()

            # ====================================
            # Validation
            # ====================================

            if (
                step % 100 == 0
                or step == steps - 1
            ):

                val_loss = (
                    self.evaluate()
                )

                # ====================================
                # Improvement
                # ====================================

                if val_loss < best_val_loss:

                    best_val_loss = (
                        val_loss
                    )

                    patience_counter = 0

                    # ====================================
                    # Save Best Checkpoint
                    # ====================================

                    checkpoint = {

                        "model_state_dict":
                            self.model.state_dict(),

                        "best_val_loss":
                            best_val_loss,

                        "step":
                            step,

                        # Model configuration
                        "vocab_size":
                            self.model.lm_head.out_features,

                        "block_size":
                            self.model.block_size,

                        "embedding_dim": self.model.embedding_dim
                    }

                    # ------------------------------------
                    # Optional architecture attributes
                    # ------------------------------------

                    if hasattr(
                        self.model,
                        "num_heads"
                    ):

                        checkpoint[
                            "num_heads"
                        ] = self.model.num_heads

                    if hasattr(
                        self.model,
                        "hidden_dim"
                    ):

                        checkpoint[
                            "hidden_dim"
                        ] = self.model.hidden_dim

                    if hasattr(
                        self.model,
                        "num_layers"
                    ):

                        checkpoint[
                            "num_layers"
                        ] = self.model.num_layers

                    # ====================================
                    # Save
                    # ====================================

                    torch.save(
                        checkpoint,
                        "checkpoints/4mulamind_bpe.pt"
                    )

                    print(
                        f"✓ Best model saved | "
                        f"Val Loss: "
                        f"{val_loss:.4f}"
                    )

                # ====================================
                # No Improvement
                # ====================================

                else:

                    patience_counter += 1

                # ====================================
                # Logging
                # ====================================

                print(
                    f"Step {step:5d} | "
                    f"Train Loss: "
                    f"{loss.item():.4f} | "
                    f"Val Loss: "
                    f"{val_loss:.4f} | "
                    f"LR: "
                    f"{learning_rate:.6f}"
                )

                # ====================================
                # Early Stopping
                # ====================================

                if (
                    patience_counter
                    >= self.patience
                ):

                    print(
                        "\n================================"
                    )

                    print(
                        "Early Stopping"
                    )

                    print(
                        "================================"
                    )

                    print(
                        f"No validation improvement "
                        f"for {self.patience} evaluations."
                    )

                    print(
                        f"Best validation loss: "
                        f"{best_val_loss:.4f}"
                    )

                    print(
                        f"Stopped at step: "
                        f"{step}"
                    )

                    break

        return best_val_loss

