import torch


class Generator:

    def __init__(
        self,
        model,
        tokenizer,
        device="cpu"
    ):

        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        self.model.to(device)
        self.model.eval()

    # ============================================================
    # REPETITION PENALTY
    # ============================================================

    def apply_repetition_penalty(
        self,
        logits,
        tokens,
        repetition_penalty
    ):

        if repetition_penalty == 1.0:
            return logits

        # Get all tokens already generated
        previous_tokens = tokens[0].unique()

        for token_id in previous_tokens:

            token_id = token_id.item()

            # Positive logits are divided
            # Negative logits are multiplied

            if logits[0, token_id] > 0:

                logits[0, token_id] /= (
                    repetition_penalty
                )

            else:

                logits[0, token_id] *= (
                    repetition_penalty
                )

        return logits

    # ============================================================
    # TOP-K
    # ============================================================

    def apply_top_k(
        self,
        logits,
        top_k
    ):

        if top_k is None:
            return logits

        if top_k <= 0:
            return logits

        top_k = min(
            top_k,
            logits.size(-1)
        )

        values, _ = torch.topk(
            logits,
            top_k
        )

        minimum_value = values[:, -1].unsqueeze(-1)

        logits = torch.where(
            logits < minimum_value,
            torch.full_like(
                logits,
                float("-inf")
            ),
            logits
        )

        return logits

    # ============================================================
    # TOP-P / NUCLEUS SAMPLING
    # ============================================================

    def apply_top_p(
        self,
        logits,
        top_p
    ):

        if top_p is None:
            return logits

        if top_p >= 1.0:
            return logits

        if top_p <= 0.0:
            return logits

        # --------------------------------
        # Convert logits → probabilities
        # --------------------------------

        probabilities = torch.softmax(
            logits,
            dim=-1
        )

        # Sort probabilities from largest
        # to smallest

        sorted_probabilities, sorted_indices = torch.sort(
            probabilities,
            descending=True
        )

        # Cumulative probability

        cumulative_probabilities = torch.cumsum(
            sorted_probabilities,
            dim=-1
        )

        # Tokens after cumulative probability
        # exceeds top_p are removed

        remove_mask = (
            cumulative_probabilities > top_p
        )

        # Always keep at least one token

        remove_mask[:, 0] = False

        # Convert sorted mask back to
        # original vocabulary positions

        original_remove_mask = torch.zeros_like(
            remove_mask
        )

        original_remove_mask.scatter_(
            1,
            sorted_indices,
            remove_mask
        )

        logits = logits.masked_fill(
            original_remove_mask,
            float("-inf")
        )

        return logits

    # ============================================================
    # GENERATE
    # ============================================================

    @torch.no_grad()
    def generate(
        self,
        prompt,
        max_new_tokens=50,
        temperature=0.8,
        top_k=10,
        top_p=0.9,
        repetition_penalty=1.15
    ):

        # --------------------------------------------------------
        # Validate temperature
        # --------------------------------------------------------

        if temperature <= 0:

            raise ValueError(
                "temperature must be greater than 0"
            )

        # --------------------------------------------------------
        # Encode prompt
        # --------------------------------------------------------

        token_ids = self.tokenizer.encode(
            prompt
        )

        # Make sure prompt is not empty

        if len(token_ids) == 0:

            raise ValueError(
                "Prompt cannot be empty."
            )

        # --------------------------------------------------------
        # Convert to tensor
        # --------------------------------------------------------

        tokens = torch.tensor(
            token_ids,
            dtype=torch.long,
            device=self.device
        ).unsqueeze(0)

        # --------------------------------------------------------
        # Generation loop
        # --------------------------------------------------------

        for _ in range(max_new_tokens):

            # Keep only model context

            context = tokens[
                :,
                -self.model.block_size:
            ]

            # ----------------------------------------------------
            # Model prediction
            # ----------------------------------------------------

            logits = self.model(
                context
            )

            # Only last position

            logits = logits[
                :,
                -1,
                :
            ]

            # ----------------------------------------------------
            # Temperature
            # ----------------------------------------------------

            logits = logits / temperature

            # ----------------------------------------------------
            # Repetition penalty
            # ----------------------------------------------------

            logits = self.apply_repetition_penalty(
                logits,
                tokens,
                repetition_penalty
            )

            # ----------------------------------------------------
            # Top-K
            # ----------------------------------------------------

            logits = self.apply_top_k(
                logits,
                top_k
            )

            # ----------------------------------------------------
            # Top-P
            # ----------------------------------------------------

            logits = self.apply_top_p(
                logits,
                top_p
            )

            # ----------------------------------------------------
            # Convert logits to probabilities
            # ----------------------------------------------------

            probabilities = torch.softmax(
                logits,
                dim=-1
            )

            # ----------------------------------------------------
            # Safety check
            # ----------------------------------------------------

            if torch.isnan(
                probabilities
            ).any():

                raise RuntimeError(
                    "NaN detected in generation probabilities."
                )

            # ----------------------------------------------------
            # Sample next token
            # ----------------------------------------------------

            next_token = torch.multinomial(
                probabilities,
                num_samples=1
            )

            # ----------------------------------------------------
            # Append token
            # ----------------------------------------------------

            tokens = torch.cat(
                [
                    tokens,
                    next_token
                ],
                dim=1
            )

        # --------------------------------------------------------
        # Decode
        # --------------------------------------------------------

        generated_ids = tokens[
            0
        ].tolist()

        generated_text = self.tokenizer.decode(
            generated_ids
        )

        return generated_text
