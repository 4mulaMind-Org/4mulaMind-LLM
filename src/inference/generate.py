import torch


def generate(
    model,
    tokenizer,
    prompt,
    max_new_tokens=50,
    temperature=1.0,
    top_k=5
):
    """
    Generate text using temperature and Top-K sampling.
    """

    model.eval()

    # --------------------------------
    # Encode prompt
    # --------------------------------

    token_ids = tokenizer.encode(prompt)

    x = torch.tensor(
        [token_ids],
        dtype=torch.long
    )

    # --------------------------------
    # Generate tokens
    # --------------------------------

    with torch.no_grad():

        for _ in range(max_new_tokens):

            # Keep context within model limit
            if x.size(1) > model.block_size:

                x_context = x[:, -model.block_size:]

            else:

                x_context = x

            # --------------------------------
            # Forward pass
            # --------------------------------

            logits = model(x_context)

            # Last position
            next_token_logits = logits[:, -1, :]

            # --------------------------------
            # Temperature
            # --------------------------------

            next_token_logits = (
                next_token_logits / temperature
            )

            # --------------------------------
            # Top-K filtering
            # --------------------------------

            if top_k is not None:

                values, indices = torch.topk(
                    next_token_logits,
                    min(top_k, next_token_logits.size(-1))
                )

                filtered_logits = torch.full_like(
                    next_token_logits,
                    float("-inf")
                )

                filtered_logits.scatter_(
                    -1,
                    indices,
                    values
                )

                next_token_logits = filtered_logits

            # --------------------------------
            # Softmax
            # --------------------------------

            probabilities = torch.softmax(
                next_token_logits,
                dim=-1
            )

            # --------------------------------
            # Sample
            # --------------------------------

            next_token_id = torch.multinomial(
                probabilities,
                num_samples=1
            )

            # --------------------------------
            # Append token
            # --------------------------------

            x = torch.cat(
                [x, next_token_id],
                dim=1
            )

    # --------------------------------
    # Decode
    # --------------------------------

    generated_ids = x[0].tolist()

    return tokenizer.decode(generated_ids)