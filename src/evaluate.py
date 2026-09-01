import torch


def evaluate(model, dataset):

    model.eval()

    total_loss = 0.0

    with torch.no_grad():

        for i in range(len(dataset)):

            x, y = dataset[i]

            x = x.unsqueeze(0)
            y = y.unsqueeze(0)

            logits = model(x)

            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y.view(-1)
            )

            total_loss += loss.item()

    model.train()

    return total_loss / len(dataset)