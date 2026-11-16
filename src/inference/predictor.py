import re
import torch
from src.tokenizer import Tokenizer
from src.model import FourmulaMindLM
from src.training.checkpoint import load_checkpoint


class IntentPredictor:
    def __init__(
        self,
        checkpoint_path,
        tokenizer_path,
        device="cpu",
        max_new_tokens=16,
    ):
        self.device = torch.device(device)
        self.max_new_tokens = max_new_tokens

        self.tokenizer = Tokenizer.load(tokenizer_path)

        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        self.model = FourmulaMindLM(
            vocab_size=ckpt["vocab_size"],
            block_size=ckpt["block_size"],
            embedding_dim=ckpt["embedding_dim"],
            num_heads=ckpt["num_heads"],
            hidden_dim=ckpt["hidden_dim"],
            num_layers=ckpt["num_layers"],
        )
        self.model = load_checkpoint(self.model, checkpoint_path)
        self.model = self.model.to(self.device)
        self.model.eval()

        self.block_size = ckpt["block_size"]

    @torch.no_grad()
    def predict(self, instruction):
        prompt = "User: " + instruction + "\nAssistant: "
        ids = self.tokenizer.encode(prompt)
        generated = list(ids)

        for _ in range(self.max_new_tokens):
            ctx = generated[-self.block_size:]
            x = torch.tensor([ctx], dtype=torch.long, device=self.device)
            logits = self.model(x)
            nxt = int(torch.argmax(logits[0, -1]).item())
            generated.append(nxt)
            if nxt == self.tokenizer.eos_token_id:
                break

        out = self.tokenizer.decode(generated[len(ids):])
        return self._normalize(out)

    @staticmethod
    def _normalize(text):
        text = text.strip().split("\n", 1)[0].strip()
        text = re.sub(r"^(assistant|answer)\s*:\s*", "", text, flags=re.IGNORECASE)
        return text.strip().upper()
