"""C2.0 CLIP zero-shot labeler (pinned backend; optional dependency).

Protocol: docs/c2_matched_labels_protocol.md. OpenCLIP ViT-B-32 /
`openai` weights, CPU, eval mode, no_grad — deterministic. torch and
open_clip are OPTIONAL dependencies: the frozen pipeline never imports
this module; run C2 tools with the project venv
(`.venv/bin/python tools/c2_run.py ...`).

Isolation: this module receives images and a class vocabulary — nothing
else. It never touches oracle labels, keys, or semantic meshes.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

MODEL_NAME = "ViT-B-32-quickgelu"
PRETRAINED = "openai"
PROMPTS = ("a photo of a {c}",
           "a photo of a {c} in a room",
           "a 3D render of a {c}")


class ClipLabeler:
    def __init__(self):
        import open_clip
        import torch
        self._torch = torch
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            MODEL_NAME, pretrained=PRETRAINED, device="cpu")
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(MODEL_NAME)
        # record the exact weights actually used (open_clip resolves the
        # 'openai' tag to the timm HF snapshot below)
        digest = None
        hub = Path.home() / ".cache" / "huggingface" / "hub"
        for p in sorted(hub.glob(
                "models--timm--vit_base_patch32_clip_224.openai/snapshots/"
                "*/open_clip_model.safetensors")):
            digest = hashlib.sha256(p.read_bytes()).hexdigest()
            break
        self.weights_sha256 = digest
        self._text_cache: dict[tuple[str, ...], "object"] = {}

    def _text_embeddings(self, vocabulary: tuple[str, ...]):
        torch = self._torch
        if vocabulary not in self._text_cache:
            prompts = [t.format(c=c) for c in vocabulary for t in PROMPTS]
            with torch.no_grad():
                emb = self.model.encode_text(self.tokenizer(prompts))
            emb = emb / emb.norm(dim=-1, keepdim=True)
            emb = emb.reshape(len(vocabulary), len(PROMPTS), -1).mean(dim=1)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            self._text_cache[vocabulary] = emb
        return self._text_cache[vocabulary]

    def text_embeddings(self, vocabulary: tuple[str, ...]):
        """Public accessor for the pinned prompt-ensembled text embeddings."""
        return self._text_embeddings(tuple(vocabulary))

    def image_embeddings(self, images):
        """Per-image L2-normalised embeddings, WITHOUT mean-pooling views.

        `classify` mean-pools an instance's views before scoring, which is
        right for assigning one label to one instance but destroys the
        per-view detail a cross-view agreement rule needs. Additive: this
        touches nothing `classify` does.
        """
        torch = self._torch
        batch = torch.stack([self.preprocess(im) for im in images])
        with torch.no_grad():
            emb = self.model.encode_image(batch)
        return emb / emb.norm(dim=-1, keepdim=True)

    def classify(self, images, vocabulary: list[str]) -> list[dict]:
        """images: list of PIL images (the instance's views).
        Returns the ranked vocabulary: [{label, score}, ...] best first."""
        torch = self._torch
        vocab = tuple(vocabulary)
        batch = torch.stack([self.preprocess(im) for im in images])
        with torch.no_grad():
            emb = self.model.encode_image(batch)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        mean = emb.mean(dim=0, keepdim=True)
        mean = mean / mean.norm(dim=-1, keepdim=True)
        sims = (mean @ self._text_embeddings(vocab).T).squeeze(0)
        order = sims.argsort(descending=True)
        return [{"label": vocab[i], "score": round(float(sims[i]), 4)}
                for i in order.tolist()]
