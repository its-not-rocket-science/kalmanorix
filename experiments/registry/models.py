"""Model loading and village construction for benchmark registry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from kalmanorix import SEF, Village
from kalmanorix.uncertainty import CentroidDistanceSigma2, KeywordSigma2


class DebugKeywordEmbedder:
    """Deterministic toy embedder for smoke tests."""

    def __init__(self, dim: int, keywords: list[str], seed: int) -> None:
        rng = np.random.default_rng(seed)
        self.keywords = keywords
        self.base = rng.normal(size=(dim,))
        self.base /= np.linalg.norm(self.base) + 1e-12
        self.direction = rng.normal(size=(dim,))
        self.direction /= np.linalg.norm(self.direction) + 1e-12

    def __call__(self, text: str) -> np.ndarray:
        vec = self.base.copy()
        if any(kw in text.lower() for kw in self.keywords):
            vec += 2.0 * self.direction
        vec /= np.linalg.norm(vec) + 1e-12
        return vec.astype(np.float64)


class HFMeanPoolEmbedder:
    """HuggingFace encoder with masked mean pooling + L2 normalization."""

    def __init__(self, model_name: str, device: str) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._torch = torch
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def __call__(self, text: str) -> np.ndarray:
        torch = self._torch
        with torch.inference_mode():
            encoded = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            outputs = self.model(**encoded)
            hidden = outputs.last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1)
            pooled = (hidden * mask).sum(dim=1) / torch.clamp(mask.sum(dim=1), min=1)
            vec = pooled[0].detach().cpu().numpy().astype(np.float64)
            vec /= np.linalg.norm(vec) + 1e-12
            return vec


@dataclass(frozen=True)
class SpecialistSpec:
    name: str
    domain: str
    model_name: str


def build_village(kind: str, payload: Any, specialists: list[dict[str, Any]], device: str) -> Village:
    """Build village from config/model kind."""
    if kind == "debug_keyword":
        modules = [
            SEF(
                name="tech",
                embed=DebugKeywordEmbedder(96, ["battery", "cpu", "gpu", "camera"], seed=7),
                sigma2=KeywordSigma2({"battery", "cpu", "gpu", "camera"}, 0.1, 0.5),
            ),
            SEF(
                name="cook",
                embed=DebugKeywordEmbedder(96, ["braise", "simmer", "sauce", "oven"], seed=11),
                sigma2=KeywordSigma2({"braise", "simmer", "sauce", "oven"}, 0.1, 0.5),
            ),
        ]
        return Village(modules)

    if kind == "hf_specialists":
        if not specialists:
            raise ValueError("models.specialists must be set for hf_specialists")
        rows = payload
        modules = []
        for raw in specialists:
            spec = SpecialistSpec(**raw)
            embedder = HFMeanPoolEmbedder(spec.model_name, device=device)
            texts = [r["query_text"] for r in rows if r.get("domain_label") == spec.domain]
            if not texts:
                texts = [f"{spec.domain} calibration", f"{spec.domain} retrieval terminology"]
            calibration = (texts * ((128 // len(texts)) + 1))[:128]
            sigma2 = CentroidDistanceSigma2.from_calibration(
                embed=embedder,
                calibration_texts=calibration,
                base_sigma2=0.15,
                scale=2.5,
            )
            modules.append(
                SEF(
                    name=spec.name,
                    embed=embedder,
                    sigma2=sigma2,
                    meta={"domain": spec.domain, "model_name": spec.model_name},
                )
            )
        return Village(modules)

    raise ValueError(f"Unsupported model kind: {kind}")
