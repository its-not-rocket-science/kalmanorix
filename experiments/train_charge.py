"""Train a SentenceTransformer model specialised for charging-related queries."""

#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import random
from typing import cast

from torch.utils.data import DataLoader, Dataset
from sentence_transformers import SentenceTransformer, InputExample, losses

from kalmanorix.toy_corpus import build_toy_corpus


def main() -> None:
    """
    Train a SentenceTransformer model specialised for charging-related queries.
    """
    # pylint: disable=too-many-locals
    random.seed(0)

    repo_root = Path(__file__).resolve().parent.parent
    base_path = repo_root / "models" / "tech-minilm"
    out_path = repo_root / "models" / "charge-minilm"

    corpus = build_toy_corpus(british_spelling=True)

    # --- Targets in toy corpus ---
    charge_doc_id = 14
    thermal_doc_id = 11

    # Hard negatives we *really* want to separate from charge
    # (tune this list aggressively)
    hard_neg_docs = [
        11,  # thermal load (your main confuser)
        8,  # battery optimisation settings
        16,  # battery optimisation w/ sauce analogy
        0,  # smartphone battery life & fast charging (too generic)
        3,  # camera low-light
        13,  # GPU pipeline
        2,  # GPU drivers/frame rates
        1,  # CPU sustained load
    ]

    charge_doc = corpus.docs[charge_doc_id]
    thermal_doc = corpus.docs[thermal_doc_id]

    # Charge intent anchors (include your dataset phrasing)
    charge_queries = [
        "fast charging usb-c pd charger rated cable wattage",
        "usb-c power delivery negotiation pdo rdo",
        "pps fast charging voltage current steps",
        "charger wattage cable e-marker limits",
        "why charging speed depends on cable and wattage",
        "fast charging with the right charger",
        "usb-c pd profiles compatibility handshake",
    ]

    # Thermal-ish anchors (to explicitly pull thermal away from charge)
    thermal_queries = [
        "thermal load spikes cause overheating reduce power draw",
        "cpu gpu throttling when hot under sustained load",
        "heatsinks airflow thermal paste prevent overheating",
    ]

    model = SentenceTransformer(str(base_path))

    # -----------------------------
    # Objective A: MN ranking (pull q -> doc14)
    # -----------------------------
    mn_examples: list[InputExample] = []
    for q in charge_queries:
        mn_examples.append(InputExample(texts=[q, charge_doc]))

    mn_loader: DataLoader[InputExample] = DataLoader(
        cast(Dataset[InputExample], mn_examples), shuffle=True, batch_size=8
    )
    mn_loss = losses.MultipleNegativesRankingLoss(model)

    # -----------------------------
    # Objective B: Triplet loss (repel hard negatives)
    # -----------------------------
    triplet_examples: list[InputExample] = []

    # (charge_query, doc14, hard_negative_doc)
    for q in charge_queries:
        for neg_id in hard_neg_docs:
            if neg_id == charge_doc_id:
                continue
            triplet_examples.append(
                InputExample(texts=[q, charge_doc, corpus.docs[neg_id]])
            )

    # Also: (thermal_query, doc11, doc14) to push doc14 away from thermal intent
    for q in thermal_queries:
        triplet_examples.append(InputExample(texts=[q, thermal_doc, charge_doc]))

    # And: doc-doc repulsion: (doc14, doc14 paraphrase, doc11)
    # crude but helps reduce doc14~doc11 similarity
    for _ in range(20):
        q = random.choice(charge_queries)
        triplet_examples.append(InputExample(texts=[charge_doc, q, thermal_doc]))

    triplet_loader: DataLoader[InputExample] = DataLoader(
        cast(Dataset[InputExample], triplet_examples), shuffle=True, batch_size=16
    )
    triplet_loss = losses.TripletLoss(
        model=model,
        distance_metric=losses.TripletDistanceMetric.COSINE,
        triplet_margin=0.25,
    )

    # -----------------------------
    # Joint training
    # -----------------------------
    model.fit(
        train_objectives=[
            (mn_loader, mn_loss),
            (triplet_loader, triplet_loss),
        ],
        epochs=1,  # bump to 2–3 if it under-separates
        warmup_steps=20,
        show_progress_bar=True,
    )

    out_path.mkdir(parents=True, exist_ok=True)
    model.save(str(out_path))
    print(f"Saved charge model to: {out_path}")


if __name__ == "__main__":
    main()
