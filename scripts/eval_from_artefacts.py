#!/usr/bin/env python3
"""
Evaluate Kalmanorix strategies from SEF JSON artefacts + an embedder registry.

This script demonstrates "Option B" (registry-driven runtime construction):

- SEF artefacts are small JSON files committed to the repo.
- Embedders (e.g., SentenceTransformer checkpoints) are resolved at runtime via
  an EmbedderRegistry mapping embedder_id -> embed(text)->vector callable.
- Model checkpoints live in models/ (usually gitignored unless using LFS).

Expected repo layout (recommended)
----------------------------------
kalmanorix/
  models/                     # local-only checkpoints (gitignored)
    tech-minilm/
    cook-minilm/
  artefacts/
    sefs/
      tech.json
      cook.json
    calibration/              # optional, for centroid_distance
      tech.txt
      cook.txt
    registries/               # optional
      local.json              # optional mapping of embedder_id -> checkpoint path

Usage
-----
  python scripts/eval_from_artefacts.py

Optional flags
--------------
  --sefs-dir artefacts/sefs
  --models-dir models
  --repo-root .
  --k 3
  --override-base-sigma2 0.2
  --override-scale 5.0
  --debug-sims
"""

# pylint: disable=duplicate-code
from __future__ import annotations

from pathlib import Path
from typing import cast, Dict, Iterable, List, Optional, Tuple

import argparse
import json

import numpy as np

from kalmanorix import (
    KalmanorixFuser,
    LearnedGateFuser,
    MeanFuser,
    Panoramix,
    ScoutRouter,
    SEF,
    Village,
    eval_retrieval,
)
from kalmanorix.types import Embedder
from kalmanorix.registry import EmbedderRegistry
from kalmanorix.sef_io import SEFArtefact
from kalmanorix.toy_corpus import build_toy_corpus, print_doc_index


# -----------------------------
# Embedding + SEF loading
# -----------------------------


def make_st_embedder(checkpoint_path: str):
    """
    Create a SentenceTransformer embedder (returns np.float64 unit vectors).

    Imported lazily so core library can stay light unless you run this script.
    """
    # pylint: disable=import-outside-toplevel
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(checkpoint_path)

    def embed(text: str) -> np.ndarray:
        v = model.encode([text], normalize_embeddings=True, convert_to_numpy=True)[0]
        return v.astype(np.float64)

    return embed


def load_embedder_registry(
    *,
    models_dir: Path,
    registry_json: Optional[Path],
) -> EmbedderRegistry:
    """
    Build an EmbedderRegistry mapping embedder_id -> callable.

    If registry_json is provided, it should map:
      { "embedder_id": "relative/or/absolute/path/to/checkpoint", ... }

    Otherwise, we fall back to a small convention:
      embedder_id == "tech-minilm" -> models_dir/"tech-minilm"
      embedder_id == "cook-minilm" -> models_dir/"cook-minilm"
    """
    mapping: Dict[str, str] = {}

    if registry_json is not None and registry_json.exists():
        raw = json.loads(registry_json.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("registry JSON must be an object mapping id -> path")
        mapping = {str(k): str(v) for k, v in raw.items()}
    else:
        mapping = {
            "tech-minilm": str(models_dir / "tech-minilm"),
            "cook-minilm": str(models_dir / "cook-minilm"),
            "charge-minilm": str(models_dir / "charge-minilm"),
        }

    embedders = {eid: make_st_embedder(path) for eid, path in mapping.items()}
    return EmbedderRegistry(embedders=embedders)


def iter_sef_jsons(sefs_dir: Path) -> Iterable[Path]:
    """Yield SEF JSON artefact files in a deterministic order."""
    if not sefs_dir.exists():
        raise FileNotFoundError(f"SEFs dir not found: {sefs_dir}")
    yield from sorted(sefs_dir.glob("*.json"))


def build_village_from_artefacts(
    *,
    sefs_dir: Path,
    registry: EmbedderRegistry,
    repo_root: Path,
    override_base_sigma2: Optional[float],
    override_scale: Optional[float],
) -> Village:
    """
    Load SEF artefacts and instantiate runtime SEF objects (embedder resolved).

    Note: repo_root is used to resolve relative calibration paths for
    centroid-distance sigma².
    """
    modules: List[SEF] = []

    for p in iter_sef_jsons(sefs_dir):
        art = SEFArtefact.load(p)

        sigma2_params = dict(art.sigma2_params)
        changed = False

        if override_base_sigma2 is not None:
            sigma2_params["base_sigma2"] = float(override_base_sigma2)
            changed = True
        if override_scale is not None:
            sigma2_params["scale"] = float(override_scale)
            changed = True

        if changed:
            print(
                "loaded + overrides:", p.name, art.name, art.sigma2_kind, sigma2_params
            )
        else:
            print("loaded:", p.name, art.name, art.sigma2_kind, sigma2_params)

        # IMPORTANT: embedder_id remains a string identifier.
        art_eff = SEFArtefact(
            name=art.name,
            embedder_id=art.embedder_id,
            meta=art.meta,
            sigma2_kind=art.sigma2_kind,
            sigma2_params=sigma2_params,
            version=art.version,
        )

        embed = cast(Embedder, registry.get(art_eff.embedder_id))
        sigma2 = art_eff.build_sigma2(registry=registry, base_dir=repo_root)

        modules.append(
            SEF(name=art_eff.name, embed=embed, sigma2=sigma2, meta=art_eff.meta)
        )

    return Village(modules=modules)


# -----------------------------
# Evaluation helpers
# -----------------------------


def build_doc_matrix(
    docs: List[str],
    village: Village,
    scout: ScoutRouter,
    pan: Panoramix,
) -> np.ndarray:
    """Embed all docs with the same routing+fusion strategy as queries."""
    embs: List[np.ndarray] = []
    for d in docs:
        potion = pan.brew(d, village=village, scout=scout)
        v = potion.vector
        v = v / (np.linalg.norm(v) + 1e-12)
        embs.append(v)
    return np.stack(embs, axis=0)


def build_doc_mats_by_module(
    docs: List[str], village: Village
) -> Dict[str, np.ndarray]:
    """Build per-module document embedding matrices."""
    mats: Dict[str, np.ndarray] = {}
    for m in village.modules:
        mats[m.name] = np.stack([m.embed(d) for d in docs], axis=0)  # unit vectors
    return mats


def eval_retrieval_hard(
    queries: List[Tuple[str, int]],
    doc_mats_by_module: Dict[str, np.ndarray],
    village: Village,
    pan: Panoramix,
    k: int = 1,
) -> float:
    """Evaluate hard retrieval accuracy at rank k across all queries."""
    ok = 0
    for q, true_id in queries:
        potion = pan.brew(q, village=village, scout=ScoutRouter(mode="hard"))
        # pick the routed module (hard => one-hot)
        routed = max(potion.weights.items(), key=lambda kv: kv[1])[0]

        sims = doc_mats_by_module[routed] @ potion.vector  # dot == cos (unit vecs)
        topk = np.argsort(-sims)[:k]
        ok += int(true_id in topk)
    return ok / len(queries)


def pretty(weights: Dict[str, float]) -> str:
    """Stable pretty-print for fusion weights."""
    return "{ " + ", ".join(f"{k}: {v:.3f}" for k, v in weights.items()) + " }"


def dump_sigma2(village: Village, query: str) -> None:
    """Print per-module sigma² values for a given query."""
    vals = {m.name: m.sigma2_for(query) for m in village.modules}
    print("      sigma2:", vals)


def cos(u: np.ndarray, v: np.ndarray) -> float:
    """Cosine similarity for debugging."""
    return float(u @ v / ((np.linalg.norm(u) * np.linalg.norm(v)) + 1e-12))


# -----------------------------
# Main
# -----------------------------


# pylint: disable=too-many-locals,too-many-statements
def main() -> None:
    """Run mixed-domain retrieval evaluation with various routing+fusion strategies."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--sefs-dir", type=str, default="artefacts/sefs")
    parser.add_argument("--models-dir", type=str, default="models")
    parser.add_argument("--registry-json", type=str, default="")
    parser.add_argument(
        "--repo-root",
        type=str,
        default="",
        help="Base directory for resolving relative calibration_path entries.",
    )
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--override-base-sigma2", type=float, default=None)
    parser.add_argument("--override-scale", type=float, default=None)
    parser.add_argument(
        "--debug-sims",
        action="store_true",
        help="Print extra cosine similarity diagnostics for a couple of docs/queries.",
    )
    args = parser.parse_args()

    sefs_dir = Path(args.sefs_dir)
    models_dir = Path(args.models_dir)
    registry_json = Path(args.registry_json) if args.registry_json else None

    # If scripts/ lives at repo_root/scripts/, this makes repo_root the parent of scripts/.
    default_repo_root = Path(__file__).resolve().parent.parent
    repo_root = Path(args.repo_root).resolve() if args.repo_root else default_repo_root

    registry = load_embedder_registry(
        models_dir=models_dir, registry_json=registry_json
    )
    village = build_village_from_artefacts(
        sefs_dir=sefs_dir,
        registry=registry,
        repo_root=repo_root,
        override_base_sigma2=args.override_base_sigma2,
        override_scale=args.override_scale,
    )

    corpus = build_toy_corpus(british_spelling=True)

    print()
    print_doc_index(corpus)
    print()

    strategies: List[Tuple[str, ScoutRouter, Panoramix]] = [
        ("hard", ScoutRouter(mode="hard"), Panoramix(fuser=MeanFuser())),
        ("mean", ScoutRouter(mode="all"), Panoramix(fuser=MeanFuser())),
        ("kalman", ScoutRouter(mode="all"), Panoramix(fuser=KalmanorixFuser())),
    ]

    if args.debug_sims:

        def _cos_from_dot(dot: float, qn: float, dn: float) -> float:
            return float(dot / ((qn * dn) + 1e-12))

        def debug_docdoc(label: str, a: int, b: int) -> None:
            print(f"\n== Debug sims (doc-doc, per specialist): {label} ==")
            print(f"doc{a}: {corpus.docs[a]}")
            print(f"doc{b}: {corpus.docs[b]}")
            for m in village.modules:
                za = m.embed(corpus.docs[a])
                zb = m.embed(corpus.docs[b])
                print(f"{m.name:>4}  doc{a} vs doc{b}: {cos(za, zb): .3f}")
            print()

        def debug_per_specialist(label: str, q: str, doc_ids: list[int]) -> None:
            print(f"\n== Debug sims (per specialist): {label} ==")
            print(f'query: "{q}"')
            for m in village.modules:
                zq = m.embed(q)
                for doc_id in doc_ids:
                    zd = m.embed(corpus.docs[doc_id])
                    similarity = cos(zq, zd)
                    doc_preview = corpus.docs[doc_id]
                    print(
                        f"{m.name:>4}  q vs doc{doc_id:>2}: {similarity: .3f}  :: {doc_preview}"
                    )
            print()

        # pylint: disable=too-many-locals
        def debug_fused(
            label: str,
            q: str,
            doc_ids: list[int],
            *,
            doc_embs_all: np.ndarray,
            strategies: list[tuple[str, ScoutRouter, Panoramix]],
        ) -> None:
            """
            Compare query against *eval's* doc index (doc_embs_all).
            Prints dot (ranking score) + cosine (normalized) per doc_id.
            """
            print(f"\n== Debug sims (fused, matches eval doc index): {label} ==")
            print(f'query: "{q}"')

            # Precompute doc norms once for cosine conversion
            doc_norms = np.linalg.norm(doc_embs_all, axis=1)

            for name, scout, pan in strategies:
                potion = pan.brew(q, village=village, scout=scout)
                qv = potion.vector
                qn = float(np.linalg.norm(qv))

                # Ranking scores (this is what eval uses)
                dots = doc_embs_all @ qv

                # Grab the requested docs
                parts = []
                for did in doc_ids:
                    dot = float(dots[did])
                    cosv = _cos_from_dot(dot, qn, float(doc_norms[did]))
                    parts.append((did, dot, cosv))

                # Sort by dot to report top/2nd/margin within this subset
                parts_sorted = sorted(parts, key=lambda t: t[1], reverse=True)
                top_did, top_dot, _ = parts_sorted[0]
                if len(parts_sorted) > 1:
                    snd_did, snd_dot, _ = parts_sorted[1]
                    margin = top_dot - snd_dot
                    suffix = (
                        f", 2nd=doc{snd_did} (dot={snd_dot:.3f}), margin={margin:.3f}"
                    )
                else:
                    suffix = ""

                row = (
                    "  "
                    + f"{name:>5}  "
                    + "  ".join(
                        f"doc{did} dot={dot:.3f} cos={cosv:.3f}"
                        for (did, dot, cosv) in parts
                    )
                )
                row += f"  -> top=doc{top_did} (dot={top_dot:.3f}){suffix}"
                print(row)
            print()

        # --- choose debug targets ---
        thermal_q = "thermal load spikes cause overheating like an oven's heat (reduce power draw)"
        thermal_q_no = "thermal load spikes cause overheating (reduce power draw)"
        thermal_docs = [11, 14, 15]

        reduce_bg_q = (
            "reduce background activity to extend battery life — like reducing a sauce"
        )
        reduce_bg_docs = [7, 8, 10, 14, 16]

        # Build eval's doc index (must match eval)
        doc_scout = ScoutRouter(mode="all")
        doc_embs_all = build_doc_matrix(
            corpus.docs,
            village=village,
            scout=doc_scout,
            pan=Panoramix(fuser=MeanFuser()),
        )

        doc_mats_by_module = build_doc_mats_by_module(corpus.docs, village)

        for name, scout, pan in strategies:
            if name == "hard":
                r1 = eval_retrieval_hard(
                    corpus.queries, doc_mats_by_module, village, pan, k=1
                )
                rk = eval_retrieval_hard(
                    corpus.queries, doc_mats_by_module, village, pan, k=args.k
                )
            else:
                r1 = eval_retrieval(
                    corpus.queries, doc_embs_all, village, scout, pan, k=1
                )
                rk = eval_retrieval(
                    corpus.queries, doc_embs_all, village, scout, pan, k=args.k
                )

            print(f"{name:>6}  Recall@1={r1:.3f}  Recall@{args.k}={rk:.3f}")

        # Optional: sanity check norms
        norms = np.linalg.norm(doc_embs_all, axis=1)
        print(
            f"doc_embs_all norms (min/mean/max): {norms.min()} {norms.mean()} {norms.max()}"
        )

        # Per-specialist doc-doc + query-doc
        debug_docdoc("doc11 vs doc14", 11, 14)
        debug_per_specialist("thermal query vs key docs", thermal_q, thermal_docs)
        debug_fused(
            "thermal query vs docs",
            thermal_q,
            thermal_docs,
            doc_embs_all=doc_embs_all,
            strategies=strategies,
        )

        debug_per_specialist(
            "thermal (no oven analogy) vs key docs", thermal_q_no, thermal_docs
        )
        debug_fused(
            "thermal (no oven analogy) vs key docs",
            thermal_q_no,
            thermal_docs,
            doc_embs_all=doc_embs_all,
            strategies=strategies,
        )

        debug_per_specialist(
            "reduce-background query vs key docs", reduce_bg_q, reduce_bg_docs
        )
        debug_fused(
            "reduce-background query vs key docs",
            reduce_bg_q,
            reduce_bg_docs,
            doc_embs_all=doc_embs_all,
            strategies=strategies,
        )

    # Learned gate baseline expects exactly 2 modules by name
    if len(village.modules) >= 2:
        a_name = village.modules[0].name
        b_name = village.modules[1].name
        gate = LearnedGateFuser(
            module_a=a_name,
            module_b=b_name,
            n_features=128,
            steps=400,
        )
        gate.fit(
            texts=[
                "battery life smartphone charger",
                "cpu gpu laptop drivers performance",
                "braise stew garlic onion simmer",
                "slow cooker recipe simmer oven",
            ],
            y=[1, 1, 0, 0],
        )
        strategies.append(("gate", ScoutRouter(mode="all"), Panoramix(fuser=gate)))

    dim = int(village.modules[0].embed("probe").shape[0])

    print("== Evaluation from SEF artefacts ==")
    print(f"repo_root: {repo_root}")
    print(f"sefs: {sefs_dir.resolve()}")
    print(f"models: {models_dir.resolve()}")
    if registry_json is not None:
        print(f"registry: {registry_json.resolve()}")
    print(f"modules: {village.list()}")
    print(f"docs: {len(corpus.docs)}, queries: {len(corpus.queries)}, dim: {dim}")
    print()

    # Build ONE doc matrix with "all" routing so documents always include all specialists.
    doc_scout = ScoutRouter(mode="all")

    doc_embs_all = build_doc_matrix(
        corpus.docs, village=village, scout=doc_scout, pan=Panoramix(fuser=MeanFuser())
    )
    # If you want doc embeddings to match each strategy’s fuser instead:
    # doc_embs_by_fuser = {name: build_doc_matrix(... doc_scout, pan) ...}
    # but start with the simplest: shared Mean-fused doc index.

    for name, scout, pan in strategies:
        r1 = eval_retrieval(corpus.queries, doc_embs_all, village, scout, pan, k=1)
        rk = eval_retrieval(corpus.queries, doc_embs_all, village, scout, pan, k=args.k)
        print(f"{name:>6}  Recall@1={r1:.3f}  Recall@{args.k}={rk:.3f}")

    print()
    print("Mixed-query fusion weights + top-1 predictions:")

    doc_mats = {name: doc_embs_all for name, _scout, _pan in strategies}

    print(
        "doc_embs_all norms (min/mean/max):",
        float(np.linalg.norm(doc_embs_all, axis=1).min()),
        float(np.linalg.norm(doc_embs_all, axis=1).mean()),
        float(np.linalg.norm(doc_embs_all, axis=1).max()),
    )

    mixed_queries = [q for q, g in zip(corpus.queries, corpus.groups) if g == "mixed"]
    for q, true_id in mixed_queries:
        print(f'  query: "{q}" (true={true_id})')
        dump_sigma2(village, q)
        for name, scout, pan in strategies:
            potion = pan.brew(q, village=village, scout=scout)
            sims = doc_mats[name] @ potion.vector
            pred = int(np.argmax(sims))
            ok = "OK" if pred == true_id else "NO"
            print(f"    {name:>6}: {pretty(potion.weights)}  top1={pred} {ok}")

            if q.startswith("battery lasts all day on my smartphone"):
                top = np.argsort(-sims)[:8]
                print(
                    "      top8:",
                    [(int(i), float(sims[i]), corpus.docs[int(i)]) for i in top],
                )

            if q.startswith("camera pipeline"):
                top = np.argsort(-sims)[:5]
                print(
                    "      top5:",
                    [(int(i), float(sims[i]), corpus.docs[int(i)]) for i in top],
                )
        print()


if __name__ == "__main__":
    main()
