#!/usr/bin/env python3
"""
Check query domain distribution in test set.
"""

import json
from pathlib import Path


def classify_query(query):
    """Heuristic classification based on keywords."""
    medical_keywords = {
        "patient",
        "treatment",
        "medicine",
        "diagnosis",
        "therapy",
        "symptom",
        "condition",
        "prescription",
        "dosage",
        "recovery",
        "clinical",
        "medical",
        "therapeutic",
        "diagnostic",
        "preventive",
        "chronic",
        "acute",
        "benign",
        "malignant",
        "symptomatic",
    }
    legal_keywords = {
        "law",
        "case",
        "court",
        "judge",
        "attorney",
        "statute",
        "regulation",
        "contract",
        "liability",
        "plaintiff",
        "legal",
        "judicial",
        "constitutional",
        "criminal",
        "civil",
        "liable",
        "admissible",
        "precedential",
        "binding",
        "unlawful",
    }

    query_lower = query.lower()
    has_medical = any(kw in query_lower for kw in medical_keywords)
    has_legal = any(kw in query_lower for kw in legal_keywords)

    if has_medical and not has_legal:
        return "medical"
    elif has_legal and not has_medical:
        return "legal"
    elif has_medical and has_legal:
        return "mixed"
    else:
        return "general"


def main():
    experiment_dir = Path("experiments/outputs/milestone_2_1")

    # Load test set
    with open(experiment_dir / "test_set.json", "r", encoding="utf-8") as f:
        test_data = json.load(f)

    test_queries = [q["query"] for q in test_data["queries"]]

    print(f"Total queries: {len(test_queries)}")

    # Classify all queries
    domain_counts = {"medical": 0, "legal": 0, "mixed": 0, "general": 0}
    domain_examples = {"medical": [], "legal": [], "mixed": [], "general": []}

    for i, query in enumerate(test_queries[:100]):  # First 100 queries
        domain = classify_query(query)
        domain_counts[domain] += 1
        if domain_counts[domain] <= 3:  # Keep up to 3 examples per domain
            domain_examples[domain].append(query)

    print("\nDomain distribution:")
    for domain, count in domain_counts.items():
        print(f"  {domain}: {count} ({count / len(test_queries[:100]) * 100:.1f}%)")

    print("\nExamples:")
    for domain, examples in domain_examples.items():
        if examples:
            print(f"\n{domain}:")
            for ex in examples:
                print(f"  - {ex[:80]}...")

    # Check if any queries contain legal keywords
    print("\nSearching for legal keywords...")
    legal_queries = []
    for query in test_queries[:100]:
        if any(
            kw in query.lower()
            for kw in ["legal", "law", "case", "court", "judge", "attorney"]
        ):
            legal_queries.append(query)

    print(f"Found {len(legal_queries)} queries with legal keywords:")
    for q in legal_queries[:5]:
        print(f"  - {q[:80]}...")


if __name__ == "__main__":
    main()
