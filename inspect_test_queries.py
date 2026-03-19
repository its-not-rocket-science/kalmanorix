#!/usr/bin/env python3
import json
from pathlib import Path

test_set_path = Path("experiments/outputs/milestone_2_1/test_set.json")
with open(test_set_path, "r", encoding="utf-8") as f:
    test_data = json.load(f)

print(f"Total documents: {len(test_data['documents'])}")
print(f"Total queries: {len(test_data['queries'])}")

# Show first 10 queries
for i, q in enumerate(test_data["queries"][:10]):
    print(f"Query {i + 1}: {q['query'][:80]}... (true doc {q['true_doc_id']})")
    doc = test_data["documents"][q["true_doc_id"]]
    print(f"  Document: {doc[:80]}...")

# Check domain keywords
medical_keywords = [
    "patient",
    "treatment",
    "medicine",
    "diagnosis",
    "clinical",
    "doctor",
    "hospital",
]
legal_keywords = ["law", "legal", "court", "case", "judge", "attorney", "statute"]

print("\nKeyword analysis:")
medical_count = 0
legal_count = 0
for q in test_data["queries"]:
    query = q["query"].lower()
    if any(kw in query for kw in medical_keywords):
        medical_count += 1
    if any(kw in query for kw in legal_keywords):
        legal_count += 1
print(f"Queries with medical keywords: {medical_count}/{len(test_data['queries'])}")
print(f"Queries with legal keywords: {legal_count}/{len(test_data['queries'])}")

# Check documents
print("\nDocument keyword analysis:")
med_docs = 0
leg_docs = 0
for doc in test_data["documents"][:100]:  # sample
    doc_low = doc.lower()
    if any(kw in doc_low for kw in medical_keywords):
        med_docs += 1
    if any(kw in doc_low for kw in legal_keywords):
        leg_docs += 1
print(f"Documents with medical keywords: {med_docs}/100")
print(f"Documents with legal keywords: {leg_docs}/100")

# Let's see a few medical pure documents
print("\nLooking for medical pure documents...")
for i, doc in enumerate(test_data["documents"][:50]):
    doc_low = doc.lower()
    if any(kw in doc_low for kw in medical_keywords):
        print(f"Doc {i}: {doc[:100]}...")
        # Find query that points to this doc
        for q in test_data["queries"]:
            if q["true_doc_id"] == i:
                print(f"  Query: {q['query'][:80]}...")
                break
