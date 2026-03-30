# Embedder Adapters API

*TODO: Auto‑generated API documentation for third‑party embedder adapters and factory functions.*

Kalmanorix provides adapters for popular embedding models, making it easy to wrap them as SEFs with constant or centroid‑distance uncertainty.

::: kalmanorix.embedder_adapters
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

## Factory Functions

### Hugging Face Transformers
- `create_huggingface_sef()` – Wrap a Hugging Face transformer model.
- `create_huggingface_sef_with_calibration()` – Add centroid‑distance uncertainty.

### Sentence‑Transformers
- `create_sentence_transformer_sef()` – Wrap a Sentence‑Transformers model.
- `create_sentence_transformer_sef_with_calibration()` – Add centroid‑distance uncertainty.

### Proprietary APIs
- `create_openai_sef()` – OpenAI text‑embedding models.
- `create_cohere_sef()` – Cohere embed models.
- `create_anthropic_sef()` – Anthropic Claude embeddings (when available).
- `create_vertex_ai_sef()` – Google Vertex AI text‑embedding models.
- `create_azure_openai_sef()` – Azure OpenAI embeddings.

Each `create_*_sef_with_calibration()` function computes a domain centroid from a provided corpus and uses `CentroidDistanceSigma2` for query‑dependent uncertainty.

*TODO: Add API‑key configuration examples, error‑handling tips, and cost‑tracking advice.*
