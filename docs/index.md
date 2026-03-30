# Kalmanorix: Efficient Specialist Fusion with Kalman Filtering

Kalmanorix is a research framework for fusing embeddings from multiple domain-specialist models using Kalman filtering. The core hypothesis is that a fused ensemble of specialists can outperform monolithic models while being computationally efficient (KEFF: Kalman Ensemble of Fusion-Frugal specialists).

## 🚀 Key Result: Semantic Routing Efficiency

**Semantic routing achieves 65% average FLOPs reduction** by selecting only relevant domain specialists per query. Benchmark results show:

- **Selection efficiency**: 35% (specialists selected/loaded) across 3-20 specialists
- **Latency reduction**: Up to 34% when routing overhead < cost of extra specialists
- **Dynamic thresholding**: Ensures at least one specialist selected even for ambiguous queries

This validates the Kalmanorix hypothesis that modular specialist fusion with intelligent routing can be computationally competitive with monolithic models.

## About This Documentation

This documentation provides comprehensive guides, API reference, and research findings for the Kalmanorix framework.

### Getting Started

New to Kalmanorix? Start here:

- [Installation](getting-started/installation.md) - Install the framework and dependencies
- [Quickstart](getting-started/quickstart.md) - Your first fusion pipeline in 5 minutes
- [Examples](getting-started/examples.md) - Overview of available examples

### Core Concepts

Understand the key ideas behind Kalmanorix:

- [Kalman Fusion](concepts/kalman-fusion.md) - Mathematical foundations of Kalman filtering for embeddings
- [Specialist Embeddings](concepts/specialist-embeddings.md) - Domain-specialist models and SEF format
- [Uncertainty Estimation](concepts/uncertainty-estimation.md) - Variance estimation and calibration
- [Alignment](concepts/alignment.md) - Procrustes alignment for embedding spaces

### API Reference

Complete API documentation for developers:

- [Village](api-reference/village.md) - SEF container and domain centroid computation
- [Panoramix](api-reference/panoramix.md) - High-level fusion orchestration
- [Scout Router](api-reference/scout-router.md) - Semantic routing and model selection
- [Embedder Adapters](api-reference/embedder-adapters.md) - Third-party model integrations
- [Kalman Engine](api-reference/kalman-engine.md) - Core Kalman fusion algorithms

### Examples & Tutorials

Step-by-step tutorials and example applications:

- [Minimal Fusion](examples/minimal-fusion.md) - Basic fusion with toy specialists
- [HuggingFace Integration](examples/huggingface-integration.md) - Transformer model specialists
- [API Server](examples/api-server.md) - REST API for remote fusion
- [Interactive Demo](examples/interactive-demo.md) - Jupyter notebook exploration

### Research & Experiments

Research findings and experimental results:

- [Experiments](research/experiments.md) - Experimental methodology and benchmarks
- [Results](research/results.md) - Performance results and analysis
- [Milestones](milestones/index.md) - Detailed milestone reports

### Contributing

Want to contribute to Kalmanorix?

- [Development](contributing/development.md) - Development setup and guidelines
- [Testing](contributing/testing.md) - Testing philosophy and practices
- [Roadmap](contributing/roadmap.md) - Project roadmap and future plans

---

**Quick Links**: [GitHub Repository](https://github.com/its-not-rocket-science/kalmanorix) | [PyPI](https://pypi.org/project/kalmanorix/) | [Issue Tracker](https://github.com/its-not-rocket-science/kalmanorix/issues)
