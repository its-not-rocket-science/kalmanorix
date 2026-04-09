# Benchmark Registry

Configuration-driven experiment runtime with explicit separation of:

- dataset loading (`datasets.py`)
- model loading (`models.py`)
- fusion/ranking (`fusion.py`)
- evaluation (`evaluation.py`)
- reporting (`reporting.py`)
- orchestration (`runner.py`)

Run an experiment:

```bash
python -m experiments.registry.runner --config experiments/configs/benchmark_registry/synthetic_smoke.json
```
