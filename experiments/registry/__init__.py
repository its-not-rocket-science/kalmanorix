"""Benchmark registry package for config-driven experiments."""

from experiments.registry.config_schema import BenchmarkExperimentConfig, load_experiment_config

__all__ = ["BenchmarkExperimentConfig", "load_experiment_config"]
