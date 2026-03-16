"""
Compute and energy tracking for training experiments.

Estimates FLOPs, wall time, and GPU energy consumption (if available)
to verify compute equivalence between specialist and monolithic training.
"""

# pylint: disable=broad-exception-caught

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Union


@dataclass
# pylint: disable=too-many-instance-attributes
class ComputeMetrics:
    """Container for compute and energy metrics."""

    # FLOPs estimation
    total_flops: int = 0
    parameters: int = 0
    tokens_processed: int = 0

    # Time tracking
    wall_time_seconds: float = 0.0
    cpu_time_seconds: float = 0.0  # If available

    # GPU metrics (optional)
    gpu_energy_joules: Optional[float] = None
    gpu_memory_mb: Optional[int] = None
    gpu_utilization_percent: Optional[float] = None

    # Derived metrics
    flops_per_second: float = field(init=False)
    tokens_per_second: float = field(init=False)

    def __post_init__(self) -> None:
        """Compute derived metrics."""
        if self.wall_time_seconds > 0:
            self.flops_per_second = self.total_flops / self.wall_time_seconds
            self.tokens_per_second = self.tokens_processed / self.wall_time_seconds
        else:
            self.flops_per_second = 0.0
            self.tokens_per_second = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        data = {
            "total_flops": self.total_flops,
            "parameters": self.parameters,
            "tokens_processed": self.tokens_processed,
            "wall_time_seconds": self.wall_time_seconds,
            "cpu_time_seconds": self.cpu_time_seconds,
            "flops_per_second": self.flops_per_second,
            "tokens_per_second": self.tokens_per_second,
        }

        # Add GPU metrics if present
        if self.gpu_energy_joules is not None:
            data["gpu_energy_joules"] = self.gpu_energy_joules
        if self.gpu_memory_mb is not None:
            data["gpu_memory_mb"] = self.gpu_memory_mb
        if self.gpu_utilization_percent is not None:
            data["gpu_utilization_percent"] = self.gpu_utilization_percent

        return data

    def to_json(self, indent: int = 2) -> str:
        """Return JSON string representation."""
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, path: Union[str, Path]) -> None:
        """Save metrics to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)


# pylint: disable=too-many-instance-attributes
class ComputeTracker:
    """
    Track compute usage during training.

    Estimates FLOPs using the standard transformer FLOPs formula:
        FLOPs ≈ 2 * parameters * tokens_processed * (forward + backward)
              = 2 * P * T * (1 + 2)  [assuming backward is 2× forward]
              = 6 * P * T

    For simplicity, we use 6× multiplier for training (forward+backward).
    For inference-only, use 2× multiplier.

    Time tracking uses perf_counter for wall time.
    GPU energy tracking requires pynvml (optional).
    """

    def __init__(
        self,
        model_parameters: int,
        track_gpu: bool = False,
        gpu_index: int = 0,
    ) -> None:
        """
        Initialize compute tracker.

        Parameters
        ----------
        model_parameters : int
            Number of trainable parameters in the model.
        track_gpu : bool
            Whether to track GPU energy (requires pynvml).
        gpu_index : int
            GPU device index to monitor.
        """
        self.parameters = model_parameters
        self.track_gpu = track_gpu
        self.gpu_index = gpu_index

        # State
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.tokens_processed: int = 0
        self.gpu_energy_start: Optional[float] = None
        self.gpu_energy_end: Optional[float] = None
        self.gpu_memory_mb: Optional[int] = None
        self.gpu_utilization: Optional[float] = None

        # Initialize GPU monitoring if requested
        self.pynvml = None
        self.handle = None
        if track_gpu:
            try:
                import pynvml  # pylint: disable=import-outside-toplevel

                self.pynvml = pynvml
                pynvml.nvmlInit()
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            except ImportError:
                print("Warning: pynvml not installed, GPU tracking disabled")
                self.track_gpu = False
            except Exception as e:
                print(f"Warning: Failed to initialize GPU tracking: {e}")
                self.track_gpu = False

    def __enter__(self) -> ComputeTracker:
        """Start tracking."""
        self.start_time = time.perf_counter()

        if self.track_gpu and self.pynvml and self.handle:
            try:
                # Get initial GPU energy
                self.gpu_energy_start = self.pynvml.nvmlDeviceGetTotalEnergyConsumption(
                    self.handle
                )
                # Sample GPU memory
                mem_info = self.pynvml.nvmlDeviceGetMemoryInfo(self.handle)
                self.gpu_memory_mb = mem_info.used // (1024 * 1024)
            except Exception:  # pylint: disable=broad-exception-caught
                pass

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Stop tracking and record final metrics."""
        self.end_time = time.perf_counter()

        if self.track_gpu and self.pynvml and self.handle:
            try:
                self.gpu_energy_end = self.pynvml.nvmlDeviceGetTotalEnergyConsumption(
                    self.handle
                )
                # Get GPU utilization (average over tracking period)
                util = self.pynvml.nvmlDeviceGetUtilizationRates(self.handle)
                self.gpu_utilization = util.gpu
            except Exception:
                pass

        # Cleanup GPU monitoring
        if self.pynvml:
            try:
                self.pynvml.nvmlShutdown()
            except Exception:
                pass

    def add_tokens(self, tokens: int) -> None:
        """Record processed tokens (forward + backward)."""
        self.tokens_processed += tokens

    def add_batch(self, batch_size: int, sequence_length: int) -> None:
        """Record a batch of tokens (assumes each token processed once)."""
        self.add_tokens(batch_size * sequence_length)

    def get_metrics(self) -> ComputeMetrics:
        """
        Compute metrics based on recorded data.

        Returns
        -------
        ComputeMetrics
            Computed metrics.
        """
        if self.start_time is None or self.end_time is None:
            raise RuntimeError("ComputeTracker not used as context manager")

        wall_time = self.end_time - self.start_time

        # Estimate FLOPs: 6 * parameters * tokens (forward+backward)
        total_flops = 6 * self.parameters * self.tokens_processed

        # GPU energy
        gpu_energy = None
        if (
            self.track_gpu
            and self.gpu_energy_start is not None
            and self.gpu_energy_end is not None
        ):
            # Energy in millijoules, convert to joules
            gpu_energy = (self.gpu_energy_end - self.gpu_energy_start) / 1000.0

        return ComputeMetrics(
            total_flops=total_flops,
            parameters=self.parameters,
            tokens_processed=self.tokens_processed,
            wall_time_seconds=wall_time,
            cpu_time_seconds=0.0,  # Not tracked
            gpu_energy_joules=gpu_energy,
            gpu_memory_mb=self.gpu_memory_mb,
            gpu_utilization_percent=self.gpu_utilization,
        )


# Utility functions for common model parameter counts
def estimate_parameters(model_name: str) -> int:
    """
    Estimate parameter count for common sentence transformer models.

    Parameters
    ----------
    model_name : str
        Hugging Face model identifier.

    Returns
    -------
    int
        Estimated parameter count.
    """
    # Common models and their approximate parameter counts
    known_models = {
        "sentence-transformers/all-MiniLM-L6-v2": 22_000_000,  # 22M
        "sentence-transformers/all-MiniLM-L12-v2": 33_000_000,  # 33M
        "sentence-transformers/all-mpnet-base-v2": 109_000_000,  # 109M
        "bert-base-uncased": 110_000_000,  # 110M
        "roberta-base": 125_000_000,  # 125M
    }

    for key, params in known_models.items():
        if key in model_name:
            return params

    # Default fallback for MiniLM-like models
    if "MiniLM" in model_name:
        if "L6" in model_name:
            return 22_000_000
        if "L12" in model_name:
            return 33_000_000

    # Very rough estimate based on embedding dimension
    # Assume 6 layers, hidden size ~ embedding_dim * 4
    if "embedding_dimension" in model_name:
        # Not a real parameter, but for testing
        return 22_000_000

    print(f"Warning: Unknown model {model_name}, using default 22M parameters")
    return 22_000_000


@contextmanager
def track_compute(
    model_name: str,
    output_path: Optional[Union[str, Path]] = None,
    track_gpu: bool = False,
) -> Iterator[ComputeTracker]:
    """
    Context manager for compute tracking.

    Parameters
    ----------
    model_name : str
        Model identifier for parameter estimation.
    output_path : Optional[Union[str, Path]]
        If provided, metrics will be saved to this path on exit.
    track_gpu : bool
        Whether to track GPU energy.

    Yields
    ------
    ComputeTracker
        Tracker instance to record token batches.
    """
    params = estimate_parameters(model_name)
    tracker = ComputeTracker(params, track_gpu=track_gpu)

    with tracker as t:
        yield t

    # Save metrics if path provided
    if output_path:
        metrics = t.get_metrics()
        metrics.save(output_path)
