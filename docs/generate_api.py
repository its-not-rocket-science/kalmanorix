#!/usr/bin/env python3
"""Generate API reference documentation for Kalmanorix.

This script is used by the mkdocs-gen-files plugin to automatically generate
API reference pages from the Python source code.

The generated pages are placed in the 'api-reference/' directory and contain
mkdocstrings directives that pull documentation from the source code.
"""

import mkdocs_gen_files
import os

# Define the API modules to document
# Each entry: (output_path, module_path, title, description)
API_MODULES = [
    (
        "api-reference/village.md",
        "kalmanorix.village",
        "Village API",
        "The `Village` is the container for all available specialists (SEFs) at runtime. "
        "It provides methods to add/remove SEFs, compute domain centroids, and retrieve specialists by name.",
    ),
    (
        "api-reference/panoramix.md",
        "kalmanorix.panoramix",
        "Panoramix API",
        "`Panoramix` is the high‑level fusion orchestrator. It combines a `Village` (specialists), "
        "a `ScoutRouter` (selection), and a `Fuser` (fusion strategy) to produce fused embeddings.",
    ),
    (
        "api-reference/scout-router.md",
        "kalmanorix.scout",
        "Scout Router API",
        "The `ScoutRouter` selects which specialists to consult for a given query. "
        "It supports multiple routing modes: `all` (fusion), `hard` (single specialist), "
        "and `semantic` (domain‑aware selection).",
    ),
    (
        "api-reference/embedder-adapters.md",
        "kalmanorix.embedder_adapters",
        "Embedder Adapters API",
        "Adapter classes and factory functions for third‑party embedders: "
        "Sentence‑Transformers, OpenAI, Cohere, Anthropic, Vertex AI, Azure OpenAI, and Hugging Face Transformers.",
    ),
    (
        "api-reference/kalman-engine.md",
        "kalmanorix.kalman_engine",
        "Kalman Engine API",
        "Core Kalman fusion algorithms and covariance estimation utilities. "
        "Includes low‑level functions for sequential and parallel Kalman updates with diagonal covariance.",
    ),
]


def generate_api_page(output_path: str, module_path: str, title: str, description: str):
    """Generate a single API reference page."""

    # Create the directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Write the markdown content
    with mkdocs_gen_files.open(output_path, "w") as f:
        f.write(f"# {title}\n\n")
        f.write(f"{description}\n\n")

        # Add the mkdocstrings directive
        f.write(f"::: {module_path}\n")
        f.write("    options:\n")
        f.write("      show_root_heading: true\n")
        f.write("      show_source: true\n")
        f.write("      heading_level: 3\n")
        f.write("      show_category_heading: true\n")
        f.write("      merge_init_into_class: true\n")

        # Add a note about auto-generation
        f.write("\n\n*This page is auto‑generated from the source code.*\n")


def main():
    """Generate all API reference pages."""
    print("Generating API documentation...")

    for output_path, module_path, title, description in API_MODULES:
        print(f"  Generating {output_path} for {module_path}")
        generate_api_page(output_path, module_path, title, description)

    print("API documentation generation complete.")


if __name__ == "__main__":
    main()
