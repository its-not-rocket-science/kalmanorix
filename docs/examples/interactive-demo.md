# Interactive Demo

The interactive demo (`examples/interactive_demo.ipynb`) is a Jupyter notebook that provides a visual, hands‑on exploration of Kalmanorix's core concepts. Using `ipywidgets` and `matplotlib`, it lets you experiment with semantic routing, dynamic thresholding, and Kalman fusion in real time, making it ideal for education, debugging, and prototyping.

## Features

- **Semantic Routing with Dynamic Thresholding**: Interactive exploration of four threshold heuristics (`threshold_top_k`, `threshold_relative_to_max`, `threshold_adaptive_spread`, `threshold_query_length_adaptive`) with live parameter adjustment.
- **Kalman Fusion Comparison**: Side‑by‑side comparison of `MeanFuser`, `KalmanorixFuser`, and `LearnedGateFuser` strategies, showing how uncertainty estimates influence fusion weights.
- **Real‑time Visualization**: Four‑panel dashboard displaying similarity scores, selected specialists, keyword matches, and threshold parameters for each query.
- **Toy Specialists with Domain Centroids**: Three keyword‑sensitive specialists (tech, cooking, medical) with pre‑computed domain centroids for semantic routing.
- **Interactive Widgets**: Sliders, dropdowns, and text areas that update visualizations instantly as you type or adjust parameters.

## Requirements

The notebook requires the optional `viz` dependency group, which includes `matplotlib` and `ipywidgets`. Install it together with the development dependencies:

```bash
pip install -e ".[dev,viz]"
```

If you only need to run the notebook (not develop Kalmanorix itself), install the minimal dependencies:

```bash
pip install -e ".[viz]"
pip install notebook  # Jupyter notebook runtime
```

For a completely isolated environment, consider using Conda or a virtual environment.

## Launching the Notebook

Start Jupyter Notebook or JupyterLab from the project root directory:

```bash
jupyter notebook examples/interactive_demo.ipynb
```

Or with JupyterLab:

```bash
jupyter lab examples/interactive_demo.ipynb
```

If you encounter import errors, ensure the parent directory is in Python's path (the notebook already includes `sys.path.insert(0, "..")` for development mode).

## Notebook Walkthrough

### 1. Setup and Imports

The notebook begins by importing Kalmanorix's public API and visualization utilities:

```python
from kalmanorix import (
    SEF, Village, ScoutRouter, Panoramix,
    MeanFuser, KalmanorixFuser, LearnedGateFuser,
    threshold_top_k, threshold_relative_to_max,
    threshold_adaptive_spread, threshold_query_length_adaptive,
)
from kalmanorix.visualization import (
    plot_embedding_with_uncertainty,
    plot_fusion_weights,
)
```

It also defines a toy `KeywordEmbedder` class—a deterministic embedder that boosts its output when the query contains specific keywords. This allows the demo to run without external model dependencies.

### 2. Creating Toy Specialists

Three specialists are created, each with its own domain keywords and `KeywordSigma2` uncertainty:

```python
tech = SEF(
    name="tech",
    embed=tech_embedder,
    sigma2=KeywordSigma2(tech_keywords, in_domain_sigma2=0.2, out_domain_sigma2=2.5),
    meta={"domain": "technology", "keywords": list(tech_keywords)},
)
# ... similarly for cook and medical
```

Domain centroids are computed from calibration texts using `.with_domain_centroid()`, enabling semantic routing:

```python
tech_with_centroid = tech.with_domain_centroid(tech_calibration)
cook_with_centroid = cook.with_domain_centroid(cook_calibration)
medical_with_centroid = medical.with_domain_centroid(medical_calibration)

village = Village([tech_with_centroid, cook_with_centroid, medical_with_centroid])
```

### 3. Semantic Routing with Dynamic Thresholding

This section introduces semantic routing: selecting specialists based on cosine similarity between the query embedding and each specialist's domain centroid.

Four threshold heuristics are demonstrated:

- **`threshold_top_k`**: Select the top‑k most similar specialists.
- **`threshold_relative_to_max`**: Threshold = `fraction × max(similarities)`.
- **`threshold_adaptive_spread`**: Adjusts threshold based on the spread of similarity values (tight clusters → higher threshold).
- **`threshold_query_length_adaptive`**: Longer queries receive a lower threshold (more specific).

The interactive widget lets you adjust the query text, threshold type, and heuristic parameters while observing:

1. **Similarity bars** (green = selected, red = below threshold)
2. **Selected modules** count and names
3. **Keyword matches** in the query for each specialist
4. **Parameter statistics** (query length, max similarity, standard deviation)

Example query to try:
> “This smartphone battery lasts longer than a slow cooker braise but patient recovery is important.”

### 4. Fusion Strategies Comparison

Once specialists are selected, the notebook compares three fusion strategies:

- **`MeanFuser`** – uniform averaging (baseline)
- **`KalmanorixFuser`** – precision‑weighted fusion using diagonal covariance (lower uncertainty → higher weight)
- **`LearnedGateFuser`** – a tiny logistic‑regression gate trained on bag‑of‑words features (demonstrates learned routing)

You can toggle semantic routing on/off and adjust the similarity threshold. The visualization shows:

1. **Fusion weights** as a bar chart
2. **Embedding with uncertainty** (±1σ) for the first specialist
3. **Fusion metadata** (strategy, routing mode, selected modules, total weight)

Example query for fusion:
> “Patient needs smartphone for telemedicine appointments and recipe suggestions.”

### 5. Conclusion and Next Steps

The notebook concludes with suggested queries to explore different routing and fusion behaviours, plus guidance for production usage:

- Replace toy embedders with real models using adapters from `embedder_adapters.py`
- Compute domain centroids on representative domain texts
- Choose a threshold heuristic based on your use case
- Monitor routing accuracy and fusion performance

## Educational Use Cases

The interactive demo is designed for:

- **Students & Researchers**: Understand Kalman filtering, uncertainty‑weighted fusion, and semantic routing through hands‑on experimentation.
- **ML Engineers**: Debug why a particular query receives unexpected weights, or prototype new threshold heuristics.
- **Product Managers**: Explore the trade‑offs between selectivity (fewer specialists) and coverage (more specialists) in a multi‑specialist system.

## Extending the Notebook

You can add new cells to:

- **Implement a custom `Fuser`** and compare it to the built‑in strategies.
- **Load real specialists** from pickle files or `SEFModel` directories.
- **Profile memory and latency** as the number of specialists grows.
- **Integrate real embedders** (Sentence Transformers, OpenAI, Cohere) using the factory functions in `embedder_adapters.py`.
- **Create new threshold heuristics** by implementing the `ThresholdHeuristic` protocol.

## Example Outputs

While the notebook produces live, interactive plots, here is a textual summary of what you might see for a mixed‑domain query:

```
Query: 'This smartphone battery lasts longer than a slow cooker braise'
Threshold: 0.650 (relative_to_max, fraction=0.8)
Selected: ['tech', 'cook']

Fusion weights (KalmanorixFuser):
  tech: 0.876
  cook: 0.124
```

The similarity bars would show high scores for `tech` and `cook`, lower for `medical`. The keyword‑match plot would indicate matches for technology and cooking keywords.

## Troubleshooting

### Matplotlib Not Displaying Plots
Ensure the notebook includes `%matplotlib inline` (it does) and that you are running a Jupyter kernel, not a plain Python interpreter.

### Widgets Not Updating
If sliders/dropdowns do not trigger updates, check that `ipywidgets` is installed and that you are viewing the notebook in a supported frontend (Jupyter Notebook ≥6.0, JupyterLab ≥3.0).

### Import Errors
If `kalmanorix` cannot be imported, make sure you have installed the package (`pip install -e .`) and that the notebook's `sys.path.insert(0, "..")` is uncommented when running locally.

### Slow Performance with Real Models
The notebook uses CPU by default. To enable GPU acceleration for Hugging Face models, modify the embedder creation to include `device="cuda"`.

## Running Online (Colab / Binder)

To run the notebook without local installation, you can upload it to:

- **Google Colab**: Upload `interactive_demo.ipynb` and install Kalmanorix with:
  ```python
  !pip install git+https://github.com/its-not-rocket-science/kalmanorix
  !pip install matplotlib ipywidgets
  ```
- **Binder**: Not yet configured; a `requirements.txt` or `environment.yml` would need to be added to the repository.

## Further Reading

- [Minimal Fusion Example](minimal-fusion.md) – Core Kalmanorix concepts with toy specialists.
- [HuggingFace Integration](huggingface-integration.md) – Wrapping real transformer models as SEFs.
- [API Server Example](api-server.md) – Production‑ready FastAPI server for remote fusion.
- [Semantic Routing API](../api-reference/scout-router.md) – Detailed documentation of routing and threshold heuristics.
