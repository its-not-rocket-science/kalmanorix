"""Kalmanorix public API."""

from kalmanorix.sef_io import SEFArtefact


def test_sef_artefact_roundtrip(tmp_path):
    """Test saving and loading a SEF artefact with query-dependent sigma2."""
    art = SEFArtefact(
        name="tech",
        embedder_id="toy-tech-v1",
        meta={"domain": "tech"},
        sigma2_kind="keyword",
        sigma2_params={
            "keywords": ["battery"],
            "in_domain_sigma2": 0.2,
            "out_domain_sigma2": 2.0,
        },
    )

    p = tmp_path / "tech.sef.json"
    art.save(p)

    art2 = SEFArtefact.load(p)
    assert art2.name == art.name
    assert art2.embedder_id == art.embedder_id
    assert art2.meta["domain"] == "tech"

    sigma2 = art2.build_sigma2(registry=None)
    assert sigma2("battery life") < sigma2("braise stew")
