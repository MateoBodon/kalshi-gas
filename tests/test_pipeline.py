from pathlib import Path

from kalshi_gas.config import load_config
from kalshi_gas.data.assemble import assemble_dataset
from kalshi_gas.etl.pipeline import run_all_etl
from kalshi_gas.pipeline import run_pipeline
from kalshi_gas.models.ensemble import EnsembleModel


def test_assemble_dataset_produces_rows() -> None:
    cfg = load_config()
    run_all_etl(cfg)
    dataset = assemble_dataset(cfg)
    assert not dataset.empty
    assert {"regular_gas_price", "rbob_settle", "kalshi_prob"}.issubset(dataset.columns)


def test_ensemble_predictions_shape() -> None:
    cfg = load_config()
    run_all_etl(cfg)
    dataset = assemble_dataset(cfg)
    ensemble = EnsembleModel(weights=cfg.model.ensemble_weights)
    ensemble.fit(dataset)
    preds = ensemble.predict(dataset)
    assert "ensemble" in preds.columns
    assert len(preds) == len(dataset)


def test_run_pipeline_outputs_build_artifacts() -> None:
    result = run_pipeline()
    assert 0.0 <= float(result["prior_weight"]) <= 1.0
    deck_path = result.get("deck_path")
    assert deck_path is not None and Path(deck_path).exists()
    artifacts_path = result.get("artifacts_path")
    assert artifacts_path is not None and Path(artifacts_path).exists()
