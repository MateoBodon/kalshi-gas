from kalshi_gas.config import load_config
from kalshi_gas.data.assemble import assemble_dataset
from kalshi_gas.etl.pipeline import run_all_etl
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
