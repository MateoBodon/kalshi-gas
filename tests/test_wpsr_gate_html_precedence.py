from pathlib import Path

from kalshi_gas.config import load_config
from kalshi_gas.data.assemble import assemble_dataset
from kalshi_gas.etl.pipeline import run_all_etl
from kalshi_gas.risk.gates import wpsr_gate


def test_wpsr_html_precedence_over_yaml(tmp_path: Path) -> None:
    # Ensure processed data exists
    cfg = load_config()
    run_all_etl(cfg)
    dataset = assemble_dataset(cfg)

    # Write conflicting YAML state and HTML snapshot
    data_raw = Path("data_raw")
    data_raw.mkdir(parents=True, exist_ok=True)

    yaml_path = data_raw / "wpsr_state.yml"
    yaml_path.write_text(
        """
refinery_util_pct: 93.0
product_supplied_mbd: 8.1
""".strip()
    )

    # Copy fixture HTML into the expected snapshot location
    fixture = Path(__file__).parent / "fixtures" / "wpsr_summary.html"
    html_path = data_raw / "wpsr_summary.html"
    html_path.write_text(fixture.read_text(encoding="utf-8"), encoding="utf-8")

    alert, details = wpsr_gate(dataset, cfg)
    # Utilization should come from HTML (fixture says 89.6)
    assert abs(float(details.get("refinery_util_pct", 0.0)) - 89.6) < 1e-6
    # Product supplied should come from HTML (fixture says 9.1)
    assert abs(float(details.get("product_supplied_mbd", 0.0)) - 9.1) < 1e-6
