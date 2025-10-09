"""Download and store the WPSR summary HTML."""

from __future__ import annotations

from pathlib import Path

import requests

DEFAULT_URL = "https://www.eia.gov/petroleum/supply/weekly/"
DEFAULT_OUTPUT = Path("data_raw/wpsr_summary.html")


def fetch_html(url: str = DEFAULT_URL) -> str:
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.text


def save_snapshot(
    output_path: Path = DEFAULT_OUTPUT,
    url: str = DEFAULT_URL,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    html = fetch_html(url=url)
    output_path.write_text(html, encoding="utf-8")
    return output_path


def main() -> None:
    path = save_snapshot()
    print(f"Saved WPSR summary HTML to {path}")


if __name__ == "__main__":
    main()
