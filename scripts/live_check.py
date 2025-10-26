#!/usr/bin/env python3
"""Diagnostics for live source status (AAA/EIA/RBOB/Kalshi).

Prints mode (live/last_good/sample), as_of date, freshness, and records.
Requires no arguments; honors environment variables.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import requests

from kalshi_gas.config import load_config
from kalshi_gas.etl.aaa import AAAExtractor, AAA_DAILY_AVG_URL
from kalshi_gas.etl.eia import EIAExtractor
from kalshi_gas.etl.rbob import RBOBExtractor
from kalshi_gas.etl.kalshi import KalshiExtractor


def _print_result(name: str, extract_result) -> None:
    prov = getattr(extract_result, "provenance", None)
    if prov is None:
        print(f"{name}: no provenance")
        return
    p = prov.serialize()
    mode = p.get("mode")
    as_of = p.get("as_of")
    fresh = p.get("fresh")
    records = p.get("records")
    details = p.get("details") or {}
    fallback = details.get("fallback")
    chain = p.get("fallback_chain")
    extra = f" (fallback={fallback})" if fallback else ""
    chain_str = f" (chain={chain})" if chain else ""
    path = p.get("path")
    print(
        f"{name}: {mode}, as_of={as_of}, fresh={fresh}, records={records}, path={path}{extra}{chain_str}"
    )


def _http_probe(label: str, url: str, headers: Optional[dict] = None) -> None:
    try:
        resp = requests.get(url, headers=headers or {}, timeout=10)
        print(f"  probe[{label}]: {resp.status_code} {resp.reason} ({url})")
    except Exception as exc:  # noqa: BLE001
        print(f"  probe[{label}]: error={exc} ({url})")


def main() -> None:
    # Force live if requested
    live = os.getenv("KALSHI_GAS_USE_LIVE", "0") == "1"
    print(f"KALSHI_GAS_USE_LIVE={int(live)}")
    cfg = load_config()
    # Show key env hints
    eia_key = os.getenv("EIA_API_KEY")
    kalshi_id = os.getenv("KALSHI_API_KEY_ID")
    kalshi_path = os.getenv("KALSHI_PRIVATE_KEY_PATH")
    kalshi_base = os.getenv("KALSHI_API_BASE", "")
    print(
        "ENVs: EIA_API_KEY={} KALSHI_API_KEY_ID={} KALSHI_PRIVATE_KEY_PATH={} KALSHI_API_BASE={}".format(
            "set" if eia_key else "unset",
            (kalshi_id[:6] + "…") if kalshi_id else "unset",
            kalshi_path or "unset",
            kalshi_base or "unset",
        )
    )

    print("Checking AAA...")
    aaa = AAAExtractor(
        last_good_path=cfg.data.raw_dir / "last_good.aaa.csv",
        sample_path=Path("data/sample/aaa_daily.csv"),
    ).extract()
    _print_result("aaa", aaa)
    _http_probe("aaa_json", AAA_DAILY_AVG_URL)
    _http_probe("aaa_home", "https://gasprices.aaa.com/")

    print("Checking EIA...")
    eia = EIAExtractor(
        last_good_path=cfg.data.raw_dir / "last_good.eia.csv",
        sample_path=Path("data/sample/eia_weekly.csv"),
    ).extract()
    _print_result("eia", eia)
    if eia_key := os.getenv("EIA_API_KEY"):
        eia_inv = (
            f"https://api.eia.gov/series/?api_key={eia_key}&series_id=PET.WGTSTUS1.W"
        )
        eia_rbob = (
            f"https://api.eia.gov/series/?api_key={eia_key}&series_id=PET.RBRTWD.W"
        )
        _http_probe("eia_inventory", eia_inv)
        _http_probe("eia_rbob", eia_rbob)
    else:
        print("  probe[eia]: EIA_API_KEY unset")

    print("Checking RBOB...")
    rbob = RBOBExtractor(
        last_good_path=cfg.data.raw_dir / "last_good.rbob.csv",
        sample_path=Path("data/sample/rbob_futures.csv"),
    ).extract()
    _print_result("rbob", rbob)

    print("Checking Kalshi markets...")
    kalshi = KalshiExtractor(
        last_good_path=cfg.data.raw_dir / "last_good.kalshi.csv",
        sample_path=Path("data/sample/kalshi_markets.csv"),
    ).extract()
    _print_result("kalshi", kalshi)
    base = os.getenv("KALSHI_API_BASE", "https://api.elections.kalshi.com")
    _http_probe("kalshi_base", base)
    print(
        "  creds: v2={} v1={}".format(
            "set"
            if (os.getenv("KALSHI_API_KEY_ID") and os.getenv("KALSHI_PRIVATE_KEY_PATH"))
            else "unset",
            "set"
            if (os.getenv("KALSHI_EMAIL") and os.getenv("KALSHI_PASSWORD"))
            else "unset",
        )
    )

    print("Done.")


if __name__ == "__main__":
    main()
