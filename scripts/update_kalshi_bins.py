#!/usr/bin/env python3
"""Update data_raw/kalshi_bins.yml from Kalshi live markets.

This script supports two auth modes:
  1) API key signing (preferred; Trade API v2):
     - KALSHI_API_KEY_ID (key id), KALSHI_PRIVATE_KEY_PATH (PEM)
     - Optional: KALSHI_API_BASE (default https://api.kalshi.com)
  2) Legacy session login (Trade API v1):
     - KALSHI_EMAIL, KALSHI_PASSWORD

It fetches energy markets and tries to infer probabilities at
the configured thresholds in data_raw/kalshi_bins.yml. We convert
Yes-probabilities for events of the form "≥ $X" into CDF points
P(Price ≤ X) = 1 - P(≥ X) and write back to the YAML.

If an exact threshold is not found, we leave the existing value.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import argparse
import base64
import json
import time

import requests
import yaml
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding


API_BASE_V1 = "https://trading-api.kalshi.com/v1"
API_BASE_V2_DEFAULT = "https://api.elections.kalshi.com"


def login_v1(email: str, password: str) -> str:
    session = requests.Session()
    resp = session.post(
        f"{API_BASE_V1}/auth/login",
        json={"email": email, "password": password},
        timeout=30,
    )
    resp.raise_for_status()
    token = resp.json()["token"]
    return token


def fetch_energy_markets_v1(token: str) -> List[Dict[str, object]]:
    headers = {"Authorization": f"Bearer {token}"}
    resp = requests.get(
        f"{API_BASE_V1}/markets?category=energy", headers=headers, timeout=30
    )
    resp.raise_for_status()
    payload = resp.json()
    return list(payload.get("markets", []))


def _load_private_key(pem_path: str):
    pem_env = os.environ.get("KALSHI_PY_PRIVATE_KEY_PEM")
    passphrase = os.environ.get("KALSHI_PRIVATE_KEY_PASSPHRASE")
    password = passphrase.encode("utf-8") if passphrase else None
    if pem_env:
        return serialization.load_pem_private_key(
            pem_env.encode("utf-8"), password=password
        )
    with open(pem_path, "rb") as f:
        return serialization.load_pem_private_key(f.read(), password=password)


def _signed_headers_v2(
    key_id: str,
    private_key,
    method: str,
    path: str,
    include_query: bool = True,
    ts_ms: bool = True,
    sign_version: str | None = None,
) -> Dict[str, str]:
    # Timestamp: milliseconds (default) or seconds (fallback)
    ts_val = int(time.time() * 1000) if ts_ms else int(time.time())
    ts = str(ts_val)
    # Some gateways expect path without query in the signature; allow toggling
    if not include_query and "?" in path:
        path_to_sign = path.split("?", 1)[0]
    else:
        path_to_sign = path
    msg = ts + method.upper() + path_to_sign
    signature = private_key.sign(
        msg.encode("utf-8"),
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH
        ),
        hashes.SHA256(),
    )
    sig_b64 = base64.b64encode(signature).decode("utf-8")
    headers: Dict[str, str] = {
        "KALSHI-ACCESS-KEY": key_id,
        "KALSHI-ACCESS-TIMESTAMP": ts,
        "KALSHI-ACCESS-SIGNATURE": sig_b64,
    }
    if sign_version:
        headers["KALSHI-ACCESS-SIGN-VERSION"] = sign_version
    return headers


def fetch_energy_markets_v2(
    key_id: str,
    pem_path: str,
    api_base: str,
    series_ticker: str | None = None,
    event_ticker: str | None = None,
) -> List[Dict[str, object]]:
    private_key = _load_private_key(pem_path)
    # Build query for AAA series; optionally narrow to a specific event
    base_query = "/trade-api/v2/markets?limit=200"
    if series_ticker:
        base_query += f"&series_ticker={series_ticker}"
    if event_ticker:
        base_query += f"&event_ticker={event_ticker}"

    # Try multiple variants; some gateways reject certain params
    path_variants = [base_query, "/trade-api/v2/markets?limit=200"]
    include_query = os.getenv("KALSHI_SIGN_INCLUDE_QUERY", "1") == "1"
    ts_ms = os.getenv("KALSHI_SIGN_TS_MS", "1") == "1"
    sign_version = os.getenv("KALSHI_SIGN_VERSION")

    last_exc: Exception | None = None
    for path in path_variants:
        url = f"{api_base}{path}"
        try:
            # Strategy A
            headers = _signed_headers_v2(
                key_id, private_key, "GET", path, include_query, ts_ms, sign_version
            )
            resp = requests.get(url, headers=headers, timeout=30)
            if resp.status_code in (401, 400):
                # Strategy B: toggle include_query
                headers = _signed_headers_v2(
                    key_id,
                    private_key,
                    "GET",
                    path,
                    not include_query,
                    ts_ms,
                    sign_version,
                )
                resp = requests.get(url, headers=headers, timeout=30)
            if resp.status_code in (401, 400):
                # Strategy C: toggle timestamp units
                headers = _signed_headers_v2(
                    key_id,
                    private_key,
                    "GET",
                    path,
                    include_query,
                    not ts_ms,
                    sign_version,
                )
                resp = requests.get(url, headers=headers, timeout=30)
            if resp.status_code in (401, 400):
                # Strategy D: toggle both
                headers = _signed_headers_v2(
                    key_id,
                    private_key,
                    "GET",
                    path,
                    not include_query,
                    not ts_ms,
                    sign_version,
                )
                resp = requests.get(url, headers=headers, timeout=30)
            resp.raise_for_status()
            payload = resp.json()
            return list(payload.get("markets", []))
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            continue
    # If all variants failed, raise the last error
    if last_exc:
        raise last_exc
    return []


def load_bins(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def write_bins(path: Path, thresholds: List[float], cdf_values: List[float]) -> None:
    payload = {"thresholds": thresholds, "probabilities": cdf_values}
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def write_meta(
    path: Path,
    source: str,
    thresholds: List[float],
    cdf_values: List[float],
    as_of: str | None,
) -> None:
    meta = {
        "source": source,
        "as_of": as_of or datetime.now(timezone.utc).date().isoformat(),
        "fetched_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "thresholds": thresholds,
        "probabilities": cdf_values,
    }
    meta_path = path.with_suffix(path.suffix + ".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _market_prob_mid(market: Dict[str, object]) -> float | None:
    # Prefer midpoint of yes_bid/yes_ask (in cents), else last_price
    def to_p01(x: object) -> float | None:
        try:
            v = float(x)
        except Exception:
            return None
        return max(0.0, min(1.0, v / 100.0))

    bid = to_p01(market.get("yes_bid"))
    ask = to_p01(market.get("yes_ask"))
    if bid is not None and ask is not None and ask >= bid and ask > 0:
        return float((bid + ask) / 2.0)
    last = to_p01(market.get("last_price"))
    if last is not None:
        return last
    prob_yes = market.get("probability_yes")
    if prob_yes is not None:
        try:
            return max(0.0, min(1.0, float(prob_yes) / 100.0))
        except Exception:
            return None
    return None


def find_prob_for_threshold(
    markets: List[Dict[str, object]], thr: float
) -> float | None:
    # Match by numeric strike (floor_strike equals threshold), else by ticker suffix
    candidates: List[float] = []
    for m in markets:
        fs = m.get("floor_strike")
        strike_match = False
        try:
            if fs is not None and abs(float(fs) - float(thr)) < 1e-6:
                strike_match = True
        except Exception:
            strike_match = False
        if not strike_match:
            tk = str(m.get("ticker", ""))
            # Expect suffix like ...-3.10
            if tk and tk.split("-")[-1].strip() == f"{thr:.2f}":
                strike_match = True
        if not strike_match:
            continue
        p = _market_prob_mid(m)
        if p is not None:
            candidates.append(p)
    if not candidates:
        return None
    candidates.sort()
    return float(candidates[len(candidates) // 2])


def parse_manual_survival(raw: str) -> Dict[float, float]:
    result: Dict[float, float] = {}
    for part in raw.split(","):
        if not part.strip():
            continue
        if "=" not in part:
            raise ValueError(f"Invalid manual entry '{part}'; expected threshold=prob")
        key, value = [item.strip() for item in part.split("=", 1)]
        thr = float(key)
        prob = float(value)
        if prob > 1:
            prob = prob / 100.0
        if prob < 0 or prob > 1:
            raise ValueError(f"Probability for {thr} must be within [0,1]")
        result[thr] = prob
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Update Kalshi bin priors")
    parser.add_argument(
        "--manual-survival",
        help="Comma separated threshold=prob_yes list (probabilities as 0-1 or percentage). Example: 3.05=0.62,3.10=0.48",
    )
    parser.add_argument(
        "--as-of",
        help="Override as_of date for metadata (YYYY-MM-DD). Defaults to today.",
    )
    args = parser.parse_args()

    bins_path = Path("data_raw/kalshi_bins.yml")
    if not bins_path.exists():
        print(f"Missing {bins_path}")
        return 1

    markets: List[Dict[str, object]] = []
    source_label = "kalshi_api"
    manual_probs: Dict[float, float] | None = None
    if args.manual_survival:
        try:
            manual_probs = parse_manual_survival(args.manual_survival)
            source_label = "manual_input"
        except ValueError as exc:
            print(exc)
            return 1
    else:
        key_id = os.environ.get("KALSHI_API_KEY_ID")
        pem_path = os.environ.get("KALSHI_PRIVATE_KEY_PATH")
        api_base = os.environ.get("KALSHI_API_BASE", API_BASE_V2_DEFAULT)
        series_ticker = os.environ.get("KALSHI_SERIES_TICKER", "KXAAAGASM")
        event_ticker = os.environ.get("KALSHI_EVENT_TICKER")

        if key_id and pem_path:
            try:
                markets = fetch_energy_markets_v2(
                    key_id,
                    pem_path,
                    api_base,
                    series_ticker=series_ticker,
                    event_ticker=event_ticker,
                )
                print("Fetched markets via API key (v2)")
            except Exception as exc:  # noqa: BLE001
                print(f"Kalshi API v2 error: {exc}")
                return 1
        else:
            email = os.environ.get("KALSHI_EMAIL")
            password = os.environ.get("KALSHI_PASSWORD")
            if not email or not password:
                print(
                    "Set KALSHI_API_KEY_ID/KALSHI_PRIVATE_KEY_PATH (preferred) or KALSHI_EMAIL/KALSHI_PASSWORD, "
                    "or use --manual-survival for offline updates."
                )
                return 1
            try:
                token = login_v1(email, password)
                markets = fetch_energy_markets_v1(token)
                print("Fetched markets via session (v1)")
            except Exception as exc:  # noqa: BLE001
                print(f"Kalshi API v1 error: {exc}")
                return 1

    bins = load_bins(bins_path)
    thresholds = [float(x) for x in bins.get("thresholds", [])]
    cdf_values = [float(x) for x in bins.get("probabilities", [])]
    if len(cdf_values) != len(thresholds):
        cdf_values = [0.5 for _ in thresholds]

    updated = False
    for idx, thr in enumerate(thresholds):
        if manual_probs is not None:
            p_ge = manual_probs.get(thr)
        else:
            p_ge = find_prob_for_threshold(markets, thr)
        if p_ge is None:
            continue
        p_le = 1.0 - max(0.0, min(1.0, p_ge))
        if abs(p_le - cdf_values[idx]) > 1e-6:
            cdf_values[idx] = p_le
            updated = True

    if manual_probs is None and not markets:
        print("No market data available; bins not updated")
        return 1

    if updated or manual_probs is not None:
        write_bins(bins_path, thresholds, cdf_values)
        print("kalshi_bins.yml updated")
    else:
        print("kalshi_bins.yml unchanged (no matching thresholds found)")

    write_meta(
        bins_path,
        source_label,
        thresholds,
        cdf_values,
        args.as_of,
    )
    print("kalshi_bins metadata recorded")
    return 0


if __name__ == "__main__":
    sys.exit(main())
