import nasdaqdatalink as ndl
from pathlib import Path

API_KEY = "ZpsSLvVbQpx5pv3ybvdo"
OUT_PATH = Path("data_raw/RB1_settles.csv")

ndl.ApiConfig.api_key = API_KEY


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = ndl.get("CHRIS/CME_RB1", rows=200)
    df.to_csv(OUT_PATH, index=False)
    print(f"Wrote {OUT_PATH}")
    print(df.tail(5))


if __name__ == "__main__":
    main()
