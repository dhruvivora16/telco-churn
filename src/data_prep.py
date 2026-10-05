"""Step 1: raw CSV -> SQLite -> SQL cleaning -> pandas checks -> processed CSV.
Run: python -m src.data_prep
"""
import sqlite3
import pandas as pd

from src.config import RAW_CSV, DB_PATH, CLEAN_SQL, PROCESSED_CSV, TARGET, ID_COL
from src.features import add_features


def main() -> pd.DataFrame:
    if not RAW_CSV.exists():
        raise FileNotFoundError(
            f"{RAW_CSV} not found. Download the IBM Telco dataset from Kaggle "
            "(or run `python scripts/make_sample_data.py` for a synthetic stand-in)."
        )
    raw = pd.read_csv(RAW_CSV)
    print(f"Raw rows: {len(raw):,}")

    with sqlite3.connect(DB_PATH) as con:
        raw.to_sql("raw_customers", con, if_exists="replace", index=False)
        con.executescript(CLEAN_SQL.read_text())
        df = pd.read_sql("SELECT * FROM clean_customers", con)

    # --- pandas validation + feature engineering ---
    before = len(df)
    df = df.drop_duplicates(subset=ID_COL).reset_index(drop=True)
    print(f"Dropped {before - len(df)} duplicate customers")
    assert df.isna().sum().sum() == 0, "Unexpected nulls after cleaning"
    assert set(df[TARGET].unique()) <= {0, 1}
    df = add_features(df)

    df.to_csv(PROCESSED_CSV, index=False)
    print(f"Saved {len(df):,} rows -> {PROCESSED_CSV}  (churn rate {df[TARGET].mean():.1%})")
    return df


if __name__ == "__main__":
    main()
