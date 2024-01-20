import pandas as pd


def clean_numeric(col: pd.Series, col_replace: str):
    return pd.to_numeric(col.astype(str).str.replace(col_replace, ""), errors="coerce")
