import pandas as pd


def clean_numeric(col: str, col_replace: str):
    return pd.to_numeric(col.str.replace(col_replace, ""), errors="coerce")
