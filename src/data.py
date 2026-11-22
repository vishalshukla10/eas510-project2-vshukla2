import pandas as pd
import config

def load_raw(path=config.FULL_CSV):
    return pd.read_csv(path)

def clean(df):
    """Drop duplicate URLs and recode so phishing = 1. Returns (df, report)."""
    report = {
        "rows_raw": len(df),
        "duplicate_urls": int(df.duplicated("URL").sum()),
        "missing_cells": int(df.isna().sum().sum()),
    }
    df = df.drop_duplicates("URL").copy()
    df["phishing"] = (df[config.LABEL_COL] == 0).astype(int)
    report["rows_clean"] = len(df)
    report["phishing_rows"] = int(df["phishing"].sum())
    report["legitimate_rows"] = int((df["phishing"] == 0).sum())
    return df, report

def feature_matrix(df):
    drop = [c for c in config.TEXT_COLS + [config.LABEL_COL, "phishing"] if c in df.columns]
    return df.drop(columns=drop).select_dtypes("number")