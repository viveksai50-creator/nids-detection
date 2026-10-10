"""Clean the raw improved CIC-IDS2017 files into one parquet + summary JSON.

Run:
    python -m src.data.clean

Steps (DATA-SOURCES.md §5): load all CSV/parquet under RAW_DIR, strip column names,
drop identifier columns, coerce features to numeric, drop Inf/NaN rows, drop exact
duplicates, drop constant columns, map labels to families with the configured
policies, abort on unmapped labels, write parquet + cleaning_summary.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src import config as C

META_COLUMNS = ["label_raw", "family", "is_attack"]


def find_raw_files() -> list[Path]:
    files = sorted(p for p in C.RAW_DIR.rglob("*") if p.suffix.lower() in {".csv", ".parquet"})
    if not files:
        raise FileNotFoundError(f"no .csv/.parquet under {C.RAW_DIR}; run src.data.download first")
    return files


def load_raw(files: list[Path]) -> pd.DataFrame:
    frames = []
    for f in files:
        df = pd.read_parquet(f) if f.suffix.lower() == ".parquet" else pd.read_csv(f, low_memory=False)
        df.columns = [str(c).strip() for c in df.columns]
        df["__source_file"] = f.name
        frames.append(df)
        print(f"loaded {f.name}: {len(df):,} rows, {df.shape[1]} cols")
    return pd.concat(frames, ignore_index=True)


def find_label_column(df: pd.DataFrame) -> str:
    for c in df.columns:
        if c.strip().lower() in C.LABEL_COLUMN_CANDIDATES:
            return c
    raise KeyError(f"no label column among {list(df.columns)}")


def build_family_mapping(labels: list[str]) -> tuple[dict[str, str], list[str]]:
    """Return {raw_label: family|BENIGN|__DROP__} and the list of unmapped labels."""
    mapping: dict[str, str] = {}
    unmapped: list[str] = []
    for lab in labels:
        fam = C.label_to_family(lab)
        if fam == "ATTEMPTED":
            if C.ATTEMPTED_POLICY == "drop":
                fam = "__DROP__"
            elif C.ATTEMPTED_POLICY == "attack":
                fam = C.label_to_family(C.strip_attempted(lab))
            elif C.ATTEMPTED_POLICY == "benign":
                fam = "BENIGN"
            else:
                raise ValueError(f"bad ATTEMPTED_POLICY {C.ATTEMPTED_POLICY}")
        if fam == "HEARTBLEED":
            fam = "__DROP__" if C.HEARTBLEED_POLICY == "drop" else "DoS"
        if fam == "UNMAPPED":
            unmapped.append(lab)
        mapping[lab] = fam
    return mapping, unmapped


def main() -> int:
    C.ensure_dirs()
    summary: dict = {"stages": []}

    df = load_raw(find_raw_files())
    summary["stages"].append({"stage": "loaded", "rows": int(len(df)), "cols": int(df.shape[1])})

    label_col = find_label_column(df)
    df["label_raw"] = df[label_col].astype(str).str.strip()
    summary["raw_label_counts"] = df["label_raw"].value_counts().to_dict()

    # Drop identifier / non-feature columns.
    drop_cols = [c for c in df.columns if c.strip().lower() in C.NON_FEATURE_COLUMNS] + ["__source_file"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    summary["dropped_non_feature_columns"] = sorted(drop_cols)

    feature_cols = [c for c in df.columns if c not in META_COLUMNS]
    # Coerce features to numeric; anything unparseable becomes NaN and is dropped below.
    for c in feature_cols:
        if not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)

    before = len(df)
    df = df.dropna(subset=feature_cols)
    summary["stages"].append({"stage": "drop_inf_nan", "rows": int(len(df)), "removed": int(before - len(df))})

    before = len(df)
    df = df.drop_duplicates(subset=feature_cols + ["label_raw"])
    summary["stages"].append({"stage": "drop_duplicates", "rows": int(len(df)), "removed": int(before - len(df))})

    constant = [c for c in feature_cols if df[c].nunique(dropna=False) <= 1]
    df = df.drop(columns=constant)
    feature_cols = [c for c in feature_cols if c not in constant]
    summary["dropped_constant_columns"] = constant

    mapping, unmapped = build_family_mapping(sorted(df["label_raw"].unique()))
    summary["label_to_family"] = mapping
    if unmapped:
        print("UNMAPPED LABELS - edit label_to_family() in src/config.py:")
        for u in unmapped:
            print("  ", repr(u))
        raise SystemExit(2)

    df["family"] = df["label_raw"].map(mapping)
    before = len(df)
    df = df[df["family"] != "__DROP__"].copy()
    summary["stages"].append({
        "stage": "apply_policies",
        "rows": int(len(df)),
        "removed": int(before - len(df)),
        "attempted_policy": C.ATTEMPTED_POLICY,
        "heartbleed_policy": C.HEARTBLEED_POLICY,
    })

    df["is_attack"] = (df["family"] != "BENIGN").astype(np.int8)
    df[feature_cols] = df[feature_cols].astype(np.float32)

    fam_counts = df["family"].value_counts().to_dict()
    summary["family_counts"] = fam_counts
    summary["feature_columns"] = feature_cols
    summary["n_features"] = len(feature_cols)
    summary["benign_share"] = float((df["family"] == "BENIGN").mean())
    missing = [f for f in C.FAMILIES if f not in fam_counts]
    small = {f: n for f, n in fam_counts.items() if f != "BENIGN" and n < 1000}
    summary["families_missing"] = missing
    summary["families_under_1000"] = small
    if missing:
        print("WARNING families with zero rows:", missing)
    if small:
        print("WARNING families under 1,000 rows (wide CIs expected):", small)

    out_cols = feature_cols + META_COLUMNS
    df[out_cols].to_parquet(C.PROCESSED_FILE, index=False)
    with open(C.PROCESSED_SUMMARY, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"wrote {C.PROCESSED_FILE} ({len(df):,} rows, {len(feature_cols)} features)")
    print(json.dumps(fam_counts, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())