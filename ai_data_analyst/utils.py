"""
utils.py — Helper functions for the AI Data Analyst system.
"""

import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Terminal colour helpers
# ─────────────────────────────────────────────────────────────────────────────

def _colour(code: str, text: str) -> str:
    if hasattr(sys.stdout, "isatty") and sys.stdout.isatty():
        return f"\033[{code}m{text}\033[0m"
    return text

def cyan(t):    return _colour("96", t)
def green(t):   return _colour("92", t)
def yellow(t):  return _colour("93", t)
def red(t):     return _colour("91", t)
def bold(t):    return _colour("1",  t)
def dim(t):     return _colour("2",  t)
def magenta(t): return _colour("95", t)
def blue(t):    return _colour("94", t)


# ─────────────────────────────────────────────────────────────────────────────
# Banner & UI helpers
# ─────────────────────────────────────────────────────────────────────────────

BANNER = r"""
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║     █████╗ ██╗    ██████╗  █████╗ ████████╗ █████╗              ║
║    ██╔══██╗██║    ██╔══██╗██╔══██╗╚══██╔══╝██╔══██╗             ║
║    ███████║██║    ██║  ██║███████║   ██║   ███████║             ║
║    ██╔══██║██║    ██║  ██║██╔══██║   ██║   ██╔══██║             ║
║    ██║  ██║██║    ██████╔╝██║  ██║   ██║   ██║  ██║             ║
║    ╚═╝  ╚═╝╚═╝    ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝             ║
║                                                                  ║
║            A N A L Y S T  —  AI Data Analysis System            ║
╚══════════════════════════════════════════════════════════════════╝
"""

DIVIDER = cyan("─" * 66)

def print_banner():
    print(cyan(BANNER))

def print_divider():
    print(DIVIDER)

def print_section(title: str):
    print(f"\n{bold(cyan('━' * 66))}")
    print(f"  {bold(yellow(title))}")
    print(f"{bold(cyan('━' * 66))}")

def print_success(msg: str):
    print(f"  {green('✓')} {msg}")

def print_error(msg: str):
    print(f"  {red('✗')} {msg}")

def print_warning(msg: str):
    print(f"  {yellow('!')} {msg}")

def print_info(msg: str):
    print(f"  {cyan('›')} {msg}")

def prompt(msg: str, default: str = "") -> str:
    suffix = f" {dim(f'[{default}]')}" if default else ""
    val = input(f"\n  {bold(msg)}{suffix}: ").strip()
    return val if val else default

def confirm(msg: str) -> bool:
    ans = input(f"\n  {bold(msg)} {dim('[y/N]')}: ").strip().lower()
    return ans in ("y", "yes")

def press_enter():
    input(dim("\n  Press Enter to continue …"))


# ─────────────────────────────────────────────────────────────────────────────
# Column type detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_column_types(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Detect numeric, categorical, and datetime columns.
    Returns dict with keys: 'numeric', 'categorical', 'datetime'.
    """
    numeric_cols    = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    datetime_cols   = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()

    # Try parsing object columns as datetime
    for col in list(categorical_cols):
        try:
            converted = pd.to_datetime(df[col], infer_datetime_format=True)
            if converted.notna().sum() / max(len(df), 1) >= 0.8:
                datetime_cols.append(col)
                categorical_cols.remove(col)
        except Exception:
            pass

    return {
        "numeric":     numeric_cols,
        "categorical": categorical_cols,
        "datetime":    datetime_cols,
    }


def detect_problem_type(df: pd.DataFrame,
                        target_col: str) -> str:
    """
    Auto-detect whether a target column suggests regression or classification.
    Returns 'regression', 'classification', or 'clustering'.
    """
    if target_col not in df.columns:
        return "clustering"

    col = df[target_col].dropna()

    # If non-numeric → classification
    if col.dtype == object or col.dtype.name == "category":
        return "classification"

    # If small number of unique integers → classification
    n_unique = col.nunique()
    if n_unique <= 15 and col.apply(float.is_integer if hasattr(col.iloc[0], 'is_integer') else lambda x: x == int(x)).all():
        return "classification"

    return "regression"


# ─────────────────────────────────────────────────────────────────────────────
# Data summary helpers
# ─────────────────────────────────────────────────────────────────────────────

def print_data_summary(df: pd.DataFrame) -> None:
    col_types = detect_column_types(df)

    print_section("DATASET SUMMARY")
    rows_str = f"{df.shape[0]:,}"
    print(f"  {'Rows':<22}: {bold(rows_str)}")
    print(f"  {'Columns':<22}: {bold(str(df.shape[1]))}")
    print(f"  {'Numeric columns':<22}: {green(str(len(col_types['numeric'])))}")
    print(f"  {'Categorical columns':<22}: {yellow(str(len(col_types['categorical'])))}")
    print(f"  {'Datetime columns':<22}: {cyan(str(len(col_types['datetime'])))}")
    print(f"  {'Memory usage':<22}: {dim(f'{df.memory_usage(deep=True).sum() / 1024:.1f} KB')}")

    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        print(f"\n  {yellow('Missing values detected:')}")
        for col, cnt in missing.items():
            pct = cnt / len(df) * 100
            bar = "█" * int(pct / 5)
            print(f"    {col:<28} {cnt:>6,}  ({pct:.1f}%)  {yellow(bar)}")
    else:
        print_success("No missing values detected.")

    print(f"\n  {bold('Numeric columns')} : {', '.join(col_types['numeric'][:8])}"
          f"{'...' if len(col_types['numeric']) > 8 else ''}")
    print(f"  {bold('Categorical')}     : {', '.join(col_types['categorical'][:8])}"
          f"{'...' if len(col_types['categorical']) > 8 else ''}")
    if col_types["datetime"]:
        print(f"  {bold('Datetime')}        : {', '.join(col_types['datetime'])}")


def print_dataframe_preview(df: pd.DataFrame, n: int = 5) -> None:
    print_section("DATA PREVIEW")
    try:
        pd.set_option("display.max_columns", 12)
        pd.set_option("display.width", 120)
        print(df.head(n).to_string())
    finally:
        pd.reset_option("display.max_columns")
        pd.reset_option("display.width")


def print_describe(df: pd.DataFrame) -> None:
    print_section("DESCRIPTIVE STATISTICS")
    num_df = df.select_dtypes(include=np.number)
    if num_df.empty:
        print_warning("No numeric columns to describe.")
        return
    desc = num_df.describe().T.round(4)
    desc["skew"]  = num_df.skew().round(4)
    desc["kurt"]  = num_df.kurtosis().round(4)
    print(desc.to_string())


# ─────────────────────────────────────────────────────────────────────────────
# Output directory
# ─────────────────────────────────────────────────────────────────────────────

def ensure_output_dir(subdir: str = "outputs") -> Path:
    path = Path(subdir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def timestamped_filename(prefix: str, ext: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = re.sub(r"[^\w\-]", "_", prefix)
    return f"{safe}_{ts}.{ext}"


# ─────────────────────────────────────────────────────────────────────────────
# Column picker helper
# ─────────────────────────────────────────────────────────────────────────────

def pick_column(df: pd.DataFrame,
                prompt_msg: str,
                allowed_types: Optional[List[str]] = None) -> Optional[str]:
    """
    Interactive column picker.  allowed_types: list of dtype kinds e.g. ['numeric'].
    """
    col_types = detect_column_types(df)

    if allowed_types:
        cols = []
        for t in allowed_types:
            cols.extend(col_types.get(t, []))
    else:
        cols = df.columns.tolist()

    if not cols:
        print_warning("No suitable columns available.")
        return None

    print(f"\n  Available columns:")
    for i, c in enumerate(cols, 1):
        dtype = str(df[c].dtype)
        print(f"    {dim(str(i)):>6}. {c:<30} {dim(dtype)}")

    raw = prompt(prompt_msg)
    if raw.isdigit():
        idx = int(raw) - 1
        if 0 <= idx < len(cols):
            return cols[idx]
    if raw in cols:
        return raw

    print_error(f"Column '{raw}' not found.")
    return None


def pick_multiple_columns(df: pd.DataFrame,
                          prompt_msg: str,
                          allowed_types: Optional[List[str]] = None,
                          min_count: int = 1) -> List[str]:
    """
    Pick multiple columns (comma-separated numbers or names).
    """
    col_types = detect_column_types(df)
    if allowed_types:
        cols = []
        for t in allowed_types:
            cols.extend(col_types.get(t, []))
    else:
        cols = df.columns.tolist()

    if not cols:
        print_warning("No suitable columns available.")
        return []

    print(f"\n  Available columns:")
    for i, c in enumerate(cols, 1):
        print(f"    {dim(str(i)):>6}. {c}")

    raw = prompt(f"{prompt_msg} (comma-separated numbers or names)")
    parts = [p.strip() for p in raw.split(",")]
    selected = []
    for p in parts:
        if p.isdigit():
            idx = int(p) - 1
            if 0 <= idx < len(cols):
                selected.append(cols[idx])
        elif p in cols:
            selected.append(p)

    if len(selected) < min_count:
        print_warning(f"Need at least {min_count} column(s). Using first available.")
        selected = cols[:min_count]

    return selected
