"""
main.py — AI Data Analyst System entry point.
Pure Python CLI. Run with: python main.py
"""

import sys
import os
from pathlib import Path
from typing import Optional

import pandas as pd

from utils import (
    print_banner, print_section, print_divider,
    print_success, print_error, print_warning, print_info,
    bold, green, red, yellow, cyan, dim, magenta,
    prompt, confirm, press_enter,
    print_data_summary, print_dataframe_preview, print_describe,
    detect_column_types
)
from visualization import GraphBuilder, print_graph_menu, GRAPH_MENU
from models import ModelTrainer


# ─────────────────────────────────────────────────────────────────────────────
# State
# ─────────────────────────────────────────────────────────────────────────────

class AppState:
    def __init__(self):
        self.df: Optional[pd.DataFrame]    = None
        self.filepath: str                 = ""
        self.graph_builder: Optional[GraphBuilder] = None


# ─────────────────────────────────────────────────────────────────────────────
# Main menu
# ─────────────────────────────────────────────────────────────────────────────

def print_main_menu(state: AppState) -> str:
    has_data = state.df is not None

    def opt(n, label, needs_data=False):
        if needs_data and not has_data:
            return dim(f"  [{n}] {label}  ← load data first")
        return green(f"  [{n}] {label}")

    print(f"\n{bold(cyan('═' * 66))}")
    print(f"  {bold('MAIN MENU')}"
          + (f"   {dim('│')}  {cyan(Path(state.filepath).name)}"
             f"  {dim(f'({state.df.shape[0]:,} rows × {state.df.shape[1]} cols)')}"
             if has_data else ""))
    print(f"{bold(cyan('═' * 66))}")
    print(opt("1", "Load / Change Dataset"))
    print(opt("2", "Show Data Preview & Summary",  needs_data=True))
    print(opt("3", "Generate Graph",                needs_data=True))
    print(opt("4", "Train ML Models",               needs_data=True))
    print(opt("5", "Save Last Graph",               needs_data=True))
    print(opt("6", "Descriptive Statistics",        needs_data=True))
    print(opt("7", "Export Clean Data (CSV)",       needs_data=True))
    print(red("  [0] Exit"))
    print(f"{bold(cyan('═' * 66))}")

    return input(f"\n  {bold('Your choice')}: ").strip()


# ─────────────────────────────────────────────────────────────────────────────
# Action handlers
# ─────────────────────────────────────────────────────────────────────────────

def handle_load(state: AppState) -> None:
    print_section("LOAD DATASET")
    print_info("Supported formats: CSV, Excel (.xlsx/.xls), JSON")

    filepath = prompt("Enter full file path to your dataset")
    if not filepath:
        print_error("No path entered.")
        return

    path = Path(filepath.strip().strip('"').strip("'"))
    if not path.exists():
        print_error(f"File not found: {path}")
        print_info("Tip: drag the file into this terminal window to get its path.")
        return

    try:
        ext = path.suffix.lower()
        print_info(f"Loading {path.name} …")

        if ext == ".csv":
            df = pd.read_csv(path)
        elif ext in (".xlsx", ".xls"):
            df = pd.read_excel(path)
        elif ext == ".json":
            df = pd.read_json(path)
        else:
            print_error(f"Unsupported file type: {ext}")
            return

        state.df           = df
        state.filepath     = str(path)
        state.graph_builder = GraphBuilder(df)

        print_success(f"Loaded {df.shape[0]:,} rows × {df.shape[1]} columns.")
        print_dataframe_preview(df, n=5)
        print_data_summary(df)

    except Exception as exc:
        print_error(f"Failed to load file: {exc}")


def handle_show_data(state: AppState) -> None:
    if state.df is None:
        print_error("No dataset loaded.")
        return

    print_section("DATA OVERVIEW")
    print("\n  What would you like to see?")
    print("    1. First N rows")
    print("    2. Last N rows")
    print("    3. Full summary (shape, types, missing)")
    print("    4. Value counts for a column")
    print("    5. Correlation matrix (text)")

    choice = prompt("Choose", "3")

    if choice == "1":
        n  = int(prompt("How many rows?", "10"))
        print_dataframe_preview(state.df, n=n)

    elif choice == "2":
        n = int(prompt("How many rows?", "10"))
        print_section("LAST ROWS")
        pd.set_option("display.max_columns", 12)
        print(state.df.tail(n).to_string())
        pd.reset_option("display.max_columns")

    elif choice == "3":
        print_data_summary(state.df)

    elif choice == "4":
        col_types = detect_column_types(state.df)
        print("\n  Columns:")
        for i, c in enumerate(state.df.columns, 1):
            print(f"    {i:>3}. {c}")
        col = prompt("Enter column name or number")
        if col.isdigit():
            idx = int(col) - 1
            if 0 <= idx < len(state.df.columns):
                col = state.df.columns[idx]
        if col in state.df.columns:
            vc = state.df[col].value_counts().head(20)
            print_section(f"VALUE COUNTS — {col}")
            total = len(state.df[col].dropna())
            for val, cnt in vc.items():
                pct = cnt / total * 100
                bar = "█" * int(pct / 2)
                print(f"  {str(val):<30} {cnt:>6,}  {pct:>5.1f}%  {cyan(bar)}")
        else:
            print_error("Column not found.")

    elif choice == "5":
        num_df = state.df.select_dtypes(include="number")
        if num_df.shape[1] < 2:
            print_warning("Need at least 2 numeric columns.")
        else:
            corr = num_df.corr().round(3)
            print_section("CORRELATION MATRIX")
            print(corr.to_string())


def handle_graph(state: AppState) -> None:
    if state.df is None:
        print_error("No dataset loaded.")
        return

    print_graph_menu()
    choice = prompt("Choose graph type", "4")

    if choice not in GRAPH_MENU:
        print_warning("Invalid choice.")
        return

    print_info(f"Building: {GRAPH_MENU[choice]} …")
    try:
        state.graph_builder.build(choice)
        print_success("Graph opened in browser (or saved if non-interactive).")
    except Exception as exc:
        print_error(f"Graph generation failed: {exc}")


def handle_train_models(state: AppState) -> None:
    if state.df is None:
        print_error("No dataset loaded.")
        return

    try:
        trainer = ModelTrainer(state.df.copy())
        trainer.run()
    except Exception as exc:
        print_error(f"Training failed: {exc}")


def handle_save_graph(state: AppState) -> None:
    if state.graph_builder is None:
        print_error("No graph has been generated yet.")
        return
    try:
        state.graph_builder.save_last_graph()
    except Exception as exc:
        print_error(f"Save failed: {exc}")


def handle_stats(state: AppState) -> None:
    if state.df is None:
        print_error("No dataset loaded.")
        return
    print_describe(state.df)


def handle_export(state: AppState) -> None:
    if state.df is None:
        print_error("No dataset loaded.")
        return

    out_dir   = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    out_path  = out_dir / f"clean_{Path(state.filepath).stem}.csv"

    state.df.to_csv(out_path, index=False)
    print_success(f"Clean data exported → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print_banner()
    state = AppState()

    while True:
        try:
            choice = print_main_menu(state)

            if choice == "1":
                handle_load(state)

            elif choice == "2":
                handle_show_data(state)

            elif choice == "3":
                handle_graph(state)

            elif choice == "4":
                handle_train_models(state)

            elif choice == "5":
                handle_save_graph(state)

            elif choice == "6":
                handle_stats(state)

            elif choice == "7":
                handle_export(state)

            elif choice == "0":
                if confirm("Are you sure you want to exit?"):
                    print(cyan("\n  Goodbye! Thanks for using AI Data Analyst.\n"))
                    sys.exit(0)

            else:
                print_warning("Invalid choice. Enter a number from the menu.")

        except KeyboardInterrupt:
            print(yellow("\n\n  [Ctrl+C] Press 0 to exit properly.\n"))
            continue

        except Exception as exc:
            print_error(f"Unexpected error: {exc}")

        press_enter()


if __name__ == "__main__":
    main()
