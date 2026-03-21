"""
Exploratory Data Analysis helper function


"""

from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import polars as pl
import psutil
import seaborn as sns

# --- Configuration ---
# Robustly determine the project root directory
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
PLOTS_DIR = SCRIPT_DIR / "plots"
POLYMARKET_DIR = DATA_DIR / "Polymarket"

# Create plots directory if it doesn't exist
PLOTS_DIR.mkdir(exist_ok=True)


# --- Memory Tracking Utilities ---


def get_memory_usage_mb() -> float:
    """
    Get current memory usage of the process in MB.

    Returns:
        Memory usage in megabytes
    """
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


def format_memory(mb: float) -> str:
    """
    Format memory value in MB to human-readable string.

    Args:
        mb: Memory value in megabytes

    Returns:
        Formatted string (e.g., "123.45 MB" or "1.23 GB")
    """
    if mb < 1024:
        return f"{mb:.2f} MB"
    else:
        return f"{mb / 1024:.2f} GB"


@contextmanager
def track_memory(operation_name: str):
    """
    Context manager to track memory usage before and after an operation.

    Args:
        operation_name: Name of the operation being tracked

    Yields:
        None
    """
    memory_before = get_memory_usage_mb()
    print(f"[Memory] Before {operation_name}: {format_memory(memory_before)}")

    try:
        yield
    finally:
        memory_after = get_memory_usage_mb()
        memory_delta = memory_after - memory_before
        print(
            f"[Memory] After {operation_name}: {format_memory(memory_after)} "
            f"(Δ {format_memory(memory_delta)})"
        )


# --- Data Loading Functions ---


def load_polymarket_data(datadir: Path) -> Optional[dict[str, pl.DataFrame]]:
    """
    Load Polymarket data from parquet files using Polars lazy scan.

    Args:
        datadir: Directory containing Polymarket parquet files

    Returns:
        Dictionary mapping data type names to Polars DataFrames, or None if loading fails
    """
    print(f"Loading Polymarket data from {datadir}...")
    markets_path = datadir / "finance_politics_markets.parquet"
    odds_path = datadir / "finance_politics_odds_history.parquet"
    summary_path = datadir / "finance_politics_summary.parquet"

    data: dict[str, pl.DataFrame] = {}

    try:
        with track_memory("loading Polymarket data"):
            if markets_path.exists():
                # Load with lazy scan, then collect and handle datetime columns
                markets_df = pl.scan_parquet(markets_path).collect()
                
                # Convert datetime columns only if they exist and are strings
                # (parquet files may already have proper datetime types)
                datetime_cols = []
                for col_name in ["created_at", "end_date"]:
                    if col_name in markets_df.columns:
                        col_dtype = markets_df[col_name].dtype
                        if col_dtype == pl.String or col_dtype == pl.Utf8:
                            datetime_cols.append(pl.col(col_name).str.to_datetime())
                
                if datetime_cols:
                    markets_df = markets_df.with_columns(datetime_cols)
                
                # Fix timestamp corruption
                for col in markets_df.columns:
                    if any(x in col.lower() for x in ["timestamp", "trade", "created_at", "end_date"]):
                        if markets_df[col].dtype == pl.Datetime or markets_df[col].dtype == pl.Date:
                            if not markets_df[col].is_empty() and markets_df[col].max() < datetime(2020, 1, 1):
                                markets_df = markets_df.with_columns((pl.col(col).cast(pl.Int64) * 1000).cast(pl.Datetime))
                                
                        # Enforce 2020+ constraint (replace placeholders/zeros with null)
                        if markets_df[col].dtype == pl.Datetime or markets_df[col].dtype == pl.Date:
                             markets_df = markets_df.with_columns(
                                 pl.when(pl.col(col) < datetime(2020, 1, 1))
                                 .then(None)
                                 .otherwise(pl.col(col))
                                 .alias(col)
                             )
                
                data["markets"] = markets_df
                print(f"Loaded {len(markets_df)} markets.")

            if odds_path.exists():
                odds_df = pl.scan_parquet(odds_path).collect()
                
                # Fix timestamp corruption
                for col in odds_df.columns:
                    if any(x in col.lower() for x in ["timestamp", "trade", "created_at", "end_date"]):
                        if odds_df[col].dtype == pl.Datetime or odds_df[col].dtype == pl.Date:
                            if not odds_df[col].is_empty() and odds_df[col].max() < datetime(2020, 1, 1):
                                odds_df = odds_df.with_columns((pl.col(col).cast(pl.Int64) * 1000).cast(pl.Datetime))
                                
                        # Enforce 2020+ constraint (replace placeholders/zeros with null)
                        if odds_df[col].dtype == pl.Datetime or odds_df[col].dtype == pl.Date:
                             odds_df = odds_df.with_columns(
                                 pl.when(pl.col(col) < datetime(2020, 1, 1))
                                 .then(None)
                                 .otherwise(pl.col(col))
                                 .alias(col)
                             )
                            
                data["odds"] = odds_df
                print(f"Loaded {len(odds_df)} odds history records.")

            if summary_path.exists():
                summary_df = pl.scan_parquet(summary_path).collect()
                
                # Fix timestamp corruption
                for col in summary_df.columns:
                    if any(x in col.lower() for x in ["timestamp", "trade", "created_at", "end_date"]):
                        if summary_df[col].dtype == pl.Datetime or summary_df[col].dtype == pl.Date:
                            if not summary_df[col].is_empty() and summary_df[col].max() < datetime(2020, 1, 1):
                                summary_df = summary_df.with_columns((pl.col(col).cast(pl.Int64) * 1000).cast(pl.Datetime))
                                
                        # Enforce 2020+ constraint (replace placeholders/zeros with null)
                        if summary_df[col].dtype == pl.Datetime or summary_df[col].dtype == pl.Date:
                             summary_df = summary_df.with_columns(
                                 pl.when(pl.col(col) < datetime(2020, 1, 1))
                                 .then(None)
                                 .otherwise(pl.col(col))
                                 .alias(col)
                             )
                            
                data["summary"] = summary_df
                print(f"Loaded {len(summary_df)} summary records.")

        return data if data else None
    except Exception as e:
        print(f"Error loading Polymarket data: {e}")
        return None



# --- Main Execution ---


def main() -> None:
    """Main execution function for EDA workflow."""
    # Track overall memory usage
    initial_memory = get_memory_usage_mb()
    print(f"\n[Memory] Initial memory usage: {format_memory(initial_memory)}\n")

    # Load data using lazy evaluation
    poly_data = load_polymarket_data(POLYMARKET_DIR)


    # Final memory summary
    final_memory = get_memory_usage_mb()
    total_delta = final_memory - initial_memory
    print(
        f"\n[Memory] Final memory usage: {format_memory(final_memory)} "
        f"(Total Δ: {format_memory(total_delta)})"
    )
    print("\nEDA Layout Complete. Check the 'plots' directory for visualizations.")


if __name__ == "__main__":
    main()
