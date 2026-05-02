"""
Entry point for the Finance Data Generator.

Usage:
    python -m finance_mlops.data_generator.main
    python -m finance_mlops.data_generator.main --offline-only
    python -m finance_mlops.data_generator.main --streaming-only
    python -m finance_mlops.data_generator.main --report           # generate data + print quality report
    python -m finance_mlops.data_generator.main --report-only      # print quality report on existing output
"""
import argparse
import logging
import sys
import time

import pandas as pd

from finance_mlops.data_generator import offline_generator, streaming_generator


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def _print_summary(offline_paths: dict, stream_path: str | None) -> None:
    """Print a human-readable summary of what was generated."""
    separator = "─" * 60
    print(f"\n{separator}")
    print("  Finance Data Generator – Output Summary")
    print(separator)

    if offline_paths:
        print("\n  📦  Offline (Parquet)")
        for table, path in offline_paths.items():
            df = pd.read_parquet(path)
            print(f"      {table:<22}  {len(df):>10,} rows  →  {path}")

    if stream_path:
        with open(stream_path, "r", encoding="utf-8") as fh:
            n_events = sum(1 for _ in fh)
        print(f"\n  📡  Streaming (JSONL)")
        print(f"      events_stream    {n_events:>10,} events →  {stream_path}")

    print(f"\n{separator}\n")


def run(streaming_only: bool = False, offline_only: bool = False) -> tuple[dict, str | None]:
    t0 = time.perf_counter()
    log = logging.getLogger(__name__)
    offline_paths: dict  = {}
    stream_path: str | None = None

    # ── Offline ─────────────────────────────────────────────────────────────
    if not streaming_only:
        offline_paths = offline_generator.run()

    # ── Streaming ───────────────────────────────────────────────────────────
    if not offline_only:
        # Reuse account IDs from the offline dataset when available,
        # otherwise generate placeholder IDs.
        if offline_paths.get("accounts"):
            accounts_df = pd.read_parquet(offline_paths["accounts"])
            account_ids = accounts_df["account_id"].tolist()
        else:
            import uuid
            account_ids = [str(uuid.uuid4()) for _ in range(1_000)]

        stream_path = streaming_generator.run(account_ids)

    elapsed = time.perf_counter() - t0
    log.info("All done in %.1f s", elapsed)
    _print_summary(offline_paths, stream_path)
    return offline_paths, stream_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Finance Data Generator")
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--offline-only", action="store_true",
        help="Generate only the offline Parquet tables.",
    )
    mode_group.add_argument(
        "--streaming-only", action="store_true",
        help="Generate only the streaming event stream.",
    )
    mode_group.add_argument(
        "--report-only", action="store_true",
        help="Run the quality report on existing output without regenerating data.",
    )
    parser.add_argument(
        "--report", action="store_true",
        help="After generating data, run and print the data quality report.",
    )
    args = parser.parse_args()

    _setup_logging()
    log = logging.getLogger(__name__)

    # ── Report-only mode (no data generation) ───────────────────────────────
    if args.report_only:
        from finance_mlops.data_generator import config, quality_report
        offline_paths = {
            tbl: f"{config.OFFLINE_OUTPUT_DIR}/{tbl}.parquet"
            for tbl in (
                "customers", "accounts", "merchants",
                "transactions", "transaction_details",
            )
        }
        stream_path = f"{config.STREAMING_OUTPUT_DIR}/events_stream.jsonl"
        quality_report.run(offline_paths, stream_path)
        return

    offline_paths, stream_path = run(streaming_only=args.streaming_only, offline_only=args.offline_only)

    # ── Optional quality report ──────────────────────────────────────────────
    if args.report:
        from finance_mlops.data_generator import quality_report
        quality_report.run(offline_paths, stream_path)


if __name__ == "__main__":
    main()
