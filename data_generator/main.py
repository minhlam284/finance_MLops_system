"""
Entry point for the Finance Data Generator.

Usage:
    python -m data_generator.main
    python -m data_generator.main --offline-only
    python -m data_generator.main --streaming-only
"""
import argparse
import logging
import sys
import time

import pandas as pd

from data_generator import offline_generator, streaming_generator


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
            print(f"      {table:<16}  {len(df):>8,} rows  →  {path}")

    if stream_path:
        with open(stream_path, "r", encoding="utf-8") as fh:
            n_events = sum(1 for _ in fh)
        print(f"\n  📡  Streaming (JSONL)")
        print(f"      events_stream    {n_events:>8,} events →  {stream_path}")

    print(f"\n{separator}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Finance Data Generator")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--offline-only",   action="store_true",
                       help="Generate only the offline Parquet tables.")
    group.add_argument("--streaming-only", action="store_true",
                       help="Generate only the streaming event stream.")
    args = parser.parse_args()

    _setup_logging()
    log = logging.getLogger(__name__)
    t0 = time.perf_counter()

    offline_paths: dict = {}
    stream_path:   str | None = None

    # ── Offline ──────────────────────────────────────────
    if not args.streaming_only:
        offline_paths = offline_generator.run()

    # ── Streaming ────────────────────────────────────────
    if not args.offline_only:
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


if __name__ == "__main__":
    main()
