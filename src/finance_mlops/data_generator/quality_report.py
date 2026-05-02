"""
Data Quality Report for the Finance Data Generator outputs.

Computes and prints statistics for all offline (Parquet) and
streaming (JSONL) datasets, verifying that intentional data-quality
challenges match the specification in 01_data_generator_finance.md.

Usage (standalone):
    python -m finance_mlops.data_generator.quality_report

Or called programmatically from main.py via ``run(offline_paths, stream_path)``.
"""
import json
import logging
import os

import pandas as pd

from finance_mlops.data_generator import config

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

SEPARATOR = "─" * 70
SECTION   = "═" * 70


def _section(title: str) -> None:
    print(f"\n{SECTION}")
    print(f"  {title}")
    print(SECTION)


def _subsection(title: str) -> None:
    print(f"\n  {SEPARATOR[:60]}")
    print(f"  {title}")
    print(f"  {SEPARATOR[:60]}")


# ─────────────────────────────────────────────────────────────────────────────
# Offline checks
# ─────────────────────────────────────────────────────────────────────────────

def _check_skew(tx: pd.DataFrame) -> None:
    """Verify 85 % city skew and 80 % retail merchant category."""
    _subsection("Skew Distribution")

    # City skew (transactions)
    big_city_mask = tx["city"].isin(config.BIG_CITIES)
    big_city_pct  = big_city_mask.mean() * 100
    spec_target   = config.BIG_CITY_WEIGHT * 100
    status = "✅" if abs(big_city_pct - spec_target) < 3 else "⚠️ "
    print(f"  {status}  City skew (big cities): {big_city_pct:.1f}%  "
          f"(spec ≈ {spec_target:.0f}%)")

    # Top 5 cities breakdown
    top_cities = (
        tx["city"].value_counts(normalize=True).head(8) * 100
    ).round(1)
    print("\n     City distribution (top 8):")
    for city, pct in top_cities.items():
        print(f"       {city:<30} {pct:>5.1f}%")


def _check_merchant_skew(merchants: pd.DataFrame) -> None:
    retail_mask = merchants["category_code"].isin(config.MERCHANT_RETAIL_CATEGORIES)
    retail_pct  = retail_mask.mean() * 100
    spec_target = config.MERCHANT_RETAIL_WEIGHT * 100
    status = "✅" if abs(retail_pct - spec_target) < 3 else "⚠️ "
    print(f"\n  {status}  Merchant retail category skew: {retail_pct:.1f}%  "
          f"(spec ≈ {spec_target:.0f}%)")


def _check_cardinality(tables: dict[str, pd.DataFrame]) -> None:
    """Report approximate distinct counts for key ID columns."""
    _subsection("Cardinality (approx_count_distinct)")
    id_columns = {
        "customers":    ["customer_id"],
        "accounts":     ["account_id", "customer_id"],
        "merchants":    ["merchant_id"],
        "transactions": ["transaction_id", "account_id", "merchant_id"],
        "transaction_details": ["detail_id", "transaction_id"],
    }
    for table, cols in id_columns.items():
        if table not in tables:
            continue
        df = tables[table]
        print(f"\n  {table} ({len(df):,} rows):")
        for col in cols:
            if col in df.columns:
                n = df[col].nunique()
                print(f"     {col:<30}  {n:>10,} distinct values")


def _check_schema_evolution(tx: pd.DataFrame) -> None:
    """Validate null pattern for device_id / ip_address in old partitions."""
    _subsection("Schema Evolution (old partitions)")
    cutoff = config.SCHEMA_EVOLUTION_CUTOFF

    # Parse timestamp column
    tx_ts = pd.to_datetime(tx["transaction_timestamp"])
    old   = tx[tx_ts < cutoff]
    new   = tx[tx_ts >= cutoff]

    for col in ("device_id", "ip_address"):
        if col not in tx.columns:
            print(f"  ❌  Column '{col}' missing from transactions table.")
            continue
        old_null_rate = old[col].isna().mean() * 100
        new_null_rate = new[col].isna().mean() * 100
        old_status = "✅" if old_null_rate > 95 else "⚠️ "
        new_status = "✅" if new_null_rate < 5  else "⚠️ "
        print(f"  {old_status}  {col} null rate — old partitions (before cutoff): "
              f"{old_null_rate:.1f}%  (expect ~100%)")
        print(f"  {new_status}  {col} null rate — new partitions (after cutoff):  "
              f"{new_null_rate:.1f}%  (expect  ~0%)")

    print(f"\n     Cutoff date: {cutoff.strftime('%Y-%m-%d')}")
    print(f"     Old partitions: {len(old):>10,} rows")
    print(f"     New partitions: {len(new):>10,} rows")


def _check_offline_duplicates(tx: pd.DataFrame) -> None:
    """Report duplicate transaction rate and before/after dedup counts."""
    _subsection("Offline Duplicate Rate")

    total       = len(tx)
    dedup_keys  = ["account_id", "merchant_id", "amount", "transaction_timestamp"]
    after_dedup = tx.drop_duplicates(subset=dedup_keys)
    n_dupes     = total - len(after_dedup)
    dup_rate    = n_dupes / total * 100 if total else 0
    spec_rate   = config.DUPLICATE_RATE * 100

    status = "✅" if abs(dup_rate - spec_rate) < 0.5 else "⚠️ "
    print(f"  {status}  Transactions before dedup : {total:>10,}")
    print(f"       Transactions after dedup  : {len(after_dedup):>10,}")
    print(f"       Duplicate rows            : {n_dupes:>10,}  ({dup_rate:.2f}%)")
    print(f"       Spec target               :            {spec_rate:.1f}%")


def _check_fraud_labels(tx: pd.DataFrame) -> None:
    """Show fraud rate and breakdown by drift scenario."""
    _subsection("Fraud / Drift Labels")
    if "is_fraudulent" not in tx.columns:
        print("  ❌  'is_fraudulent' column not found.")
        return

    total = len(tx)
    fraud = tx["is_fraudulent"].sum()
    print(f"  ℹ️   Total transactions : {total:>10,}")
    print(f"  ℹ️   Fraudulent (flag=1): {fraud:>10,}  ({100*fraud/max(1,total):.2f}%)")


# ─────────────────────────────────────────────────────────────────────────────
# Streaming checks
# ─────────────────────────────────────────────────────────────────────────────

def _load_stream(stream_path: str) -> list[dict]:
    events = []
    with open(stream_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def _check_streaming_stats(events: list[dict]) -> None:
    """Report burst, late-arrival, and duplicate rates for the event stream."""
    _subsection("Streaming Event Stats")

    total  = len(events)
    bursts = sum(1 for e in events if e.get("_is_burst", False))
    late   = sum(1 for e in events if e.get("_late_arrival", False))
    dupes  = sum(1 for e in events if e.get("_is_duplicate", False))

    burst_pct = bursts / total * 100 if total else 0
    late_pct  = late  / total * 100 if total else 0
    dupe_pct  = dupes / total * 100 if total else 0

    late_status = "✅" if abs(late_pct - config.LATE_ARRIVAL_RATE * 100) < 2 else "⚠️ "
    dupe_status = "✅" if abs(dupe_pct - config.DUPLICATE_RATE_STREAM * 100) < 0.5 else "⚠️ "

    print(f"  ℹ️   Total events       : {total:>10,}")
    print(f"  ℹ️   Burst events       : {bursts:>10,}  ({burst_pct:.1f}%)")
    print(f"  {late_status}  Late arrivals      : {late:>10,}  ({late_pct:.1f}%)  "
          f"(spec ≈ {config.LATE_ARRIVAL_RATE*100:.0f}%)")
    print(f"  {dupe_status}  Duplicate events   : {dupes:>10,}  ({dupe_pct:.2f}%)  "
          f"(spec ≈ {config.DUPLICATE_RATE_STREAM*100:.1f}%)")

    # Dedup: group by event_id and take first
    from collections import Counter
    event_id_counts = Counter(e["event_id"] for e in events)
    unique_after_dedup = sum(1 for c in event_id_counts.values() if c == 1)
    multi             = sum(1 for c in event_id_counts.values() if c > 1)
    print(f"\n  ℹ️   Unique event_ids         : {len(event_id_counts):>10,}")
    print(f"  ℹ️   event_ids with duplicates : {multi:>10,}")
    print(f"  ℹ️   Events after dedup        : {len(event_id_counts):>10,}")


def _check_event_types(events: list[dict]) -> None:
    """Show breakdown of event_type distribution."""
    _subsection("Event Type Distribution")
    from collections import Counter
    counts = Counter(e["event_type"] for e in events)
    total  = len(events)
    for etype, n in sorted(counts.items(), key=lambda x: -x[1]):
        pct = n / total * 100
        print(f"  {'✅' if etype in ['fraud_alert'] else 'ℹ️ '}  "
              f"{etype:<25} {n:>8,}  ({pct:.1f}%)")


def _check_source_field(events: list[dict]) -> None:
    """Verify the `source` field is present and populated correctly."""
    _subsection("Source Field (app|web|atm|pos)")
    valid_sources = {"app", "web", "atm", "pos"}
    from collections import Counter
    source_counts = Counter(e.get("source", "MISSING") for e in events)
    total = len(events)
    all_valid = all(s in valid_sources for s in source_counts)
    status = "✅" if all_valid else "❌"
    print(f"  {status}  All source values valid: {all_valid}")
    for src, n in sorted(source_counts.items(), key=lambda x: -x[1]):
        pct = n / total * 100
        marker = "✅" if src in valid_sources else "❌"
        print(f"  {marker}  {src:<20} {n:>8,}  ({pct:.1f}%)")


# ─────────────────────────────────────────────────────────────────────────────
# Main entry-point
# ─────────────────────────────────────────────────────────────────────────────

def run(offline_paths: dict, stream_path: str | None) -> None:
    """
    Execute all quality checks and print a human-readable report.

    Parameters
    ----------
    offline_paths : dict
        Mapping of table name → Parquet file path (from offline_generator.run).
    stream_path : str | None
        Path to the events_stream.jsonl file (from streaming_generator.run).
    """
    print(f"\n{SECTION}")
    print("  Finance Data Generator – Quality Report")
    print(SECTION)

    # ── Offline ─────────────────────────────────────────────────────────────
    if offline_paths:
        _section("OFFLINE DATA QUALITY")
        tables: dict[str, pd.DataFrame] = {}
        for name, path in offline_paths.items():
            if os.path.exists(path):
                tables[name] = pd.read_parquet(path)
                log.info("Loaded %s: %d rows", name, len(tables[name]))

        if "transactions" in tables:
            tx = tables["transactions"]
            _check_skew(tx)
        if "merchants" in tables:
            _check_merchant_skew(tables["merchants"])

        _check_cardinality(tables)

        if "transactions" in tables:
            _check_schema_evolution(tables["transactions"])
            _check_offline_duplicates(tables["transactions"])
            _check_fraud_labels(tables["transactions"])

    # ── Streaming ───────────────────────────────────────────────────────────
    if stream_path and os.path.exists(stream_path):
        _section("STREAMING DATA QUALITY")
        events = _load_stream(stream_path)
        _check_streaming_stats(events)
        _check_event_types(events)
        _check_source_field(events)

    print(f"\n{SECTION}")
    print("  Quality Report Complete")
    print(f"{SECTION}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Standalone execution
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-8s %(message)s")

    offline_paths = {
        "customers":           os.path.join(config.OFFLINE_OUTPUT_DIR, "customers.parquet"),
        "accounts":            os.path.join(config.OFFLINE_OUTPUT_DIR, "accounts.parquet"),
        "merchants":           os.path.join(config.OFFLINE_OUTPUT_DIR, "merchants.parquet"),
        "transactions":        os.path.join(config.OFFLINE_OUTPUT_DIR, "transactions.parquet"),
        "transaction_details": os.path.join(config.OFFLINE_OUTPUT_DIR, "transaction_details.parquet"),
    }
    stream_path = os.path.join(config.STREAMING_OUTPUT_DIR, "events_stream.jsonl")
    run(offline_paths, stream_path)
