"""
Offline data generator – creates Parquet files for all four tables:
  customers, accounts, merchants, transactions
with intentional data-quality challenges.
"""
import os
import random
import logging
from datetime import timedelta

import pandas as pd
from faker import Faker

from data_generator import config
from data_generator.utils import (
    new_uuid,
    biased_city,
    random_timestamp,
    iso,
)

log = logging.getLogger(__name__)
fake = Faker()
random.seed(42)
Faker.seed(42)

# ──────────────────────────────────────────────
# Reference data
# ──────────────────────────────────────────────
OTHER_CITIES = [
    "Vung Tau", "Nha Trang", "Bien Hoa", "Hue", "Buon Ma Thuot",
    "Quy Nhon", "Long Xuyen", "Rach Gia", "My Tho", "Thai Nguyen",
]

COUNTRIES = ["VN", "SG", "TH", "US", "JP", "KR", "GB", "AU"]

CREDIT_SEGMENTS = ["standard", "gold", "platinum", "diamond"]
KYC_STATUSES    = ["verified", "pending", "failed"]

ACCOUNT_TYPES  = ["checking", "savings", "credit"]
CURRENCIES     = ["VND", "USD", "EUR", "SGD"]

# MCC codes – retail first (matches MERCHANT_RETAIL_CATEGORIES in config)
ALL_MCC = config.MERCHANT_RETAIL_CATEGORIES + [
    "5812", "7011", "4111", "5945", "5047", "6011", "4816", "5999",
]
RISK_TIERS = ["low", "medium", "high"]

TX_STATUSES = ["approved", "declined", "pending"]


# ──────────────────────────────────────────────
# Table generators
# ──────────────────────────────────────────────

def generate_customers(n: int = config.NUM_CUSTOMERS) -> pd.DataFrame:
    """Generate the `customers` table."""
    log.info("Generating %d customers …", n)
    rows = []
    for _ in range(n):
        rows.append({
            "customer_id":     new_uuid(),
            "signup_ts":       iso(random_timestamp(config.HISTORY_START, config.HISTORY_END)),
            "country":         random.choices(COUNTRIES, weights=[50,15,10,8,5,5,4,3], k=1)[0],
            "credit_segment":  random.choices(CREDIT_SEGMENTS, weights=[50,30,15,5], k=1)[0],
            "kyc_status":      random.choices(KYC_STATUSES, weights=[80,15,5], k=1)[0],
            "first_name":      fake.first_name(),
            "last_name":       fake.last_name(),
            "email":           fake.email(),
            "phone":           fake.phone_number(),
        })
    return pd.DataFrame(rows)


def generate_accounts(customers_df: pd.DataFrame,
                      n: int = config.NUM_ACCOUNTS) -> pd.DataFrame:
    """
    Generate the `accounts` table.
    Some customers have >1 account, achieved by sampling with replacement.
    """
    log.info("Generating %d accounts …", n)
    customer_ids = customers_df["customer_id"].tolist()
    rows = []
    for _ in range(n):
        rows.append({
            "account_id":    new_uuid(),
            "customer_id":   random.choice(customer_ids),
            "account_type":  random.choices(ACCOUNT_TYPES, weights=[40, 35, 25], k=1)[0],
            "credit_limit":  round(random.uniform(1_000, 50_000), 2),
            "currency":      random.choices(CURRENCIES, weights=[60,25,10,5], k=1)[0],
            "created_ts":    iso(random_timestamp(config.HISTORY_START, config.HISTORY_END)),
            "is_active":     random.choices([True, False], weights=[90, 10], k=1)[0],
        })
    return pd.DataFrame(rows)


def generate_merchants(n: int = config.NUM_MERCHANTS) -> pd.DataFrame:
    """
    Generate the `merchants` table.
    80 % of merchants belong to retail / supermarket MCC codes.
    """
    log.info("Generating %d merchants …", n)
    rows = []
    for _ in range(n):
        # 80 % retail bias
        if random.random() < config.MERCHANT_RETAIL_WEIGHT:
            mcc = random.choice(config.MERCHANT_RETAIL_CATEGORIES)
        else:
            non_retail = [m for m in ALL_MCC if m not in config.MERCHANT_RETAIL_CATEGORIES]
            mcc = random.choice(non_retail)

        rows.append({
            "merchant_id":   new_uuid(),
            "merchant_name": fake.company(),
            "category_code": mcc,
            "country":       random.choice(COUNTRIES),
            "city":          biased_city(config.BIG_CITIES, OTHER_CITIES, config.BIG_CITY_WEIGHT),
            "risk_tier":     random.choices(RISK_TIERS, weights=[70, 20, 10], k=1)[0],
        })
    return pd.DataFrame(rows)


def generate_transactions(accounts_df: pd.DataFrame,
                          merchants_df: pd.DataFrame,
                          n: int = config.NUM_TRANSACTIONS) -> pd.DataFrame:
    """
    Generate the `transactions` table with:
    - 85 % of transactions in big cities  (skew)
    - Schema evolution: partitions before SCHEMA_EVOLUTION_CUTOFF lack
      `device_id` and `ip_address`.
    - 2 % duplicates (optional, controlled by DUPLICATE_RATE).
    """
    log.info("Generating %d transactions …", n)
    account_ids  = accounts_df["account_id"].tolist()
    merchant_ids = merchants_df["merchant_id"].tolist()

    cutoff = config.SCHEMA_EVOLUTION_CUTOFF

    rows = []
    for _ in range(n):
        ts = random_timestamp(config.HISTORY_START, config.HISTORY_END)
        city = biased_city(config.BIG_CITIES, OTHER_CITIES, config.BIG_CITY_WEIGHT)

        row = {
            "transaction_id":       new_uuid(),
            "account_id":           random.choice(account_ids),
            "merchant_id":          random.choice(merchant_ids),
            "transaction_timestamp": iso(ts),
            "amount":               round(random.uniform(5, 5_000), 2),
            "currency":             random.choices(CURRENCIES, weights=[60,25,10,5], k=1)[0],
            "status":               random.choices(TX_STATUSES, weights=[85,10,5], k=1)[0],
            "city":                 city,
        }

        # Schema evolution: new fields only exist after the cutoff
        if ts >= cutoff:
            row["device_id"]   = fake.uuid4()
            row["ip_address"]  = fake.ipv4()
        else:
            row["device_id"]   = None   # will become NaN / null in Parquet
            row["ip_address"]  = None

        rows.append(row)

    df = pd.DataFrame(rows)

    # ── Duplicates ────────────────────────────────────────
    n_dupes = int(len(df) * config.DUPLICATE_RATE)
    if n_dupes > 0:
        log.info("Injecting %d duplicate transactions (%.0f%%) …",
                 n_dupes, config.DUPLICATE_RATE * 100)
        dupes = df.sample(n=n_dupes, replace=True, random_state=1)
        df = pd.concat([df, dupes], ignore_index=True)

    return df


# ──────────────────────────────────────────────
# Save helpers
# ──────────────────────────────────────────────

def _save_parquet(df: pd.DataFrame, name: str) -> str:
    """Save a DataFrame as Parquet under OFFLINE_OUTPUT_DIR."""
    path = os.path.join(config.OFFLINE_OUTPUT_DIR, f"{name}.parquet")
    os.makedirs(config.OFFLINE_OUTPUT_DIR, exist_ok=True)
    df.to_parquet(path, index=False, engine="pyarrow")
    log.info("Saved %s → %s  (%d rows, %d cols)", name, path, len(df), len(df.columns))
    return path


def run() -> dict:
    """
    Orchestrate all offline table generation and persist to Parquet.
    Returns a dict of {table_name: file_path}.
    """
    customers_df    = generate_customers()
    accounts_df     = generate_accounts(customers_df)
    merchants_df    = generate_merchants()
    transactions_df = generate_transactions(accounts_df, merchants_df)

    paths = {}
    paths["customers"]    = _save_parquet(customers_df,    "customers")
    paths["accounts"]     = _save_parquet(accounts_df,     "accounts")
    paths["merchants"]    = _save_parquet(merchants_df,    "merchants")
    paths["transactions"] = _save_parquet(transactions_df, "transactions")

    return paths
