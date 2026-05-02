"""
Configuration and constants for the Finance Data Generator.

Scale and all data-quality knobs are aligned with the specification in
``md/01_data_generator_finance.md``.

Drift simulation is controlled by DRIFT_START_DATE and the
SCENARIO_A / SCENARIO_B knobs below. Affected accounts produce
transactions that are flagged is_fraudulent = 1.
"""
from datetime import datetime

from finance_mlops.config.paths import SOURCE_OFFLINE_DIR, SOURCE_STREAMING_DIR

# ──────────────────────────────────────────────
# Scale  (matches 01_data_generator_finance.md)
# ──────────────────────────────────────────────
NUM_CUSTOMERS    = 120_000
NUM_ACCOUNTS     = 135_000   # some customers have >1 account
NUM_MERCHANTS    =  45_000
NUM_TRANSACTIONS = 1_000_000

# ──────────────────────────────────────────────
# Date / time range for offline history
# 180 days per spec: 2025-01-01 → 2025-06-30
# ──────────────────────────────────────────────
HISTORY_START = datetime(2025, 1, 1)
HISTORY_END   = datetime(2025, 6, 30, 23, 59, 59)

# Cutoff: transactions BEFORE this date are "old partitions"
# → missing device_id and ip_address (schema evolution)
# 60% of the 180-day window ≈ 2025-05-09; spec pins it at 2025-07-01
# We use the 60% formula to stay faithful to the spec logic.
SCHEMA_EVOLUTION_CUTOFF = HISTORY_START + (HISTORY_END - HISTORY_START) * 0.6

# ──────────────────────────────────────────────
# Feature Drift Simulation
# ──────────────────────────────────────────────
# Date after which drift accounts start behaving anomalously
DRIFT_START_DATE = datetime(2025, 5, 1)

# ── Scenario A: Transaction Frequency Drift (Carding Attack) ──
# Fraction of accounts that suddenly increase transaction frequency
SCENARIO_A_ACCOUNT_RATIO   = 0.05        # 5% of accounts
# Normal baseline: ~1.2 tx/account/day → drifted: ~5.5 tx/account/day
SCENARIO_A_BASELINE_TX_PER_DAY = 1.2
SCENARIO_A_DRIFTED_TX_PER_DAY  = 5.5
# PSI alert threshold for f_account_total_tx_90d
SCENARIO_A_PSI_THRESHOLD = 0.1

# ── Scenario B: Average Transaction Value Drift (High-ticket Fraud) ──
# Fraction of accounts with inflated transaction amounts
SCENARIO_B_ACCOUNT_RATIO = 0.05          # 5% of accounts
# Amount multiplier applied to post-drift transactions (~$45 → ~$180)
SCENARIO_B_AMOUNT_MULTIPLIER = 4.0
# PSI alert threshold for f_account_avg_tx_value_90d
SCENARIO_B_PSI_THRESHOLD = 0.1

# Global PSI alert threshold for agg_feature_health_daily
FEATURE_HEALTH_PSI_ALERT = 0.15

# ──────────────────────────────────────────────
# Offline data-quality knobs
# ──────────────────────────────────────────────
# Skew: 85 % of transactions come from these cities
BIG_CITIES = ["Ho Chi Minh City", "Hanoi", "Da Nang", "Can Tho", "Hai Phong"]
BIG_CITY_WEIGHT = 0.85          # probability mass on big cities

# 80 % merchants in retail / supermarket
MERCHANT_RETAIL_CATEGORIES = ["5411", "5412", "5300", "5310", "5399"]  # MCC codes
MERCHANT_RETAIL_WEIGHT = 0.80

DUPLICATE_RATE = 0.02           # 2 % duplicate transactions (optional)

# ──────────────────────────────────────────────
# Streaming knobs
# ──────────────────────────────────────────────
BASELINE_EVENTS_PER_MINUTE = 100
BURST_EVENTS_PER_MINUTE    = 3_000

# Total stream duration to simulate (in minutes) – full 60-min hour
STREAM_DURATION_MINUTES = 60

# 20-minute burst windows within the 60-minute simulation
# spec: "08:00-08:20", "12:00-12:20" → mapped to minutes 0-59 range
# We use offsets so both windows fit in one 60-min simulation:
#   window 1: minute 8 → 28  (first business burst)
#   window 2: minute 40 → 60  (second business burst)
BURST_WINDOWS = [
    (8,  28),   # 20-min spike (payday / salary window)
    (40, 60),   # 20-min spike (Black Friday / end-of-day window)
]

LATE_ARRIVAL_RATE = 0.12        # 12 % events arrive late
MIN_LATE_SECONDS  =   300       # minimum 5-minute delay
MAX_LATE_SECONDS  = 2_700       # maximum 45-minute delay (per spec [5, 45] min)

DUPLICATE_RATE_STREAM = 0.015   # 1.5 % duplicate events (same event_id, 1-3 min delay)

# ──────────────────────────────────────────────
# Output paths
# ──────────────────────────────────────────────
OFFLINE_OUTPUT_DIR = str(SOURCE_OFFLINE_DIR)
STREAMING_OUTPUT_DIR = str(SOURCE_STREAMING_DIR)
