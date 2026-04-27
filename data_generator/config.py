"""
Configuration and constants for the Finance Data Generator.

Drift simulation is controlled by DRIFT_START_DATE and the
SCENARIO_A / SCENARIO_B knobs below. Affected accounts produce
transactions that are flagged is_fraudulent = 1.
"""
from datetime import datetime, timedelta

# ──────────────────────────────────────────────
# Scale
# ──────────────────────────────────────────────
NUM_CUSTOMERS    = 10_000
NUM_ACCOUNTS     = 12_000   # some customers have >1 account
NUM_MERCHANTS    = 3_000
NUM_TRANSACTIONS = 200_000

# ──────────────────────────────────────────────
# Date / time range for offline history
# ──────────────────────────────────────────────
HISTORY_START = datetime(2023, 1, 1)
HISTORY_END   = datetime(2024, 12, 31, 23, 59, 59)

# Cutoff: transactions BEFORE this date are "old partitions"
# → missing device_id and ip_address (schema evolution)
SCHEMA_EVOLUTION_CUTOFF = HISTORY_START + (HISTORY_END - HISTORY_START) * 0.6

# ──────────────────────────────────────────────
# Feature Drift Simulation
# ──────────────────────────────────────────────
# Date after which drift accounts start behaving anomalously
DRIFT_START_DATE = datetime(2024, 11, 1)

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

# Total stream duration to simulate (in minutes)
STREAM_DURATION_MINUTES = 60

# Minutes during which bursts occur (e.g. 30-min window)
BURST_WINDOWS = [
    (20, 25),   # 5-minute spike (payday simulation)
    (45, 50),   # 5-minute spike (Black Friday)
]

LATE_ARRIVAL_RATE = 0.12        # 12 % events arrive late
MAX_LATE_SECONDS  = 3_600       # up to 1 hour late

# ──────────────────────────────────────────────
# Output paths
# ──────────────────────────────────────────────
OFFLINE_OUTPUT_DIR   = "output/offline"
STREAMING_OUTPUT_DIR = "output/streaming"
