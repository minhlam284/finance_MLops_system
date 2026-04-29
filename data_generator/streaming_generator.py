"""
Streaming data generator – simulates a unified Kafka-topic-like event stream
and writes it to newline-delimited JSON files under STREAMING_OUTPUT_DIR.

Event types: login_attempt | balance_inquiry | transaction_auth | pin_change | fraud_alert

Data-quality challenges introduced:
  - Bursts: within BURST_WINDOWS the volume jumps from baseline to 3000/min.
  - Late arrivals: 12 % of events have created_ts > event_timestamp
    (by MIN_LATE_SECONDS–MAX_LATE_SECONDS = 5–45 minutes).
  - Duplicates: 1.5 % of events are re-emitted with a 1-3 minute delay,
    simulating network retries / at-least-once delivery.
"""
import json
import logging
import os
import random
from datetime import datetime, timedelta

from faker import Faker

from data_generator import config
from data_generator.utils import new_uuid, iso

log = logging.getLogger(__name__)
fake = Faker()
random.seed(99)
Faker.seed(99)

EVENT_TYPES = [
    "login_attempt",
    "balance_inquiry",
    "transaction_auth",
    "pin_change",
    "fraud_alert",
]

# device_type → source mapping (spec: app|web|atm|pos)
# mobile_ios and mobile_android both map to "app"
DEVICE_SOURCE_MAP = {
    "mobile_ios":     "app",
    "mobile_android": "app",
    "web":            "web",
    "atm":            "atm",
    "pos":            "pos",
}
DEVICE_TYPES = list(DEVICE_SOURCE_MAP.keys())


def _is_burst_minute(minute: int) -> bool:
    """Return True if *minute* falls inside a burst window."""
    for start, end in config.BURST_WINDOWS:
        if start <= minute < end:
            return True
    return False


def _make_event(
    account_id: str,
    event_ts: datetime,
    minute: int,
    event_id: str | None = None,
) -> dict:
    """
    Build a single event dict.

    Parameters
    ----------
    event_id : str | None
        If provided, reuse this ID (duplicate scenario).
    """
    event_type = random.choices(
        EVENT_TYPES,
        weights=[35, 20, 25, 10, 10],   # fraud_alert ~10 % of events
        k=1,
    )[0]

    # Late arrival
    if random.random() < config.LATE_ARRIVAL_RATE:
        late_seconds = random.randint(config.MIN_LATE_SECONDS, config.MAX_LATE_SECONDS)
        created_ts = event_ts + timedelta(seconds=late_seconds)
    else:
        # Normal: created_ts slightly after event (processing delay 0-5 s)
        created_ts = event_ts + timedelta(seconds=random.randint(0, 5))

    is_tx = event_type == "transaction_auth"
    device_type = random.choice(DEVICE_TYPES)
    source = DEVICE_SOURCE_MAP[device_type]

    event = {
        "event_id":        event_id if event_id else new_uuid(),
        "event_type":      event_type,
        "event_timestamp": iso(event_ts),
        "created_ts":      iso(created_ts),
        "account_id":      account_id,
        "session_id":      new_uuid(),
        "device_type":     device_type,
        "source":          source,                    # app|web|atm|pos per spec
        "location_ip":     fake.ipv4(),
        "transaction_id":  new_uuid() if is_tx else None,
        "merchant_id":     fake.uuid4() if is_tx else None,
        "amount":          round(random.uniform(5, 5_000), 2) if is_tx else None,
        # metadata helpers (useful for quality report / dedup validation)
        "_burst_minute":   minute,
        "_is_burst":       _is_burst_minute(minute),
        "_late_arrival":   created_ts > event_ts + timedelta(seconds=5),
        "_is_duplicate":   False,
    }
    return event


def generate_stream(
    account_ids: list,
    duration_minutes: int = config.STREAM_DURATION_MINUTES,
) -> list:
    """
    Simulate *duration_minutes* of event traffic.

    Includes:
    - Burst windows (3 000 events/min vs 100 baseline).
    - Late arrivals (12 % of events, 5-45 min delay).
    - Streaming duplicates (1.5 % of events re-emitted 1-3 min later).

    Returns a flat list of event dicts ordered by event_timestamp.
    """
    log.info("Generating streaming events over %d minutes …", duration_minutes)
    stream_start = datetime(2025, 1, 10, 9, 0, 0)  # arbitrary reference start
    all_events: list = []
    duplicate_buffer: list = []  # (emit_at_minute, original_event)

    for minute in range(duration_minutes):
        minute_start = stream_start + timedelta(minutes=minute)

        if _is_burst_minute(minute):
            volume = config.BURST_EVENTS_PER_MINUTE
        else:
            volume = config.BASELINE_EVENTS_PER_MINUTE

        for _ in range(volume):
            # Spread events uniformly within the minute
            offset_seconds = random.uniform(0, 59)
            event_ts = minute_start + timedelta(seconds=offset_seconds)
            account_id = random.choice(account_ids)
            evt = _make_event(account_id, event_ts, minute)
            all_events.append(evt)

            # Queue 1.5 % of events as future duplicates (1-3 min delay).
            # Clamp emit_at to the last valid minute so no duplicates are lost.
            if random.random() < config.DUPLICATE_RATE_STREAM:
                delay_minutes = random.randint(1, 3)
                emit_at = min(minute + delay_minutes, duration_minutes - 1)
                duplicate_buffer.append((emit_at, evt["event_id"], account_id))

        # Emit any queued duplicates for this minute
        still_pending = []
        for emit_at, orig_event_id, acct_id in duplicate_buffer:
            if emit_at == minute:
                dup_offset = random.uniform(0, 59)
                dup_ts = minute_start + timedelta(seconds=dup_offset)
                dup_evt = _make_event(acct_id, dup_ts, minute, event_id=orig_event_id)
                dup_evt["_is_duplicate"] = True
                all_events.append(dup_evt)
            else:
                still_pending.append((emit_at, orig_event_id, acct_id))
        duplicate_buffer = still_pending

    # Sort by event_timestamp (stream order)
    all_events.sort(key=lambda e: e["event_timestamp"])

    n_dupes = sum(1 for e in all_events if e["_is_duplicate"])
    n_late  = sum(1 for e in all_events if e["_late_arrival"])
    n_burst = sum(1 for e in all_events if e["_is_burst"])
    log.info(
        "Generated %d streaming events total "
        "(burst=%d, late=%d, duplicates=%d).",
        len(all_events), n_burst, n_late, n_dupes,
    )
    return all_events


def save_stream(events: list) -> str:
    """
    Persist events as newline-delimited JSON (one event per line).
    Returns the output file path.
    """
    os.makedirs(config.STREAMING_OUTPUT_DIR, exist_ok=True)
    path = os.path.join(config.STREAMING_OUTPUT_DIR, "events_stream.jsonl")

    with open(path, "w", encoding="utf-8") as fh:
        for evt in events:
            fh.write(json.dumps(evt, default=str) + "\n")

    log.info("Saved %d events → %s", len(events), path)
    return path


def run(account_ids: list) -> str:
    """Entry point: generate and save the event stream."""
    events = generate_stream(account_ids)
    return save_stream(events)
