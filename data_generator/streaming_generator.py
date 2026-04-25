"""
Streaming data generator – simulates a unified Kafka-topic-like event stream
and writes it to newline-delimited JSON files under STREAMING_OUTPUT_DIR.

Event types: login_attempt | balance_inquiry | transaction_auth | pin_change

Data-quality challenges introduced:
  - Bursts: within BURST_WINDOWS the volume jumps from baseline to 3000/min.
  - Late arrivals: 12 % of events have created_ts > event_timestamp
    (by up to MAX_LATE_SECONDS).
"""
import json
import logging
import math
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

EVENT_TYPES = ["login_attempt", "balance_inquiry", "transaction_auth", "pin_change"]
DEVICE_TYPES = ["mobile_ios", "mobile_android", "web", "atm", "pos"]


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
) -> dict:
    """Build a single event dict."""
    event_type = random.choices(
        EVENT_TYPES,
        weights=[40, 25, 25, 10],
        k=1,
    )[0]

    # Late arrival
    if random.random() < config.LATE_ARRIVAL_RATE:
        late_seconds = random.randint(60, config.MAX_LATE_SECONDS)
        created_ts = event_ts + timedelta(seconds=late_seconds)
    else:
        # Normal: created_ts slightly after event (processing delay 0–5 s)
        created_ts = event_ts + timedelta(seconds=random.randint(0, 5))

    is_tx = event_type == "transaction_auth"

    event = {
        "event_id":        new_uuid(),
        "event_type":      event_type,
        "event_timestamp": iso(event_ts),
        "created_ts":      iso(created_ts),
        "account_id":      account_id,
        "session_id":      new_uuid(),
        "device_type":     random.choice(DEVICE_TYPES),
        "location_ip":     fake.ipv4(),
        "transaction_id":  new_uuid() if is_tx else None,
        "amount":          round(random.uniform(5, 5_000), 2) if is_tx else None,
        # metadata
        "_burst_minute":   minute,
        "_is_burst":       _is_burst_minute(minute),
        "_late_arrival":   created_ts > event_ts + timedelta(seconds=5),
    }
    return event


def generate_stream(
    account_ids: list,
    duration_minutes: int = config.STREAM_DURATION_MINUTES,
) -> list:
    """
    Simulate *duration_minutes* of event traffic.
    Returns a flat list of event dicts ordered by event_timestamp.
    """
    log.info("Generating streaming events over %d minutes …", duration_minutes)
    stream_start = datetime(2025, 1, 10, 9, 0, 0)  # arbitrary reference start
    all_events = []

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
            all_events.append(_make_event(account_id, event_ts, minute))

    # Sort by event_timestamp (stream order)
    all_events.sort(key=lambda e: e["event_timestamp"])
    log.info("Generated %d streaming events total.", len(all_events))
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
