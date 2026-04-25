"""
Shared utility helpers for the Finance Data Generator.
"""
import random
import uuid
from datetime import datetime, timedelta
from typing import List, Tuple


def new_uuid() -> str:
    """Generate a new random UUID string."""
    return str(uuid.uuid4())


def weighted_choice(choices: List[str], main_weight: float) -> str:
    """
    Pick from *choices* where the *first* item gets `main_weight`
    probability and the rest share the remainder equally.

    This mirrors the spec: 85 % big city, 80 % retail merchant, etc.
    """
    if len(choices) == 1:
        return choices[0]

    tail_weight = (1.0 - main_weight) / (len(choices) - 1)
    weights = [main_weight] + [tail_weight] * (len(choices) - 1)
    return random.choices(choices, weights=weights, k=1)[0]


def biased_city(big_cities: List[str], other_cities: List[str], bias: float) -> str:
    """
    Return a city name with `bias` probability from *big_cities*
    and (1-bias) from *other_cities*.
    """
    if random.random() < bias:
        return random.choice(big_cities)
    return random.choice(other_cities)


def random_timestamp(start: datetime, end: datetime) -> datetime:
    """Uniform random datetime within [start, end]."""
    delta = end - start
    random_seconds = random.randint(0, int(delta.total_seconds()))
    return start + timedelta(seconds=random_seconds)


def iso(dt: datetime) -> str:
    """Return an ISO-8601 string with seconds precision."""
    return dt.strftime("%Y-%m-%dT%H:%M:%S")
