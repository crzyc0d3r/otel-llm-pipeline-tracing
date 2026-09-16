"""Tools the agentic path can call, each one its own span.

Tool spans carry the tool name and its arguments. That pairing is what lets the
analysis step notice an agent calling the same tool with the same arguments over
and over - the single most common way an agentic pipeline gets slow and
expensive without failing.
"""

from __future__ import annotations

import time
from typing import Callable, Dict

from opentelemetry.trace import Status, StatusCode

from .telemetry import APP_TOOL, APP_TOOL_ARGS

_WEATHER = {
    "kyoto": "18C, light rain, humid",
    "lisbon": "24C, clear",
    "reykjavik": "3C, overcast, windy",
}

_FLIGHTS = {
    "kyoto": "3 nonstop options, from 780 USD",
    "lisbon": "6 nonstop options, from 410 USD",
    "reykjavik": "2 nonstop options, from 520 USD",
}

_HOTELS = {
    "kyoto": "median 210 USD/night in Gion",
    "lisbon": "median 140 USD/night in Alfama",
    "reykjavik": "median 260 USD/night downtown",
}


def _lookup(table: Dict[str, str], city: str, label: str) -> str:
    key = city.strip().lower()
    if key not in table:
        raise KeyError("no {} data for {!r}".format(label, city))
    return table[key]


def lookup_weather(city: str) -> str:
    time.sleep(0.005)
    return _lookup(_WEATHER, city, "weather")


def lookup_flights(city: str) -> str:
    time.sleep(0.008)
    return _lookup(_FLIGHTS, city, "flight")


def lookup_hotels(city: str) -> str:
    time.sleep(0.006)
    return _lookup(_HOTELS, city, "hotel")


REGISTRY: Dict[str, Callable[[str], str]] = {
    "lookup_weather": lookup_weather,
    "lookup_flights": lookup_flights,
    "lookup_hotels": lookup_hotels,
}


def call_tool(tracer, name: str, city: str) -> str:
    """Execute one tool inside a span, recording name and arguments."""

    with tracer.start_as_current_span("tool {}".format(name)) as span:
        span.set_attribute(APP_TOOL, name)
        span.set_attribute(APP_TOOL_ARGS, "city={}".format(city))
        handler = REGISTRY.get(name)
        if handler is None:
            error = KeyError("unknown tool {!r}".format(name))
            span.record_exception(error)
            span.set_status(Status(StatusCode.ERROR, str(error)))
            raise error
        try:
            result = handler(city)
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR, str(exc)))
            raise
        span.set_attribute("advisor.tool.result_characters", len(result))
        span.set_status(Status(StatusCode.OK))
        return result
