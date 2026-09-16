import pytest

from app.telemetry import configure


@pytest.fixture()
def telemetry():
    """A fresh in-memory tracer per test.

    Deliberately not the global tracer provider: a second
    ``set_tracer_provider`` call in the same process is silently ignored, so
    tests sharing a global would quietly export into the first one's collector.
    """

    instance = configure("memory")
    yield instance
    instance.shutdown()


@pytest.fixture()
def tracer(telemetry):
    return telemetry.tracer
