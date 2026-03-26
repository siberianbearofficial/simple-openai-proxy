from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from openai_proxy.services.polza_cost_control import (
    CostLimitExceededError,
    PolzaCostControl,
)
from openai_proxy.settings import PolzaCostControlSettings


class FakeStream:
    def __init__(self, chunks: list[object]) -> None:
        self._chunks = list(chunks)
        self.closed = False

    def __aiter__(self) -> "FakeStream":
        return self

    async def __anext__(self) -> object:
        if not self._chunks:
            raise StopAsyncIteration
        return self._chunks.pop(0)

    async def close(self) -> None:
        self.closed = True


def _make_soft_limit_settings() -> PolzaCostControlSettings:
    return PolzaCostControlSettings(
        soft_threshold_rub=1.0,
        hard_threshold_rub=5.0,
        logs_api_base_url="https://logs.example.com",
        logs_api_username="logger",
        logs_api_password="secret",  # noqa: S106
    )


@pytest.mark.asyncio
async def test_soft_threshold_sends_notification_only_on_crossing() -> None:
    notifier = SimpleNamespace(notify=AsyncMock())
    monitor = PolzaCostControl(
        settings=_make_soft_limit_settings(),
        notifier=notifier,
        now_provider=lambda: 1_000.0,
    )

    await monitor.record_response_cost(
        provider="polza",
        response={"id": "gen_1", "usage": {"cost_rub": 0.6}},
        request={"model": "chat-1"},
    )
    await monitor.record_response_cost(
        provider="polza",
        response={"id": "gen_2", "usage": {"cost_rub": 0.5}},
        request={"model": "chat-1"},
    )
    await monitor.record_response_cost(
        provider="polza",
        response={"id": "gen_3", "usage": {"cost_rub": 0.1}},
        request={"model": "chat-1"},
    )

    notifier.notify.assert_awaited_once()
    notification_text, log_content = notifier.notify.await_args.args
    assert "Пробит мягкий лимит стоимости запросов к polza" in notification_text
    assert '"response_id": "gen_2"' in log_content
    assert '"current_total_cost_rub": 1.1' in log_content


@pytest.mark.asyncio
async def test_hard_threshold_blocks_requests_until_window_expires() -> None:
    clock = {"now": 2_000.0}
    monitor = PolzaCostControl(
        settings=PolzaCostControlSettings(hard_threshold_rub=1.0),
        now_provider=lambda: clock["now"],
    )

    await monitor.record_response_cost(
        provider="polza",
        response={"usage": {"cost_rub": 1.1}},
    )

    with pytest.raises(CostLimitExceededError, match="Превышен жесткий лимит"):
        await monitor.check_hard_limit("polza")

    clock["now"] += 3_601.0

    await monitor.check_hard_limit("polza")


@pytest.mark.asyncio
async def test_wrapped_stream_tracks_cost_from_usage_chunk() -> None:
    monitor = PolzaCostControl(
        settings=PolzaCostControlSettings(hard_threshold_rub=0.5),
        now_provider=lambda: 3_000.0,
    )
    stream = FakeStream(
        [
            {"choices": [{"delta": {"content": "hi"}}]},
            {"usage": {"cost_rub": 0.7}},
        ],
    )

    wrapped_stream = monitor.wrap_stream(
        provider="polza",
        response=stream,
        request={"model": "chat-1"},
    )
    chunks = [chunk async for chunk in wrapped_stream]

    await wrapped_stream.close()

    assert chunks[-1] == {"usage": {"cost_rub": 0.7}}
    assert stream.closed is True
    with pytest.raises(CostLimitExceededError, match="Превышен жесткий лимит"):
        await monitor.check_hard_limit("polza")
