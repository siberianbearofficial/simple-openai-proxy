from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from openai import OpenAIError

from openai_proxy import schemas
from openai_proxy.openai_compat import normalize_chat_completion_request
from openai_proxy.services.model_routing import (
    AUTO_OFFICIAL_MODEL,
    AUTO_POLZA_MODEL,
    DEFAULT_ROUTE_ATTEMPTS,
    ModelRouter,
)
from openai_proxy.services.openai_service import OpenAIService
from openai_proxy.services.polza_cost_control import CostLimitExceededError


class FakeStream:
    def __init__(self, chunks: list[object] | None = None) -> None:
        self._chunks = list(chunks or [])
        self.closed = False

    def __aiter__(self) -> "FakeStream":
        return self

    async def __anext__(self) -> object:
        if not self._chunks:
            raise StopAsyncIteration
        return self._chunks.pop(0)

    async def close(self) -> None:
        self.closed = True


def _make_request(model: str | schemas.OpenAIModel) -> dict[str, object]:
    return normalize_chat_completion_request(
        schemas.OpenAIRequest(
            model=model,
            messages=[
                schemas.OpenAIMessage(
                    role=schemas.OpenAIRole.USER,
                    content="ping",
                ),
            ],
        ).to_chat_completion_params(),
    )


def _make_legacy_request(model: str | schemas.OpenAIModel) -> schemas.OpenAIRequest:
    return schemas.OpenAIRequest(
        model=model,
        messages=[
            schemas.OpenAIMessage(
                role=schemas.OpenAIRole.USER,
                content="ping",
            ),
        ],
    )


@pytest.mark.asyncio
async def test_auto_prefers_deepseek() -> None:
    official = SimpleNamespace(request=AsyncMock())
    deepseek = SimpleNamespace(request=AsyncMock())
    polza = SimpleNamespace(request=AsyncMock())
    service = OpenAIService(official_client=official, deepseek_client=deepseek, polza_client=polza)

    request = _make_request("auto")
    deepseek.request.return_value = "ok"

    result = await service.request(request)

    assert result == "ok"
    deepseek.request.assert_awaited_once_with(
        {**request, "model": schemas.OpenAIModel.DEEPSEEK.value},
    )
    official.request.assert_not_called()
    polza.request.assert_not_called()


@pytest.mark.asyncio
async def test_auto_fallbacks_to_official_when_deepseek_fails() -> None:
    official = SimpleNamespace(request=AsyncMock())
    deepseek = SimpleNamespace(request=AsyncMock())
    polza = SimpleNamespace(request=AsyncMock())
    service = OpenAIService(official_client=official, deepseek_client=deepseek, polza_client=polza)

    request = _make_request("auto")
    deepseek.request.side_effect = OpenAIError("boom")
    official.request.return_value = "official"

    result = await service.request(request)

    assert result == "official"
    assert deepseek.request.await_count == DEFAULT_ROUTE_ATTEMPTS
    assert official.request.await_count == 1
    deepseek.request.assert_called_with({**request, "model": schemas.OpenAIModel.DEEPSEEK.value})
    official.request.assert_awaited_once_with({**request, "model": AUTO_OFFICIAL_MODEL})
    polza.request.assert_not_called()


@pytest.mark.asyncio
async def test_auto_fallbacks_to_polza_when_official_also_fails() -> None:
    official = SimpleNamespace(request=AsyncMock())
    deepseek = SimpleNamespace(request=AsyncMock())
    polza = SimpleNamespace(request=AsyncMock())
    service = OpenAIService(official_client=official, deepseek_client=deepseek, polza_client=polza)

    request = _make_request("auto")
    deepseek.request.side_effect = OpenAIError("deepseek boom")
    official.request.side_effect = OpenAIError("official boom")
    polza.request.return_value = "polza"

    result = await service.request(request)

    assert result == "polza"
    assert deepseek.request.await_count == DEFAULT_ROUTE_ATTEMPTS
    assert official.request.await_count == DEFAULT_ROUTE_ATTEMPTS
    polza.request.assert_awaited_once_with({**request, "model": AUTO_POLZA_MODEL})


@pytest.mark.asyncio
async def test_explicit_deepseek_model_uses_deepseek_only() -> None:
    official = SimpleNamespace(request=AsyncMock())
    deepseek = SimpleNamespace(request=AsyncMock())
    polza = SimpleNamespace(request=AsyncMock())
    service = OpenAIService(official_client=official, deepseek_client=deepseek, polza_client=polza)

    request = _make_request(schemas.OpenAIModel.DEEPSEEK.value)
    deepseek.request.return_value = "deepseek"

    result = await service.request(request)

    assert result == "deepseek"
    deepseek.request.assert_awaited_once_with(request)
    official.request.assert_not_called()
    polza.request.assert_not_called()


@pytest.mark.asyncio
async def test_prefixed_deepseek_model_routes_to_deepseek_client() -> None:
    official = SimpleNamespace(request=AsyncMock())
    deepseek = SimpleNamespace(request=AsyncMock())
    polza = SimpleNamespace(request=AsyncMock())
    service = OpenAIService(official_client=official, deepseek_client=deepseek, polza_client=polza)

    request = _make_request("deepseek:reasoner")
    deepseek.request.return_value = "deepseek"

    result = await service.request(request)

    assert result == "deepseek"
    deepseek.request.assert_awaited_once_with({**request, "model": "reasoner"})
    official.request.assert_not_called()
    polza.request.assert_not_called()


@pytest.mark.asyncio
async def test_prefixed_official_model_routes_to_official_client() -> None:
    official = SimpleNamespace(request=AsyncMock())
    deepseek = SimpleNamespace(request=AsyncMock())
    polza = SimpleNamespace(request=AsyncMock())
    service = OpenAIService(official_client=official, deepseek_client=deepseek, polza_client=polza)

    request = _make_request("official:gpt-4o-mini")
    official.request.return_value = "official"

    result = await service.request(request)

    assert result == "official"
    official.request.assert_awaited_once_with({**request, "model": "gpt-4o-mini"})
    deepseek.request.assert_not_called()
    polza.request.assert_not_called()


@pytest.mark.asyncio
async def test_polza_model_routes_to_polza_client() -> None:
    official = SimpleNamespace(request=AsyncMock())
    deepseek = SimpleNamespace(request=AsyncMock())
    polza = SimpleNamespace(request=AsyncMock())
    service = OpenAIService(official_client=official, deepseek_client=deepseek, polza_client=polza)

    request = _make_request("polza:chat-1")
    polza.request.return_value = "polza"

    result = await service.request(request)

    assert result == "polza"
    deepseek.request.assert_not_called()
    official.request.assert_not_called()
    polza.request.assert_awaited_once_with({**request, "model": "chat-1"})


@pytest.mark.asyncio
async def test_polza_model_checks_hard_limit_before_request() -> None:
    official = SimpleNamespace(request=AsyncMock())
    deepseek = SimpleNamespace(request=AsyncMock())
    polza = SimpleNamespace(request=AsyncMock())
    polza_cost_control = SimpleNamespace(
        check_hard_limit=AsyncMock(
            side_effect=CostLimitExceededError(
                total_cost_rub=10.0,
                threshold_rub=10.0,
                window_seconds=3600,
            ),
        ),
        record_response_cost=AsyncMock(),
    )
    service = OpenAIService(
        official_client=official,
        deepseek_client=deepseek,
        polza_client=polza,
        polza_cost_control=polza_cost_control,
    )

    request = _make_request("polza:chat-1")

    with pytest.raises(CostLimitExceededError):
        await service.request(request)

    polza_cost_control.check_hard_limit.assert_awaited_once_with("polza")
    polza_cost_control.record_response_cost.assert_not_called()
    deepseek.request.assert_not_called()
    official.request.assert_not_called()
    polza.request.assert_not_called()


@pytest.mark.asyncio
async def test_polza_model_records_response_cost() -> None:
    official = SimpleNamespace(request=AsyncMock())
    deepseek = SimpleNamespace(request=AsyncMock())
    polza = SimpleNamespace(request=AsyncMock())
    polza_cost_control = SimpleNamespace(
        check_hard_limit=AsyncMock(),
        record_response_cost=AsyncMock(),
    )
    service = OpenAIService(
        official_client=official,
        deepseek_client=deepseek,
        polza_client=polza,
        polza_cost_control=polza_cost_control,
    )

    request = _make_request("polza:chat-1")
    response = {"id": "gen_1", "usage": {"cost_rub": 0.42}}
    polza.request.return_value = response

    result = await service.request(request)

    assert result == response
    polza_cost_control.check_hard_limit.assert_awaited_once_with("polza")
    polza_cost_control.record_response_cost.assert_awaited_once_with(
        provider="polza",
        response=response,
        request={**request, "model": "chat-1"},
    )
    deepseek.request.assert_not_called()
    official.request.assert_not_called()
    polza.request.assert_awaited_once_with({**request, "model": "chat-1"})


@pytest.mark.asyncio
async def test_polza_stream_response_is_wrapped_for_cost_tracking() -> None:
    official = SimpleNamespace(request=AsyncMock())
    deepseek = SimpleNamespace(request=AsyncMock())
    polza = SimpleNamespace(request=AsyncMock())
    raw_stream = FakeStream([{"usage": {"cost_rub": 0.42}}])
    wrapped_stream = FakeStream()
    polza_cost_control = SimpleNamespace(
        check_hard_limit=AsyncMock(),
        wrap_stream=Mock(return_value=wrapped_stream),
        record_response_cost=AsyncMock(),
    )
    service = OpenAIService(
        official_client=official,
        deepseek_client=deepseek,
        polza_client=polza,
        polza_cost_control=polza_cost_control,
    )

    request = {**_make_request("polza:chat-1"), "stream": True}
    polza.request.return_value = raw_stream

    result = await service.request(request)

    assert result is wrapped_stream
    polza_cost_control.check_hard_limit.assert_awaited_once_with("polza")
    polza_cost_control.wrap_stream.assert_called_once_with(
        provider="polza",
        response=raw_stream,
        request={**request, "model": "chat-1"},
    )
    polza_cost_control.record_response_cost.assert_not_called()
    deepseek.request.assert_not_called()
    official.request.assert_not_called()
    polza.request.assert_awaited_once_with({**request, "model": "chat-1"})


@pytest.mark.asyncio
async def test_other_models_use_official_client() -> None:
    official = SimpleNamespace(request=AsyncMock())
    deepseek = SimpleNamespace(request=AsyncMock())
    polza = SimpleNamespace(request=AsyncMock())
    service = OpenAIService(official_client=official, deepseek_client=deepseek, polza_client=polza)

    request = _make_request(schemas.OpenAIModel.GPT4.value)
    official.request.return_value = "official"

    result = await service.request(request)

    assert result == "official"
    official.request.assert_awaited_once_with(request)
    deepseek.request.assert_not_called()
    polza.request.assert_not_called()


@pytest.mark.asyncio
async def test_legacy_request_is_adapted_through_compatible_flow() -> None:
    official = SimpleNamespace(request=AsyncMock())
    deepseek = SimpleNamespace(request=AsyncMock())
    polza = SimpleNamespace(request=AsyncMock())
    service = OpenAIService(official_client=official, deepseek_client=deepseek, polza_client=polza)

    legacy_request = _make_legacy_request(schemas.OpenAIModel.GPT4.value)
    completion = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(role="assistant", content="pong", tool_calls=None),
            ),
        ],
    )
    official.request.return_value = completion

    result = await service.request_legacy(legacy_request)

    assert result.messages[-1].content == "pong"
    official.request.assert_awaited_once()


def test_model_router_treats_none_as_auto() -> None:
    routes = ModelRouter().build_routes(None)

    assert [(route.provider, route.model, route.attempts) for route in routes] == [
        ("deepseek", schemas.OpenAIModel.DEEPSEEK.value, DEFAULT_ROUTE_ATTEMPTS),
        ("official", AUTO_OFFICIAL_MODEL, DEFAULT_ROUTE_ATTEMPTS),
        ("polza", AUTO_POLZA_MODEL, DEFAULT_ROUTE_ATTEMPTS),
    ]


def test_model_router_requires_model_after_prefix() -> None:
    with pytest.raises(ValueError, match="deepseek:"):
        ModelRouter().build_routes("deepseek:")
