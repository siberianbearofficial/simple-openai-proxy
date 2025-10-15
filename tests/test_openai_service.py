from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from openai import OpenAIError

from openai_proxy import schemas
from openai_proxy.services.openai_service import OpenAIService


def _make_request(model: str | schemas.OpenAIModel) -> schemas.OpenAIRequest:
    return schemas.OpenAIRequest(
        model=model,
        messages=[
            schemas.OpenAIMessage(
                role=schemas.OpenAIRole.USER,
                content="ping",
            )
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
    deepseek.request.assert_awaited_once_with(request)
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
    deepseek.request.assert_awaited_once_with(request)
    official.request.assert_awaited_once_with(request)
    polza.request.assert_not_called()


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
    polza.request.assert_awaited_once()
    polza_request = polza.request.call_args.args[0]
    assert polza_request.model == "chat-1"


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
