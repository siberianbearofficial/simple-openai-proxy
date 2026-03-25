from functools import lru_cache

from openai import AsyncOpenAI

from openai_proxy import openai_compat
from openai_proxy.settings import (
    DeepseekOpenAISettings,
    OfficialOpenAISettings,
    OpenAISettings,
    PolzaOpenAISettings,
)


class OpenAIClient:
    def __init__(self, settings: OpenAISettings) -> None:
        self._default_model = settings.default_model
        self._client = AsyncOpenAI(
            api_key=settings.token,
            base_url=str(settings.base_url),
        )

    async def request(
        self,
        request: openai_compat.OpenAICompatibleRequest,
    ) -> openai_compat.OpenAICompatibleResponse:
        payload = openai_compat.normalize_chat_completion_request(request)
        if str(payload["model"]) == "auto":
            if self._default_model is None:
                err = "Provider does not define a default model for automatic routing"
                raise ValueError(err)

            payload = {**payload, "model": self._default_model}

        return await self._client.chat.completions.create(**payload)


class OfficialOpenAIClient(OpenAIClient):
    def __init__(self, settings: OfficialOpenAISettings) -> None:
        super().__init__(settings)


class DeepseekOpenAIClient(OpenAIClient):
    def __init__(self, settings: DeepseekOpenAISettings) -> None:
        super().__init__(settings)


class PolzaOpenAIClient(OpenAIClient):
    def __init__(self, settings: PolzaOpenAISettings) -> None:
        super().__init__(settings)


@lru_cache
def get_official_openai_client() -> OfficialOpenAIClient:
    return OfficialOpenAIClient(OfficialOpenAISettings())


@lru_cache
def get_deepseek_openai_client() -> DeepseekOpenAIClient:
    return DeepseekOpenAIClient(DeepseekOpenAISettings())


@lru_cache
def get_polza_openai_client() -> PolzaOpenAIClient:
    return PolzaOpenAIClient(PolzaOpenAISettings())
