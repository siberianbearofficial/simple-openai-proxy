from functools import lru_cache

from openai import AsyncOpenAI

from openai_proxy import schemas
from openai_proxy.settings import (
    DeepseekOpenAISettings,
    OfficialOpenAISettings,
    OpenAISettings,
)


class OpenAIClient:
    def __init__(self, settings: OpenAISettings) -> None:
        self._api_key = settings.token
        self._client = AsyncOpenAI(
            api_key=self._api_key,
            base_url=str(settings.base_url),
        )

    async def request(self, request: schemas.OpenAIRequest) -> schemas.OpenAIResponse:
        response = await self._client.chat.completions.create(
            **request.to_gpt().model_dump(),
        )
        return schemas.OpenAIResponse.from_gpt(request, response)


class OfficialOpenAIClient(OpenAIClient):
    def __init__(self, settings: OfficialOpenAISettings) -> None:
        super().__init__(settings)


class DeepseekOpenAIClient(OpenAIClient):
    def __init__(self, settings: DeepseekOpenAISettings) -> None:
        super().__init__(settings)


@lru_cache
def get_official_openai_client() -> OfficialOpenAIClient:
    return OfficialOpenAIClient(OfficialOpenAISettings())


@lru_cache
def get_deepseek_openai_client() -> DeepseekOpenAIClient:
    return DeepseekOpenAIClient(DeepseekOpenAISettings())
