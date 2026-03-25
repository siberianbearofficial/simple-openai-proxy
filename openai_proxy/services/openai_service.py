from functools import lru_cache
from typing import Annotated

from fastapi import Depends
from loguru import logger
from openai import AsyncStream, OpenAIError

from openai_proxy import openai_compat, schemas
from openai_proxy.client import (
    DeepseekOpenAIClient,
    OfficialOpenAIClient,
    PolzaOpenAIClient,
    get_deepseek_openai_client,
    get_official_openai_client,
    get_polza_openai_client,
)


class OpenAIService:
    def __init__(
        self,
        official_client: OfficialOpenAIClient,
        deepseek_client: DeepseekOpenAIClient,
        polza_client: PolzaOpenAIClient,
    ) -> None:
        self._official = official_client
        self._deepseek = deepseek_client
        self._polza = polza_client

    async def request(
        self,
        req: openai_compat.OpenAICompatibleRequest,
    ) -> openai_compat.OpenAICompatibleResponse:
        """Routes an OpenAI-compatible chat completion request to the right provider."""

        req = openai_compat.normalize_chat_completion_request(req)
        model_str = str(req["model"])

        if model_str == "auto":
            try:
                return await self._deepseek.request(req)
            except OpenAIError as ex:
                logger.error(f"Unable to request DeepSeek, trying official provider: {ex}")
                return await self._official.request(req)

        if model_str.startswith("polza:"):
            _, _, pure_model = model_str.partition(":")
            if not pure_model:
                err = "Polza model name must follow the 'polza:' prefix"
                raise ValueError(err)

            polza_request = {**req, "model": pure_model}
            return await self._polza.request(polza_request)

        if model_str in {
            schemas.OpenAIModel.DEEPSEEK.value,
            schemas.OpenAIModel.DEEPSEEK_FAST.value,
        }:
            return await self._deepseek.request(req)

        return await self._official.request(req)

    async def request_legacy(self, req: schemas.OpenAIRequest) -> schemas.OpenAIResponse:
        response = await self.request(req.to_chat_completion_params())
        if isinstance(response, AsyncStream):
            err = "Legacy endpoint does not support streaming responses"
            raise TypeError(err)

        return schemas.OpenAIResponse.from_gpt(req, response)


@lru_cache
def get_openai_service() -> OpenAIService:
    return OpenAIService(
        official_client=get_official_openai_client(),
        deepseek_client=get_deepseek_openai_client(),
        polza_client=get_polza_openai_client(),
    )


OpenAIServiceDep = Annotated[OpenAIService, Depends(get_openai_service)]
