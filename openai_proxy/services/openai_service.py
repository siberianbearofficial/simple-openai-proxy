from functools import lru_cache
from typing import Annotated

from fastapi import Depends
from loguru import logger
from openai import OpenAIError

from openai_proxy import schemas
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

    async def request(self, req: schemas.OpenAIRequest) -> schemas.OpenAIResponse:
        """Выполняет запрос к LLM с учетом выбранной модели и провайдера."""

        model_value = req.model
        model_str = str(model_value)

        if model_str == "auto":
            try:
                return await self._deepseek.request(req)
            except OpenAIError as e:
                err = f"Unable to request deepseek, trying official: {e}"
                logger.error(err)
                return await self._official.request(req)

        if model_str.startswith("polza:"):
            _, _, pure_model = model_str.partition(":")
            if not pure_model:
                err = "Polza model name must follow the 'polza:' prefix"
                raise ValueError(err)

            polza_request = req.model_copy(update={"model": pure_model})
            return await self._polza.request(polza_request)

        if model_str in {
            schemas.OpenAIModel.DEEPSEEK.value,
            schemas.OpenAIModel.DEEPSEEK_FAST.value,
        }:
            return await self._deepseek.request(req)

        return await self._official.request(req)


@lru_cache
def get_openai_service() -> OpenAIService:
    return OpenAIService(
        official_client=get_official_openai_client(),
        deepseek_client=get_deepseek_openai_client(),
        polza_client=get_polza_openai_client(),
    )


OpenAIServiceDep = Annotated[OpenAIService, Depends(get_openai_service)]
