from functools import lru_cache
from typing import Annotated

from fastapi import Depends
from openai import OpenAIError

from openai_proxy import schemas
from openai_proxy.client import (
    DeepseekOpenAIClient,
    OfficialOpenAIClient,
    get_deepseek_openai_client,
    get_official_openai_client,
)
from openai_proxy.logger import measure, logger


class OpenAIService:
    def __init__(
        self,
        official_client: OfficialOpenAIClient,
        deepseek_client: DeepseekOpenAIClient,
    ) -> None:
        self._official = official_client
        self._deepseek = deepseek_client

    @measure
    async def request(self, req: schemas.OpenAIRequest) -> schemas.OpenAIResponse:
        """Временная реализация с походом в deepseek, а при ошибке в гпт."""
        try:
            return await self._deepseek.request(req)
        except OpenAIError as e:
            err = f"Unable to request deepseek, trying official: {e}"
            logger.error(err)
            return await self._official.request(req)


@lru_cache
def get_openai_service() -> OpenAIService:
    return OpenAIService(
        official_client=get_official_openai_client(),
        deepseek_client=get_deepseek_openai_client(),
    )


OpenAIServiceDep = Annotated[OpenAIService, Depends(get_openai_service)]
