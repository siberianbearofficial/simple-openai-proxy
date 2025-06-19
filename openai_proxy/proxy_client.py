from __future__ import annotations

import typing

import aiohttp

from openai_proxy import schemas
from openai_proxy.settings import OpenAIProxyClientSettings


class OpenAIProxyClient:
    def __init__(self, settings: typing.Optional[OpenAIProxyClientSettings] = None) -> None:
        self._settings: OpenAIProxyClientSettings = settings or OpenAIProxyClientSettings()

    def __get_session(self) -> aiohttp.ClientSession:
        return aiohttp.ClientSession(base_url=str(self._settings.base_url), raise_for_status=True)

    async def request(self, request: schemas.OpenAIRequest) -> schemas.OpenAIResponse:
        async with (
            self.__get_session() as session,
            session.post(
                url="/api/v1/openai/request",
                json=request.model_dump(),
                ssl=self._settings.verify_ssl,
            ) as response,
        ):
            response_json = await response.json()
            return schemas.OpenAIResponse.model_validate(response_json)
