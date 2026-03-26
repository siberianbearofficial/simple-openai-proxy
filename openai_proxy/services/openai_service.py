from functools import lru_cache
from typing import Annotated

from fastapi import Depends
from loguru import logger
from openai import OpenAIError

from openai_proxy import openai_compat, schemas
from openai_proxy.client import (
    DeepseekOpenAIClient,
    OfficialOpenAIClient,
    PolzaOpenAIClient,
    get_deepseek_openai_client,
    get_official_openai_client,
    get_polza_openai_client,
)
from openai_proxy.services.model_routing import ModelRouter, ProviderName
from openai_proxy.services.polza_cost_control import (
    PolzaCostControl,
    get_polza_cost_control,
)


class OpenAIService:
    def __init__(
        self,
        official_client: OfficialOpenAIClient,
        deepseek_client: DeepseekOpenAIClient,
        polza_client: PolzaOpenAIClient,
        model_router: ModelRouter | None = None,
        polza_cost_control: PolzaCostControl | None = None,
    ) -> None:
        self._official = official_client
        self._deepseek = deepseek_client
        self._polza = polza_client
        self._model_router = model_router or ModelRouter()
        self._polza_cost_control = polza_cost_control

    async def request(
        self,
        req: openai_compat.OpenAICompatibleRequest,
    ) -> openai_compat.OpenAICompatibleResponse:
        """Routes an OpenAI-compatible chat completion request to the right provider."""

        req = openai_compat.normalize_chat_completion_request(req)
        last_error: OpenAIError | None = None

        for route in self._model_router.build_routes(req.get("model")):
            routed_request = route.apply_to(req)
            client = self._get_client(route.provider)

            for attempt in range(1, route.attempts + 1):
                try:
                    if self._polza_cost_control is not None:
                        await self._polza_cost_control.check_hard_limit(route.provider)

                    response = await client.request(routed_request)
                    if self._polza_cost_control is not None:
                        if openai_compat.is_streaming_chat_completion_response(response):
                            response = self._polza_cost_control.wrap_stream(
                                provider=route.provider,
                                response=response,
                                request=routed_request,
                            )
                        else:
                            await self._polza_cost_control.record_response_cost(
                                provider=route.provider,
                                response=response,
                                request=routed_request,
                            )
                except OpenAIError as ex:
                    last_error = ex
                    if attempt < route.attempts:
                        logger.warning(
                            f"Unable to request {route.provider} model {route.model} "
                            f"(attempt {attempt}/{route.attempts}), retrying: {ex}",
                        )
                        continue

                    logger.error(
                        f"Unable to request {route.provider} model {route.model}, "
                        f"trying next route if available: {ex}",
                    )
                    break
                else:
                    return response

        if last_error is not None:
            raise last_error

        err = "Unable to build a model route"
        raise RuntimeError(err)

    async def request_legacy(self, req: schemas.OpenAIRequest) -> schemas.OpenAIResponse:
        response = await self.request(req.to_chat_completion_params())
        if openai_compat.is_streaming_chat_completion_response(response):
            err = "Legacy endpoint does not support streaming responses"
            raise TypeError(err)

        return schemas.OpenAIResponse.from_gpt(req, response)

    def _get_client(
        self,
        provider: ProviderName,
    ) -> OfficialOpenAIClient | DeepseekOpenAIClient | PolzaOpenAIClient:
        if provider == "deepseek":
            return self._deepseek
        if provider == "polza":
            return self._polza
        return self._official


@lru_cache
def get_openai_service() -> OpenAIService:
    return OpenAIService(
        official_client=get_official_openai_client(),
        deepseek_client=get_deepseek_openai_client(),
        polza_client=get_polza_openai_client(),
        polza_cost_control=get_polza_cost_control(),
    )


OpenAIServiceDep = Annotated[OpenAIService, Depends(get_openai_service)]
