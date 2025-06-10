from fastapi import APIRouter

from openai_proxy import schemas, services

openai_router = APIRouter()


@openai_router.post(
    "/api/v1/openai/request",
    summary="Request anything from ChatGPT, from plain messages to function calling",
    description=(
        "Будут ретраи, автоматический выбор доступной модели, "
        "метрики и многое другое. Возможно. Когда-нибудь."
    ),
    tags=["openai"],
    response_model=schemas.OpenAIResponse,
)
async def request_handler(
    openai_service: services.OpenAIServiceDep,
    request: schemas.OpenAIRequest,
) -> schemas.OpenAIResponse:
    return await openai_service.request(request)
