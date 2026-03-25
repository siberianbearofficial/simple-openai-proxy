from collections.abc import AsyncIterator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from openai import AsyncStream
from openai.types.chat import ChatCompletion, ChatCompletionChunk, CompletionCreateParams

from openai_proxy import openai_compat, schemas, services

openai_router = APIRouter()


async def _stream_chat_completion(
    stream: AsyncStream[ChatCompletionChunk],
) -> AsyncIterator[str]:
    try:
        async for chunk in stream:
            yield f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"
        yield "data: [DONE]\n\n"
    finally:
        await stream.close()


@openai_router.post(
    "/v1/chat/completions",
    summary="OpenAI-compatible chat completions endpoint",
    description=(
        "Primary endpoint. Accepts the official OpenAI Chat Completions payload and "
        "can be used with the official OpenAI SDK by pointing `base_url` to this proxy."
    ),
    tags=["openai"],
    response_model=ChatCompletion,
)
async def chat_completions_handler(
    openai_service: services.OpenAIServiceDep,
    request: CompletionCreateParams,
) -> ChatCompletion | StreamingResponse:
    normalized_request = openai_compat.normalize_chat_completion_request(request)
    response = await openai_service.request(normalized_request)
    if isinstance(response, AsyncStream):
        return StreamingResponse(
            _stream_chat_completion(response),
            media_type="text/event-stream",
        )

    return response


@openai_router.post(
    "/api/v1/openai/request",
    summary="Deprecated legacy endpoint with simplified schemas",
    description=(
        "Deprecated compatibility layer. It accepts the legacy simplified request schema "
        "and internally adapts it to the OpenAI-compatible `/v1/chat/completions` endpoint."
    ),
    tags=["openai"],
    response_model=schemas.OpenAIResponse,
    deprecated=True,
)
async def legacy_request_handler(
    openai_service: services.OpenAIServiceDep,
    request: schemas.OpenAIRequest,
) -> schemas.OpenAIResponse:
    return await openai_service.request_legacy(request)
