from __future__ import annotations

from collections.abc import AsyncIterator, Iterable, Mapping
from typing import Any, Protocol, TypeAlias, cast, runtime_checkable

from openai.types.chat import ChatCompletion, ChatCompletionChunk, CompletionCreateParams
from openai.types.chat.completion_create_params import (
    CompletionCreateParamsNonStreaming,
    CompletionCreateParamsStreaming,
)
from pydantic import TypeAdapter

OpenAICompatibleRequest: TypeAlias = (
    CompletionCreateParamsNonStreaming | CompletionCreateParamsStreaming
)

_CHAT_COMPLETION_REQUEST_ADAPTER = TypeAdapter(CompletionCreateParams)
_NON_STREAMING_CHAT_COMPLETION_REQUEST_ADAPTER = TypeAdapter(
    CompletionCreateParamsNonStreaming,
)


@runtime_checkable
class ChatCompletionStreamResponse(Protocol):
    def __aiter__(self) -> AsyncIterator[ChatCompletionChunk]: ...

    async def close(self) -> None: ...


OpenAICompatibleResponse: TypeAlias = ChatCompletion | ChatCompletionStreamResponse


def normalize_chat_completion_request(
    request: CompletionCreateParams,
) -> OpenAICompatibleRequest:
    validated_request = _CHAT_COMPLETION_REQUEST_ADAPTER.validate_python(request)
    return cast(
        "OpenAICompatibleRequest",
        _materialize_json_compatible_value(validated_request),
    )


def normalize_non_streaming_chat_completion_request(
    request: CompletionCreateParamsNonStreaming,
) -> CompletionCreateParamsNonStreaming:
    validated_request = _NON_STREAMING_CHAT_COMPLETION_REQUEST_ADAPTER.validate_python(request)
    return cast(
        "CompletionCreateParamsNonStreaming",
        _materialize_json_compatible_value(validated_request),
    )


def is_streaming_chat_completion_request(request: OpenAICompatibleRequest) -> bool:
    return bool(request.get("stream"))


def is_streaming_chat_completion_response(response: OpenAICompatibleResponse) -> bool:
    return isinstance(response, ChatCompletionStreamResponse)


def _materialize_json_compatible_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            key: _materialize_json_compatible_value(item)
            for key, item in value.items()
            if item is not None
        }

    if isinstance(value, (str, bytes, bytearray)):
        return value

    if isinstance(value, Iterable):
        return [_materialize_json_compatible_value(item) for item in value]

    return value
