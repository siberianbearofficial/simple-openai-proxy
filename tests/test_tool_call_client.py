from types import SimpleNamespace
from typing import Annotated
from unittest.mock import AsyncMock

import pytest
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
)
from pydantic import BaseModel, Field

from openai_proxy import OpenAIProxyToolCallClient

EXPECTED_CHAT_COMPLETION_CALLS = 2


class EchoRequest(BaseModel):
    text: Annotated[str, Field(description="Text to echo")]


class EchoResponse(BaseModel):
    echoed: Annotated[str, Field(description="Echoed text")]


class EchoToolClient(OpenAIProxyToolCallClient):
    def __init__(self) -> None:
        super().__init__(system_prompts=["You are a helpful assistant"])

    @OpenAIProxyToolCallClient.tool("Echo text")
    async def echo(self, req: EchoRequest) -> EchoResponse:
        return EchoResponse(echoed=req.text)


def _make_tool_call() -> ChatCompletionMessageToolCall:
    return ChatCompletionMessageToolCall(
        id="call_1",
        type="function",
        function=Function(
            name="echo",
            arguments='{"text":"hi"}',
        ),
    )


def _make_completion(
    message: ChatCompletionMessage,
    finish_reason: str,
) -> ChatCompletion:
    return ChatCompletion(
        id="chatcmpl_1",
        choices=[Choice(finish_reason=finish_reason, index=0, message=message)],
        created=0,
        model="gpt-4.1",
        object="chat.completion",
    )


@pytest.mark.asyncio
async def test_call_tool_reads_name_and_arguments_from_function_payload(
    mocker,
) -> None:
    fake_openai = SimpleNamespace(close=AsyncMock())
    mocker.patch(
        "openai_proxy.tool_call_client.client.DefaultAsyncHttpxClient",
        return_value=object(),
    )
    mocker.patch(
        "openai_proxy.tool_call_client.client.AsyncOpenAI",
        return_value=fake_openai,
    )
    client = EchoToolClient()

    await client._call_tool(_make_tool_call())

    assert client._messages[-1] == {
        "role": "tool",
        "content": '{"echoed":"hi"}',
        "tool_call_id": "call_1",
    }
    await client.close()


@pytest.mark.asyncio
async def test_request_executes_tool_call_from_official_openai_response(
    mocker,
) -> None:
    tool_call = _make_tool_call()
    create = AsyncMock(
        side_effect=[
            _make_completion(
                ChatCompletionMessage(role="assistant", content=None, tool_calls=[tool_call]),
                "tool_calls",
            ),
            _make_completion(
                ChatCompletionMessage(role="assistant", content="done", tool_calls=None),
                "stop",
            ),
        ],
    )
    fake_openai = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
        close=AsyncMock(),
    )
    mocker.patch(
        "openai_proxy.tool_call_client.client.DefaultAsyncHttpxClient",
        return_value=object(),
    )
    mocker.patch(
        "openai_proxy.tool_call_client.client.AsyncOpenAI",
        return_value=fake_openai,
    )
    client = EchoToolClient()

    result = await client.request("ping")

    assert result == "done"
    assert create.await_count == EXPECTED_CHAT_COMPLETION_CALLS
    second_request_messages = create.await_args_list[1].kwargs["messages"]
    assistant_tool_message = next(
        message
        for message in second_request_messages
        if message["role"] == "assistant" and "tool_calls" in message
    )
    tool_response_message = next(
        message
        for message in second_request_messages
        if message["role"] == "tool"
    )
    assert assistant_tool_message["tool_calls"][0]["function"]["name"] == "echo"
    assert assistant_tool_message["tool_calls"][0]["function"]["arguments"] == '{"text":"hi"}'
    assert tool_response_message == {
        "role": "tool",
        "content": '{"echoed":"hi"}',
        "tool_call_id": "call_1",
    }
    await client.close()
