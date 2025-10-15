# pylint: disable=use-dict-literal
from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal, Optional

from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessage,
    ChatCompletionMessageToolCall,
    ChatCompletionToolParam,
)
from openai.types.chat.chat_completion_message_tool_call import Function
from openai.types.shared_params import FunctionDefinition
from pydantic import BaseModel, Field, fields

from openai_proxy import helpers


class OpenAIModel(StrEnum):
    TURBO = "gpt-3.5-turbo"
    GPT4 = "gpt-4.1"
    GPT4_TURBO = "gpt-4-turbo-preview"
    GPT_4O = "gpt-4o-2024-11-20"
    DEEPSEEK = "deepseek-chat"
    DEEPSEEK_FAST = "deepseek-chat-fast"


class OpenAIRole(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSIST = "assistant"
    TOOL = "tool"


class OpenAIToolParameter(BaseModel):
    name: Annotated[str, Field(description="Имя аргумента", examples=["file_path"])]
    type: Annotated[str, Field(description="Тип аргумента", examples=["string"])]
    format: Annotated[
        str,
        Field(
            description="В каком формате представлено значение",
            examples=["string", "datetime", "uuid"],
        ),
    ]
    description: Annotated[
        str,
        Field(
            description="Человекочитаемое описание аргумента для LLM",
            examples=["Relative path from current directory"],
        ),
    ]
    required: Annotated[
        bool,
        Field(
            description="Является ли передача аргумента обязательной",
            examples=[True],
        ),
    ]

    def to_gpt(self) -> dict[str, object]:
        return self.model_dump(exclude={"name", "required"})

    @classmethod
    def from_pydantic_field(
        cls,
        field_name: str,
        field_info: fields.FieldInfo,
    ) -> OpenAIToolParameter:
        if field_info.annotation is None:
            err = "Only type annotated fields are supported."
            raise TypeError(err)

        if field_info.description is None:
            err = "Field description is required."
            raise TypeError(err)

        base_type, is_optional = helpers.parse_annotation(field_info.annotation)

        return cls(
            name=field_name,
            description=field_info.description,
            type=helpers.get_openapi_type(base_type),
            format=helpers.get_openapi_format(base_type),
            required=not is_optional,
        )


class OpenAITool(BaseModel):
    name: Annotated[str, Field(description="Имя функции", examples=["get_file"])]
    description: Annotated[
        str,
        Field(description="Описание функции", examples=["Read file content"]),
    ]
    parameters: Annotated[
        list[OpenAIToolParameter],
        Field(description="Аргументы функции"),
    ]

    def to_gpt(self) -> ChatCompletionToolParam:
        properties = {p.name: p.to_gpt() for p in self.parameters}
        required = [p.name for p in self.parameters if p.required]

        return ChatCompletionToolParam(
            type="function",
            function=FunctionDefinition(
                name=self.name,
                description=self.description,
                parameters=dict(  # noqa
                    type="object",
                    properties=properties,
                    required=required,
                ),
            ),
        )


class OpenAIMessage(BaseModel):
    role: Annotated[
        OpenAIRole,
        Field(description="Роль отправителя сообщения", examples=["user"]),
    ]
    content: Annotated[
        Optional[str],
        Field(
            description="Содержимое сообщения",
            examples=["Есть ли слово котик в файле exxxtreme.txt?"],
        ),
    ] = None
    tool_call_id: Annotated[
        Optional[str],
        Field(
            description="ID вызова тула для отправки результата работы функции",
            examples=["call_0_2428f7f3-4857-4c6d-b4fe-877a3bdb04e5"],
        ),
    ] = None
    tool_calls: Annotated[
        Optional[list[OpenAIToolCall]],
        Field(description="Список всех осуществленных вызовов тулов в этом диалоге"),
    ] = None

    @classmethod
    def from_gpt(cls, gpt: ChatCompletionMessage) -> OpenAIMessage:
        return cls(
            role=gpt.role,
            content=gpt.content,
            tool_calls=(
                [OpenAIToolCall.from_gpt(c) for c in gpt.tool_calls] if gpt.tool_calls else None
            ),
        )

    def to_gpt(self) -> dict:  # type: ignore
        calls = self.tool_calls or []
        return dict(  # noqa
            role=self.role,
            content=self.content or None,
            tool_call_id=self.tool_call_id or None,
            tool_calls=[t.to_gpt() for t in calls] or None,
        )


class OpenAIToolCall(BaseModel):
    id: Annotated[
        str,
        Field(
            description="ID вызова",
            examples=["call_0_2428f7f3-4857-4c6d-b4fe-877a3bdb04e5"],
        ),
    ]
    name: Annotated[str, Field(description="Имя функции", examples=["get_file"])]
    arguments: Annotated[
        str,
        Field(
            description="Аргументы функции в формате json-строки",
            examples=['{"file_path":"exxxtreme.txt"}'],
        ),
    ]

    @classmethod
    def from_gpt(cls, resp: ChatCompletionMessageToolCall) -> OpenAIToolCall:
        return cls(
            id=resp.id,
            name=resp.function.name,
            arguments=resp.function.arguments,
        )

    def to_gpt(self) -> ChatCompletionMessageToolCall:
        return ChatCompletionMessageToolCall(
            id=self.id,
            type="function",
            function=Function(
                name=self.name,
                arguments=self.arguments,
            ),
        )


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[dict]  # type: ignore
    tools: list[ChatCompletionToolParam] | None = None
    tool_choice: Literal["none", "auto", "required"] | None = None


class OpenAIRequest(BaseModel):
    model: Annotated[
        OpenAIModel | Literal["auto"] | str,
        Field(description="Модель гпт, которую хочется использовать"),
    ]
    messages: Annotated[list[OpenAIMessage], Field(description="Список сообщений")]
    tools: Annotated[
        list[OpenAITool],
        Field(description="Список доступных тулов", default_factory=list),
    ]
    tool_choice: Annotated[
        Literal["none", "auto", "required"],
        Field(description="Обязательно ли вызывать тул"),
    ] = "auto"

    def to_gpt(self) -> ChatCompletionRequest:
        model: str | OpenAIModel
        if self.model == "auto":
            model = OpenAIModel.DEEPSEEK.value
        elif isinstance(self.model, OpenAIModel):
            model = self.model.value
        else:
            try:
                model = OpenAIModel(str(self.model)).value
            except ValueError:
                model = str(self.model)
        tools = [f.to_gpt() for f in self.tools]
        tool_choice: Literal["none", "auto", "required"] | None
        if tools:
            tool_choice = self.tool_choice
        elif self.tool_choice == "auto":
            tool_choice = None
        else:
            tool_choice = self.tool_choice

        return ChatCompletionRequest(
            model=model,
            messages=[m.to_gpt() for m in self.messages],
            tools=tools or None,
            tool_choice=tool_choice,
        )


class OpenAIResponse(BaseModel):
    messages: list[OpenAIMessage]

    @classmethod
    def from_gpt(
        cls,
        request: OpenAIRequest,
        response: ChatCompletion,
    ) -> OpenAIResponse:
        message = OpenAIMessage.from_gpt(response.choices[0].message)
        return cls(messages=[*request.messages, message])
