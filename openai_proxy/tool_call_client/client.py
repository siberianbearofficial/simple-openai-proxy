import inspect
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Optional, get_type_hints

from loguru import logger
from pydantic import BaseModel

from openai_proxy import schemas
from openai_proxy.helpers import ensure_prompts
from openai_proxy.proxy_client import OpenAIProxyClient
from openai_proxy.settings import OpenAIProxyClientSettings
from openai_proxy.tool_call_client.models import ClientTool, ClientToolInfo

INFO_ATTR = "_tool_info"


def client_tool_decorator(func: Callable, description: str):
    sig = inspect.signature(func)
    # отфильтровываем self/cls
    params = [p for p in sig.parameters.values() if p.name not in {"self", "cls"}]
    if len(params) > 1:
        err = (
            f"Decorated method {func.__name__} should have only one argument, "
            f"but {func.__name__} has {len(params)}"
        )
        raise ValueError(err)

    # вытягиваем тип через get_type_hints
    hints = get_type_hints(func)
    param_type: Optional[Any] = None
    if params:
        param_type = hints.get(params[0].name, None)
    if param_type is None:
        err = f"Method {func.__name__} should provide valid type annotation for argument"
        raise TypeError(err)
    if not issubclass(param_type, BaseModel):
        err = f"Method {func.__name__} should inherit from BaseModel"
        raise TypeError(err)

    # парсим модель пидантика
    tool_parameters: list[schemas.OpenAIToolParameter] = []
    for field_name, field_info in param_type.model_fields.items():
        tool_parameters.append(
            schemas.OpenAIToolParameter.from_pydantic_field(field_name, field_info),
        )

    tool_info = ClientToolInfo(
        name=func.__name__,
        parameters=tool_parameters,
        description=description,
        param_type=param_type,
    )

    # сохраняем информацию о туле прямо в метод
    setattr(func, INFO_ATTR, tool_info)

    @wraps(func)
    async def wrapper(self, *args, **kwargs):
        return await func(self, *args, **kwargs)

    return wrapper


class OpenAIProxyToolCallClient:
    """
    Client for OpenAI proxy tool calls.
    Use OpenAIProxyToolCallClient.tool decorator to mark methods as tools.
    Or mark them explicitly with OpenAIProxyToolCallClient.mark_tool_methods.
    Note that only async methods with type annotated arguments are supported.
    """

    def __init__(
        self,
        system_prompts: Optional[list[str]] = None,
        system_prompt_paths: Optional[list[Path]] = None,
        openai_proxy_client_settings: Optional[OpenAIProxyClientSettings] = None,
        tools: Optional[list[ClientTool]] = None,
    ) -> None:
        system_prompts = ensure_prompts(system_prompts, system_prompt_paths)

        self._client = OpenAIProxyClient(openai_proxy_client_settings)
        self._messages = [
            schemas.OpenAIMessage(
                role=schemas.OpenAIRole.SYSTEM,
                content=prompt,
            )
            for prompt in system_prompts
        ]
        self._tools: list[ClientTool] = self.collect_tools(self) if tools is None else tools

    async def request(self, user_prompt: str) -> str:
        """
        Start requesting OpenAI proxy with a given user_prompt.
        :param user_prompt: any prompt from user.
        :return: gpt answer.
        """
        self._messages.append(
            schemas.OpenAIMessage(
                role=schemas.OpenAIRole.USER,
                content=user_prompt,
            ),
        )

        answer_parts: list[str] = []

        while True:
            answer = await self._request_openai()
            logger.debug(f"OpenAI answer: {answer.model_dump_json()}")

            if answer.content:
                answer_parts.append(answer.content)

            if answer.tool_calls:
                for tool_call in answer.tool_calls:
                    await self._call_tool(tool_call, answer.tool_calls)
            else:
                break

        return "\n".join(answer_parts)

    async def _request_openai(self) -> schemas.OpenAIMessage:
        req = schemas.OpenAIRequest(
            model="auto",
            messages=self._messages,
            tools=self._tools,
        )
        resp = await self._client.request(req)
        self._messages = resp.messages
        return resp.messages[-1]

    async def _call_tool(
        self,
        tool_call: schemas.OpenAIToolCall,
        prev_tool_calls: list[schemas.OpenAIToolCall],
    ) -> None:
        logger.debug("OpenAI wants tool call")
        tool = self._find_tool_by_name(tool_call.name)
        logger.debug(f"Tool found: {tool.name}")
        req = tool.param_type.model_validate_json(tool_call.arguments)
        logger.debug(f"Input: {req.model_dump_json()}")
        resp = await tool.python_method(req)
        logger.debug(f"Output: {resp.model_dump_json()}")
        self._messages.append(
            schemas.OpenAIMessage(
                role=schemas.OpenAIRole.TOOL,
                content=resp.model_dump_json(),
                tool_calls=prev_tool_calls,
                tool_call_id=tool_call.id,
            ),
        )

    def _find_tool_by_name(self, name: str) -> ClientTool:
        for tool in self._tools:
            if tool.name == name:
                return tool

        err = f"Tool {name} is not implemented"
        raise NotImplementedError(err)

    @staticmethod
    def tool(description: str):
        """
        Декоратор, который:
        1) проверяет, что метод принимает не более одного аргумента (кроме self/cls),
        2) достаёт у него аннотацию параметра (если есть),
        3) сохраняет информацию о туле прямо в метод (func),
        4) возвращает обёртку, прокси для самого метода.
        """

        def decorator(func: Callable):
            return client_tool_decorator(func, description)

        return decorator

    @staticmethod
    def collect_tools(obj: object) -> list[ClientTool]:
        """
        Собирает все доступные ClientTool, добавленные через
        декоратор tool и mark_tool_methods.
        :param obj: Объект, с которого собрать тулы.
        :return:
        """
        tools: list[ClientTool] = []
        for attr_name in dir(obj):
            method = getattr(obj, attr_name)
            tool_info: Optional[ClientToolInfo] = getattr(method, INFO_ATTR, None)
            if tool_info is not None:
                tool = ClientTool(
                    name=tool_info.name,
                    description=tool_info.description,
                    parameters=tool_info.parameters,
                    python_method=method,
                    param_type=tool_info.param_type,
                )
                tools.append(tool)
                logger.debug(f"Tool registered: {tool.model_dump()}")

        return tools

    @staticmethod
    def mark_tool_methods(obj: object, descriptions: dict[str, str]) -> None:
        """
        Отмечает методы из словаря description как ClientTool.
        :param obj: Объект, в котором отметить тулы.
        :param descriptions: Словарь, где ключ - имя метода, значение - описание метода.
        :return:
        """
        cls = obj.__class__
        for method_name, method_description in descriptions.items():
            if hasattr(cls, method_name):
                method = getattr(cls, method_name)
                client_tool_decorator(method, method_description)

        # если тулы в классе клиента, нужно обновить список
        if isinstance(obj, OpenAIProxyToolCallClient):
            obj._tools = obj.collect_tools(obj)
