from typing import Awaitable, Callable, Type

from pydantic import BaseModel, ConfigDict

from openai_proxy import schemas


class ClientToolInfo(schemas.OpenAITool):
    model_config = ConfigDict(from_attributes=True)

    param_type: Type[BaseModel]


class ClientTool(ClientToolInfo):
    python_method: Callable[[BaseModel], Awaitable[BaseModel]]
