from typing import Any, Awaitable, Callable, Type

from pydantic import BaseModel, ConfigDict


class ClientToolInfo(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    description: str
    tool_schema: dict[str, Any]
    param_type: Type[BaseModel]


class ClientTool(ClientToolInfo):
    python_method: Callable[[BaseModel], Awaitable[BaseModel]]
