# OpenAI Proxy Library

[![status-badge](https://woodpecker.dev.nachert.art/api/badges/2/status.svg)](https://woodpecker.dev.nachert.art/repos/2)

The library exposes an asynchronous Python interface for calling OpenAI models through a
simple proxy. It includes helpers for function calling and for parsing structured output from
the model.

## Installation

```bash
pip install openai-async-functions
```

Or install from source:

```bash
git clone https://github.com/USER/simple-openai-proxy.git
cd simple-openai-proxy
pip install .
```

## Function calling with tools

Function calling is based on asynchronous methods that accept and return `pydantic` models.
Use `OpenAIProxyToolCallClient.tool()` decorator to expose a method as a tool. The model will
call tools on its own and insert results back into the conversation.

```python
import asyncio
from typing import Annotated

from pydantic import BaseModel, Field

from openai_proxy import OpenAIProxyToolCallClient, OpenAIProxyClientSettings


class WeatherRequest(BaseModel):
    city: Annotated[str, Field(description="City name")]


class WeatherResponse(BaseModel):
    forecast: Annotated[str, Field(description="Weather forecast")]


class WeatherClient(OpenAIProxyToolCallClient):
    def __init__(self) -> None:
        super().__init__(
            system_prompts=["You are a helpful assistant"],
            openai_proxy_client_settings=OpenAIProxyClientSettings(verify_ssl=False),
        )

    @OpenAIProxyToolCallClient.tool("Get weather forecast")
    async def get_weather(self, req: WeatherRequest) -> WeatherResponse:
        return WeatherResponse(forecast=f"It's always sunny in {req.city}")


async def main() -> None:
    client = WeatherClient()
    answer = await client.request("What's the weather in London?")
    print(answer)


if __name__ == "__main__":
    asyncio.run(main())
```

### Registering tools at runtime

You can also expose methods without decorators as tools by marking them at runtime:

```python
from typing import Annotated
from pydantic import BaseModel, Field
from openai_proxy import OpenAIProxyToolCallClient


class AddRequest(BaseModel):
    a: Annotated[int, Field(description="First addend")]
    b: Annotated[int, Field(description="Second addend")]


class AddResponse(BaseModel):
    result: Annotated[int, Field(description="Sum")]


class MathClient(OpenAIProxyToolCallClient):
    def __init__(self) -> None:
        super().__init__(system_prompts=["You are a math assistant"])

    async def add(self, req: AddRequest) -> AddResponse:
        return AddResponse(result=req.a + req.b)


client = MathClient()
OpenAIProxyToolCallClient.mark_tool_methods(client, {"add": "Add two numbers"})
```

## Parsing JSON blocks

`CodeBlocksParser` extracts JSON returned by the model inside Markdown code blocks:

```python
from pydantic import BaseModel
from openai_proxy import CodeBlocksParser


class Data(BaseModel):
    value: int


doc = """
Here is the answer:
```json
{"value": 1}
```
"""

parser = CodeBlocksParser(doc)
block = parser.find_json_block()
data = Data.model_validate_json(block)
```

## Low-level client

`OpenAIProxyClient` sends arbitrary `OpenAIRequest` objects:

```python
import asyncio
from openai_proxy import OpenAIProxyClient, OpenAIProxyClientSettings, schemas


async def main() -> None:
    client = OpenAIProxyClient(OpenAIProxyClientSettings())
    request = schemas.OpenAIRequest(
        model="auto",
        messages=[schemas.OpenAIMessage(role=schemas.OpenAIRole.USER, content="ping")],
    )
    response = await client.request(request)
    print(response)


if __name__ == "__main__":
    asyncio.run(main())
```

`OpenAIProxyClientSettings` lets you change the base URL and SSL verification parameters.

## Tests

```bash
pytest
```

Distributed under the MIT license.
