from typing import Annotated, Optional

from pydantic import BaseModel, Field

from openai_proxy import OpenAIProxyToolCallClient
from openai_proxy.settings import OpenAIProxyClientSettings


class WhatToDoRequest(BaseModel):
    situation: Annotated[str, Field(description="Situation description to decide what to do")]
    do_not_use_this_field: str = Field(description="This field is not used")


class WhatToDoResponse(BaseModel):
    suggestion: Annotated[str, Field(description="Suggestion to make a decision")]


class AdviceRequest(BaseModel):
    situation: Annotated[str, Field(description="Situation description to give advice")]


class AdviceResponse(BaseModel):
    advice: Annotated[str, Field(description="Valuable advice for user in given situation")]


class ConcreteClient(OpenAIProxyToolCallClient):
    def __init__(
        self,
        openai_proxy_client_settings: Optional[OpenAIProxyClientSettings] = None,
    ) -> None:
        super().__init__(
            system_prompts=[
                (
                    "You are a chatbot that should help users "
                    "to make decisions in various situations. "
                    "Also it is necessary to give advice after "
                    "each decision. Trust tool responses."
                ),
            ],
            openai_proxy_client_settings=openai_proxy_client_settings,
        )
        self.activity = "extreme sports"

    @OpenAIProxyToolCallClient.tool(
        description="Get a suggestion about what to do in the given situation.",
    )
    async def what_to_do(self, req: WhatToDoRequest) -> WhatToDoResponse:
        if "bored" in req.situation:
            return WhatToDoResponse(suggestion=f"Do some {self.activity}!")
        if "hungry" in req.situation:
            return WhatToDoResponse(suggestion="Order a Pizza")
        return WhatToDoResponse(suggestion="I can't help you with this situation")

    async def advice(self, req: AdviceRequest) -> AdviceResponse:  # noqa
        return AdviceResponse(advice="Today's advice: never ask silly questions.")


async def main() -> None:
    client = ConcreteClient(OpenAIProxyClientSettings(verify_ssl=False))
    OpenAIProxyToolCallClient.mark_tool_methods(client, {"advice": "Get daily advice"})

    user_prompt = input("Your message: ")
    res = await client.request(user_prompt)
    print(res)  # noqa


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
