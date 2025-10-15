from openai_proxy import schemas


def _base_message() -> schemas.OpenAIMessage:
    return schemas.OpenAIMessage(role=schemas.OpenAIRole.USER, content="ping")


def test_to_gpt_omits_tools_when_not_provided() -> None:
    request = schemas.OpenAIRequest(model="auto", messages=[_base_message()])

    payload = request.to_gpt().model_dump(exclude_none=True)

    assert "tools" not in payload
    assert "tool_choice" not in payload


def test_to_gpt_includes_tools_when_present() -> None:
    request = schemas.OpenAIRequest(
        model="auto",
        messages=[_base_message()],
        tools=[
            schemas.OpenAITool(
                name="echo",
                description="Echo text",
                parameters=[
                    schemas.OpenAIToolParameter(
                        name="text",
                        type="string",
                        format="string",
                        description="Text to echo",
                        required=True,
                    )
                ],
            )
        ],
    )

    payload = request.to_gpt().model_dump(exclude_none=True)

    assert "tools" in payload
    assert payload["tools"]
    assert payload.get("tool_choice") == "auto"
