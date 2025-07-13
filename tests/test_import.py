def test_import() -> None:
    import openai_proxy  # pylint: disable=import-outside-toplevel

    assert openai_proxy
    assert openai_proxy.OpenAIProxyClient
    assert openai_proxy.OpenAIProxyToolCallClient
    assert openai_proxy.OpenAIProxyClientSettings
    assert openai_proxy.CodeBlocksParser
