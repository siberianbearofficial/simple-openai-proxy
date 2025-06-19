from pydantic import HttpUrl
from pydantic_settings import BaseSettings


class OpenAIProxyClientSettings(BaseSettings):
    """
    Settings for proxy client.
    """

    base_url: HttpUrl = HttpUrl("https://simple-openai-proxy.nachert.art")
    verify_ssl: bool = True
