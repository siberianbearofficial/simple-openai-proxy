from pydantic import HttpUrl
from pydantic_settings import BaseSettings


class OpenAIProxyClientSettings(BaseSettings):
    """
    Settings for proxy client.
    """

    base_url: HttpUrl = HttpUrl("https://simple-openai-proxy.nachert.art")
    api_key: str = "proxy"
    verify_ssl: bool = True

    @property
    def openai_base_url(self) -> str:
        return f"{str(self.base_url).rstrip('/')}/v1"
