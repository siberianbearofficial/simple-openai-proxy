from pydantic import HttpUrl
from pydantic_settings import SettingsConfigDict

from openai_proxy.settings.env_settings import EnvSettings


class OpenAISettings(EnvSettings):
    token: str
    base_url: HttpUrl = HttpUrl("https://api.openai.com/v1")
    max_message_size: int = 100000


class OfficialOpenAISettings(OpenAISettings):
    model_config = SettingsConfigDict(
        env_prefix="OFFICIAL_OPENAI__",
    )

    base_url: HttpUrl = HttpUrl("https://api.openai.com/v1")


class DeepseekOpenAISettings(OpenAISettings):
    model_config = SettingsConfigDict(
        env_prefix="DEEPSEEK_OPENAI__",
    )

    base_url: HttpUrl = HttpUrl("https://api.deepseek.com/v1")
