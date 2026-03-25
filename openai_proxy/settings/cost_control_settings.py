from __future__ import annotations

from pydantic import HttpUrl, SecretStr, model_validator
from pydantic_settings import SettingsConfigDict

from openai_proxy.settings.env_settings import EnvSettings


class PolzaCostControlSettings(EnvSettings):
    model_config = SettingsConfigDict(
        env_prefix="POLZA_COST_CONTROL__",
    )

    soft_threshold_rub: float | None = None
    hard_threshold_rub: float | None = None
    window_seconds: int = 3600
    logs_api_base_url: HttpUrl | None = None
    logs_api_username: str | None = None
    logs_api_password: SecretStr | None = None
    logs_api_timeout_seconds: float = 5.0
    application_name: str = "openai-proxy"
    notification_user: str = "anonymous"

    @property
    def soft_limit_enabled(self) -> bool:
        return self.soft_threshold_rub is not None

    @property
    def hard_limit_enabled(self) -> bool:
        return self.hard_threshold_rub is not None

    @property
    def any_limit_enabled(self) -> bool:
        return self.soft_limit_enabled or self.hard_limit_enabled

    @model_validator(mode="after")
    def validate_settings(self) -> "PolzaCostControlSettings":
        if self.window_seconds <= 0:
            err = "POLZA_COST_CONTROL__WINDOW_SECONDS must be greater than zero"
            raise ValueError(err)

        if self.soft_threshold_rub is not None and self.soft_threshold_rub <= 0:
            err = "POLZA_COST_CONTROL__SOFT_THRESHOLD_RUB must be greater than zero"
            raise ValueError(err)

        if self.hard_threshold_rub is not None and self.hard_threshold_rub <= 0:
            err = "POLZA_COST_CONTROL__HARD_THRESHOLD_RUB must be greater than zero"
            raise ValueError(err)

        if (
            self.soft_threshold_rub is not None
            and self.hard_threshold_rub is not None
            and self.hard_threshold_rub < self.soft_threshold_rub
        ):
            err = (
                "POLZA_COST_CONTROL__HARD_THRESHOLD_RUB must be greater than or equal "
                "to POLZA_COST_CONTROL__SOFT_THRESHOLD_RUB"
            )
            raise ValueError(err)

        if self.soft_limit_enabled:
            missing_fields = [
                field_name
                for field_name, value in (
                    ("POLZA_COST_CONTROL__LOGS_API_BASE_URL", self.logs_api_base_url),
                    ("POLZA_COST_CONTROL__LOGS_API_USERNAME", self.logs_api_username),
                    ("POLZA_COST_CONTROL__LOGS_API_PASSWORD", self.logs_api_password),
                )
                if value is None
            ]
            if missing_fields:
                err = (
                    "Soft polza cost limit requires logs API settings: "
                    + ", ".join(missing_fields)
                )
                raise ValueError(err)

        return self
