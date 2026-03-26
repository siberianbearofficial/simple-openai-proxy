from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

from openai_proxy import openai_compat, schemas

ProviderName = Literal["official", "deepseek", "polza"]

DEFAULT_ROUTE_ATTEMPTS = 3
AUTO_OFFICIAL_MODEL = "gpt-4o"
AUTO_POLZA_MODEL = "deepseek/deepseek-chat"


@dataclass(frozen=True)
class RequestRoute:
    provider: ProviderName
    model: str
    attempts: int = 1

    def apply_to(
        self,
        request: openai_compat.OpenAICompatibleRequest,
    ) -> openai_compat.OpenAICompatibleRequest:
        return cast("openai_compat.OpenAICompatibleRequest", {**request, "model": self.model})


class ModelRouter:
    def build_routes(self, model: str | schemas.OpenAIModel | None) -> list[RequestRoute]:
        model_str = self._normalize_model(model)

        if model_str in {None, "", "auto"}:
            return [
                RequestRoute(
                    provider="deepseek",
                    model=schemas.OpenAIModel.DEEPSEEK.value,
                    attempts=DEFAULT_ROUTE_ATTEMPTS,
                ),
                RequestRoute(
                    provider="official",
                    model=AUTO_OFFICIAL_MODEL,
                    attempts=DEFAULT_ROUTE_ATTEMPTS,
                ),
                RequestRoute(
                    provider="polza",
                    model=AUTO_POLZA_MODEL,
                    attempts=DEFAULT_ROUTE_ATTEMPTS,
                ),
            ]

        if model_str.startswith("deepseek:"):
            return [
                RequestRoute(
                    provider="deepseek",
                    model=self._strip_prefix(model_str, "deepseek:"),
                ),
            ]

        if model_str.startswith("official:"):
            return [
                RequestRoute(
                    provider="official",
                    model=self._strip_prefix(model_str, "official:"),
                ),
            ]

        if model_str.startswith("polza:"):
            return [
                RequestRoute(
                    provider="polza",
                    model=self._strip_prefix(model_str, "polza:"),
                ),
            ]

        if model_str in {
            schemas.OpenAIModel.DEEPSEEK.value,
            schemas.OpenAIModel.DEEPSEEK_FAST.value,
        }:
            return [RequestRoute(provider="deepseek", model=model_str)]

        return [RequestRoute(provider="official", model=model_str)]

    @staticmethod
    def _normalize_model(model: str | schemas.OpenAIModel | None) -> str | None:
        if isinstance(model, schemas.OpenAIModel):
            return model.value
        if model is None:
            return None
        return str(model)

    @staticmethod
    def _strip_prefix(model: str, prefix: str) -> str:
        pure_model = model.removeprefix(prefix)
        if pure_model:
            return pure_model

        err = f"Model name must follow the '{prefix}' prefix"
        raise ValueError(err)
