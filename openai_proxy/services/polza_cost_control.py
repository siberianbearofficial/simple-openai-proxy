from __future__ import annotations

import asyncio
import json
import time
from collections import deque
from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Protocol, cast

import httpx
from loguru import logger

from openai_proxy import openai_compat
from openai_proxy.services.model_routing import ProviderName
from openai_proxy.settings import PolzaCostControlSettings

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletionChunk


class CostLimitExceededError(Exception):
    def __init__(
        self,
        total_cost_rub: float,
        threshold_rub: float,
        window_seconds: int,
    ) -> None:
        message = (
            "Превышен жесткий лимит стоимости запросов к polza: "
            f"за {_format_window(window_seconds)} накоплено {total_cost_rub:.6f} RUB "
            f"при пороге {threshold_rub:.6f} RUB. "
            "Повторите запрос позже."
        )
        super().__init__(message)


class CostThresholdNotifier(Protocol):
    async def notify(
        self,
        notification_text: str,
        log_content: str,
    ) -> None: ...


class LogsAPINotifier:
    def __init__(self, settings: PolzaCostControlSettings) -> None:
        self._settings = settings

    async def notify(
        self,
        notification_text: str,
        log_content: str,
    ) -> None:
        if self._settings.logs_api_base_url is None:
            err = "logs_api_base_url is not configured"
            raise RuntimeError(err)
        if self._settings.logs_api_username is None:
            err = "logs_api_username is not configured"
            raise RuntimeError(err)
        if self._settings.logs_api_password is None:
            err = "logs_api_password is not configured"
            raise RuntimeError(err)

        payload = {
            "application_name": self._settings.application_name,
            "user": self._settings.notification_user,
            "notification_text": notification_text,
            "log_content": log_content,
        }
        auth = httpx.BasicAuth(
            self._settings.logs_api_username,
            self._settings.logs_api_password.get_secret_value(),
        )

        try:
            async with httpx.AsyncClient(
                base_url=str(self._settings.logs_api_base_url).rstrip("/"),
                auth=auth,
                timeout=self._settings.logs_api_timeout_seconds,
            ) as client:
                response = await client.post("/logs", json=payload)
                response.raise_for_status()
        except httpx.HTTPError as ex:
            logger.exception(f"Unable to send polza cost notification to logs-api: {ex}")


@dataclass(slots=True)
class CostEntry:
    created_at: float
    cost_rub: float


class CostTrackingAsyncStream:
    def __init__(
        self,
        stream: openai_compat.ChatCompletionStreamResponse,
        cost_control: PolzaCostControl,
        provider: ProviderName,
        request: Mapping[str, object] | None = None,
    ) -> None:
        self._stream = stream
        self._iterator: AsyncIterator[ChatCompletionChunk] = stream.__aiter__()
        self._cost_control = cost_control
        self._provider = provider
        self._request = request
        self._cost_recorded = False
        self._closed = False

    def __aiter__(self) -> CostTrackingAsyncStream:
        return self

    async def __anext__(self) -> ChatCompletionChunk:
        chunk = await self._iterator.__anext__()
        if not self._cost_recorded and self._cost_control._extract_cost_rub(chunk) is not None:
            await self._cost_control.record_response_cost(
                provider=self._provider,
                response=chunk,
                request=self._request,
            )
            self._cost_recorded = True
        return chunk

    async def close(self) -> None:
        if self._closed:
            return

        self._closed = True
        await self._stream.close()


class PolzaCostControl:
    def __init__(
        self,
        settings: PolzaCostControlSettings | None = None,
        notifier: CostThresholdNotifier | None = None,
        now_provider: Callable[[], float] | None = None,
    ) -> None:
        self._settings = settings or PolzaCostControlSettings()
        self._notifier = notifier
        self._now = now_provider or time.monotonic
        self._entries: deque[CostEntry] = deque()
        self._total_cost_rub = 0.0
        self._lock = asyncio.Lock()

    def wrap_stream(
        self,
        provider: ProviderName,
        response: openai_compat.ChatCompletionStreamResponse,
        request: Mapping[str, object] | None = None,
    ) -> openai_compat.ChatCompletionStreamResponse:
        if provider != "polza" or not self._settings.any_limit_enabled:
            return response
        if isinstance(response, CostTrackingAsyncStream):
            return response

        return CostTrackingAsyncStream(
            stream=response,
            cost_control=self,
            provider=provider,
            request=request,
        )

    async def check_hard_limit(self, provider: ProviderName) -> None:
        if provider != "polza" or not self._settings.hard_limit_enabled:
            return

        async with self._lock:
            now = self._now()
            self._prune_expired_entries(now)
            total_cost_rub = self._total_cost_rub

        hard_threshold_rub = self._settings.hard_threshold_rub
        if hard_threshold_rub is None:
            err = "hard_threshold_rub is not configured"
            raise RuntimeError(err)
        if total_cost_rub >= hard_threshold_rub:
            raise CostLimitExceededError(
                total_cost_rub=total_cost_rub,
                threshold_rub=hard_threshold_rub,
                window_seconds=self._settings.window_seconds,
            )

    async def record_response_cost(
        self,
        provider: ProviderName,
        response: object,
        request: Mapping[str, object] | None = None,
    ) -> None:
        if provider != "polza" or not self._settings.any_limit_enabled:
            return

        cost_rub = self._extract_cost_rub(response)
        if cost_rub is None:
            logger.debug("Polza response does not contain usage.cost_rub, skipping cost tracking")
            return
        if cost_rub <= 0:
            logger.debug(f"Polza response cost is non-positive ({cost_rub}), skipping")
            return

        soft_notification: tuple[str, str] | None = None
        hard_limit_crossed = False

        async with self._lock:
            now = self._now()
            self._prune_expired_entries(now)
            previous_total_cost_rub = self._total_cost_rub
            self._entries.append(CostEntry(created_at=now, cost_rub=cost_rub))
            self._total_cost_rub += cost_rub
            current_total_cost_rub = self._total_cost_rub

            soft_threshold_rub = self._settings.soft_threshold_rub
            if (
                soft_threshold_rub is not None
                and previous_total_cost_rub < soft_threshold_rub <= current_total_cost_rub
            ):
                soft_notification = self._build_soft_limit_notification(
                    request=request,
                    response=response,
                    request_cost_rub=cost_rub,
                    current_total_cost_rub=current_total_cost_rub,
                )

            hard_threshold_rub = self._settings.hard_threshold_rub
            hard_limit_crossed = (
                hard_threshold_rub is not None
                and previous_total_cost_rub < hard_threshold_rub <= current_total_cost_rub
            )

        if hard_limit_crossed:
            window = _format_window(self._settings.window_seconds)
            logger.warning(
                "Polza hard cost limit crossed: "
                f"{current_total_cost_rub:.6f} RUB in {window}",
            )

        if soft_notification is not None and self._notifier is not None:
            notification_text, log_content = soft_notification
            await self._notifier.notify(notification_text, log_content)

    def _prune_expired_entries(self, now: float) -> None:
        cutoff = now - self._settings.window_seconds
        while self._entries and self._entries[0].created_at <= cutoff:
            entry = self._entries.popleft()
            self._total_cost_rub -= entry.cost_rub

        self._total_cost_rub = max(self._total_cost_rub, 0.0)

    def _build_soft_limit_notification(
        self,
        request: Mapping[str, object] | None,
        response: object,
        request_cost_rub: float,
        current_total_cost_rub: float,
    ) -> tuple[str, str]:
        soft_threshold_rub = self._settings.soft_threshold_rub
        if soft_threshold_rub is None:
            err = "soft_threshold_rub is not configured"
            raise RuntimeError(err)

        notification_text = (
            "Пробит мягкий лимит стоимости запросов к polza: "
            f"за {_format_window(self._settings.window_seconds)} накоплено "
            f"{current_total_cost_rub:.6f} RUB при пороге {soft_threshold_rub:.6f} RUB."
        )
        log_content = json.dumps(
            {
                "event": "polza_soft_threshold_exceeded",
                "provider": "polza",
                "model": request.get("model") if request is not None else None,
                "response_id": self._extract_field(response, "id"),
                "request_cost_rub": request_cost_rub,
                "current_total_cost_rub": current_total_cost_rub,
                "soft_threshold_rub": soft_threshold_rub,
                "hard_threshold_rub": self._settings.hard_threshold_rub,
                "window_seconds": self._settings.window_seconds,
                "usage": self._extract_usage_payload(response),
            },
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
        return notification_text, log_content

    def _extract_cost_rub(self, response: object) -> float | None:
        usage = self._extract_usage_payload(response)
        if usage is None:
            return None

        raw_cost = usage.get("cost_rub", usage.get("cost"))
        if raw_cost is None:
            return None

        try:
            return float(raw_cost)
        except (TypeError, ValueError):
            logger.warning(f"Unable to parse polza response cost: {raw_cost!r}")
            return None

    def _extract_usage_payload(self, response: object) -> Mapping[str, object] | None:
        response_mapping = self._to_mapping(response)
        if response_mapping is not None:
            usage_mapping = self._to_mapping(response_mapping.get("usage"))
            if usage_mapping is not None:
                return usage_mapping

        return self._to_mapping(getattr(response, "usage", None))

    def _extract_field(self, response: object, field_name: str) -> object | None:
        response_mapping = self._to_mapping(response)
        if response_mapping is not None and field_name in response_mapping:
            return response_mapping[field_name]

        return getattr(response, field_name, None)

    @staticmethod
    def _to_mapping(value: object) -> Mapping[str, object] | None:
        if isinstance(value, Mapping):
            return cast("Mapping[str, object]", value)

        model_dump = getattr(value, "model_dump", None)
        if callable(model_dump):
            try:
                dumped_value = model_dump(exclude_none=True)
            except TypeError:
                dumped_value = model_dump()

            if isinstance(dumped_value, Mapping):
                return cast("Mapping[str, object]", dumped_value)

        model_extra = getattr(value, "model_extra", None)
        if isinstance(model_extra, Mapping):
            return cast("Mapping[str, object]", model_extra)

        return None


def _format_window(window_seconds: int) -> str:
    if window_seconds % 3600 == 0:
        hours = window_seconds // 3600
        if hours == 1:
            return "последний час"
        return f"последние {hours} ч."

    if window_seconds % 60 == 0:
        minutes = window_seconds // 60
        if minutes == 1:
            return "последнюю минуту"
        return f"последние {minutes} мин."

    return f"последние {window_seconds} сек."


@lru_cache
def get_polza_cost_control() -> PolzaCostControl:
    settings = PolzaCostControlSettings()
    notifier: CostThresholdNotifier | None = None
    if settings.soft_limit_enabled:
        notifier = LogsAPINotifier(settings)
    return PolzaCostControl(settings=settings, notifier=notifier)
