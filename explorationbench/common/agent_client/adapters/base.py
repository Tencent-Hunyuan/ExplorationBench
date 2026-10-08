from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Protocol

import httpx

from ..routing import AgentClientConfig
from ..types import (
    HistoryEvent,
    ProviderResult,
    ToolDefinition,
    ToolResult,
)


class MalformedTurnError(RuntimeError):
    """The provider returned a turn this protocol cannot read.

    Tool-call arguments that are not JSON are the common case. It is the
    model's mistake rather than the transport's, so the session treats it as
    retryable: one unreadable turn should cost a re-ask, not the whole run.
    """


@dataclass(slots=True)
class PreparedInput:
    provider_items: list[dict[str, Any]]
    history_events: list[HistoryEvent]


@dataclass(slots=True)
class TransportResponse:
    status_code: int
    data: dict[str, Any] | None
    text: str
    headers: dict[str, str]


class Transport(Protocol):
    def post(
        self,
        endpoint: str,
        *,
        headers: dict[str, str],
        payload: dict[str, Any],
        timeout: float,
    ) -> TransportResponse: ...


#: httpx defaults to 100 connections with 20 of them kept alive, which is a
#: ceiling the caller never sees: a run scoring 300 questions with 150 workers
#: queues a third of them on the pool, and every request past the twentieth
#: pays a fresh TLS handshake. One client serves a whole run, so the pool has
#: to be sized for that run's worker count rather than for a browser.
_POOL_LIMIT = max(1, int(os.environ.get("AGENT_HTTP_POOL", "256") or 256))


class HttpxTransport:
    def __init__(self, pool_limit: int | None = None) -> None:
        limit = pool_limit or _POOL_LIMIT
        self._client = httpx.Client(
            limits=httpx.Limits(
                max_connections=limit,
                max_keepalive_connections=limit,
            )
        )

    def post(
        self,
        endpoint: str,
        *,
        headers: dict[str, str],
        payload: dict[str, Any],
        timeout: float,
    ) -> TransportResponse:
        response = self._client.post(
            endpoint,
            headers=headers,
            json=payload,
            timeout=timeout,
        )
        try:
            data = response.json()
        except ValueError:
            data = None
        return TransportResponse(
            status_code=response.status_code,
            data=data,
            text=response.text,
            headers=dict(response.headers),
        )

    def close(self) -> None:
        self._client.close()


class ProviderAdapter(ABC):
    def __init__(self, config: AgentClientConfig) -> None:
        self.config = config

    @abstractmethod
    def prepare_user(self, content: Any) -> PreparedInput:
        raise NotImplementedError

    @abstractmethod
    def prepare_tool_results(
        self,
        results: list[ToolResult],
        *,
        call_names: dict[str, str],
    ) -> PreparedInput:
        raise NotImplementedError

    @abstractmethod
    def build_request(
        self,
        *,
        system: str | list[dict[str, Any]] | None,
        tools: list[ToolDefinition],
        provider_history: list[dict[str, Any]],
        input_items: list[dict[str, Any]],
        last_response_id: str | None,
        request_overrides: dict[str, Any],
    ) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def parse_response(
        self,
        raw: dict[str, Any],
        *,
        previous_response_id: str | None,
    ) -> ProviderResult:
        raise NotImplementedError

    def response_error(self, data: dict[str, Any] | None) -> str | None:
        if not data:
            return None
        error = data.get("error")
        if error:
            return str(error)
        return None


def text_history_event(
    role: str,
    content: Any,
    *,
    provider_payload: Any,
) -> HistoryEvent:
    return HistoryEvent(
        kind="message",
        role=role,
        payload={"content": content},
        provider_payload=provider_payload,
    )
