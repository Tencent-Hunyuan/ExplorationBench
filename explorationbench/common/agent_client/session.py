from __future__ import annotations

import hashlib
import json
import random
import re
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator

import httpx

from .adapters.base import (
    HttpxTransport,
    MalformedTurnError,
    PreparedInput,
    ProviderAdapter,
    Transport,
    TransportResponse,
)
from .endpoint_gate import hold_slot
from .routing import AgentClientConfig, new_cache_task_id, same_model
from .trace import JsonlTraceStore, UsageLedger
from .types import (
    AgentResponse,
    HistoryEvent,
    Provider,
    ProviderResult,
    SessionState,
    ToolCall,
    ToolDefinition,
    ToolResult,
    UsageRecord,
    json_copy,
    new_id,
    normalize_tools,
)


#: OpenAI's documented ceiling for `prompt_cache_key`; longer is a 400.
_CACHE_KEY_LIMIT = 64


def _clamp_cache_key(value: str) -> str:
    if len(value) <= _CACHE_KEY_LIMIT:
        return value
    digest = hashlib.md5(value.encode("utf-8")).hexdigest()[:10]
    return f"{value[:_CACHE_KEY_LIMIT - 11]}-{digest}"


class AgentClientError(RuntimeError):
    pass


class AgentEmptyCompletionError(AgentClientError):
    """A 2xx turn that carries no message and no tool call."""


class AgentHTTPError(AgentClientError):
    def __init__(
        self,
        status_code: int,
        message: str,
        *,
        response: dict[str, Any] | None = None,
        retryable: bool | None = None,
    ) -> None:
        super().__init__(f"HTTP {status_code}: {message}")
        self.status_code = status_code
        self.response = response
        # Set when the caller knows something the status line does not, such
        # as an auth failure on a key that has already been accepted.
        self.retryable = retryable


def _adapter_for(config: AgentClientConfig) -> ProviderAdapter:
    if config.provider is Provider.ANTHROPIC:
        from .adapters.anthropic import AnthropicMessagesAdapter

        return AnthropicMessagesAdapter(config)
    if config.provider is Provider.GEMINI:
        from .adapters.gemini import GeminiGenerateContentAdapter

        return GeminiGenerateContentAdapter(config)
    if config.provider is Provider.OPENAI_RESPONSES:
        from .adapters.openai_responses import OpenAIResponsesAdapter

        return OpenAIResponsesAdapter(config)
    from .adapters.legacy_chat import LegacyChatAdapter

    return LegacyChatAdapter(config)


# The gateway occasionally answers a perfectly valid request with a 403 whose
# body is an IDC network-policy notice instead of an authorization decision.
# Those are transient and worth retrying. A real 403 -- a model this account
# may not call -- is not, and retrying it only burns quota, so match on the
# notice text rather than on the status alone.
_TRANSIENT_403 = re.compile(r"IDC环境|igate")


def _is_transient_403(message: str) -> bool:
    return bool(_TRANSIENT_403.search(message))


# A stored Responses item lives on one Azure resource; when the gateway sends
# the continuation to a different one it answers 400 rather than 404, so the
# text is the only way to tell this apart from a malformed request.
# A stored response lives on one upstream, so a load-balanced gateway can send
# a continuation somewhere that has never heard of it. Vendors word that
# differently: Azure names the resource, others just fail to find the id.
_ORPHANED_CONTINUATION = re.compile(
    r"created under a different Azure OpenAI resource"
    r"|previous_response_id[^.]{0,80}?(?:not found|does not exist|invalid)"
    r"|(?:not found|no such|unknown)[^.]{0,40}?previous_response_id"
    # Prose, not a field name: GPT-5.6 answers "Previous response with id
    # 'resp_...' not found" and matched none of the patterns above, so the
    # rebuild below never fired and 175 of one run's 280 graded questions
    # died on a recovery that was already written.
    r"|previous response with id[^.]{0,80}?not found",
    re.IGNORECASE,
)


def _is_orphaned_continuation(message: str) -> bool:
    return bool(_ORPHANED_CONTINUATION.search(message))


# The same orphaned continuation, worded as a capacity problem. Asked to
# resume a stored response whose Azure resource has left the pool, the GatewayA
# gateway does not say the id is unknown -- it reports having no account at
# all, with a 500 rather than a 400. Read literally that is a transient
# outage, so the client retried it forever: one repeated-answer job spent
# five hours on 3500 calls and completed none, while the identical body with
# previous_response_id removed answered normally at the same minute.
#
# A genuinely empty pool fails the rebuilt request too, and costs only the
# replayed history, so this is worth trying before treating it as transient.
_NO_ACCOUNT = re.compile(r"PlatformNoAvailableAccount", re.IGNORECASE)


def _is_no_account(message: str) -> bool:
    return bool(_NO_ACCOUNT.search(message))


# The completion budget is reserved inside the context window, not on top of
# it, so a long enough history turns a budget that was fine all run into a 400.
# One AlienCode episode explored to 921k tokens and then died on the next turn
# because 921k + a 131k budget is over a 1M window. The vendor counts the
# prompt for us, so the reply says exactly how much room is left.
_CONTEXT_OVERFLOW = re.compile(
    r"maximum context length is (\d+) tokens"
    r".*?\((\d+) in the messages",
    re.IGNORECASE | re.DOTALL,
)

# Room kept for whatever the next attempt adds to the prompt.
_CONTEXT_OVERFLOW_MARGIN = 1024

# Whichever of these a protocol uses, it means the same thing.
_BUDGET_KEYS = ("max_completion_tokens", "max_output_tokens", "max_tokens")


def _context_overflow_budget(message: str) -> int | None:
    """The completion budget that would still fit, from the vendor's count."""

    match = _CONTEXT_OVERFLOW.search(message)
    if not match:
        return None
    window, prompt = (int(group) for group in match.groups())
    room = window - prompt - _CONTEXT_OVERFLOW_MARGIN
    return room if room > 0 else None


def _drop_oldest_reasoning(
    request: dict[str, Any],
    message: str,
    *,
    keep_recent: int = 2,
) -> int:
    """Drop the oldest replayed reasoning until the prompt fits the window.

    Only routes that cannot continue server-side reach this: they replay the
    whole history inline, so a long trajectory eventually asks for more than
    the model can hold, and no completion budget is small enough to rescue it
    because the messages alone are over.

    Reasoning is the model's own scratchpad. Dropping the earliest of it keeps
    every probe, tool result and answer intact, which is what the run is
    scored on, and is what a server-side continuation would have compacted
    away anyway. The vendor reports both the window and what it counted, so
    the tokens-per-character rate is measured here rather than guessed.
    """

    match = _CONTEXT_OVERFLOW.search(message)
    items = request.get("input")
    if not match or not isinstance(items, list):
        return 0
    window, prompt = (int(group) for group in match.groups())
    total_chars = sum(len(json.dumps(item, ensure_ascii=False)) for item in items)
    if prompt <= 0 or total_chars <= 0:
        return 0
    chars_per_token = total_chars / prompt
    over_tokens = prompt - window + _CONTEXT_OVERFLOW_MARGIN
    if over_tokens <= 0:
        return 0
    target_chars = over_tokens * chars_per_token

    protected = {
        id(item)
        for item in [
            item for item in items
            if isinstance(item, dict) and item.get("type") == "reasoning"
        ][-keep_recent:]
    }
    freed = 0.0
    kept: list[Any] = []
    dropped = 0
    for item in items:
        if (
            freed < target_chars
            and isinstance(item, dict)
            and item.get("type") == "reasoning"
            and id(item) not in protected
        ):
            freed += len(json.dumps(item, ensure_ascii=False))
            dropped += 1
            continue
        kept.append(item)
    if dropped:
        request["input"] = kept
    return dropped


# Reasoning replayed inline travels as ciphertext only the Azure resource
# that wrote it can read. Recovering a dropped continuation moves the turn to
# a different resource, which then rejects the whole request rather than
# ignoring what it cannot read.
_UNREADABLE_REASONING = re.compile(
    r"invalid_encrypted_content"
    r"|encrypted content could not be",
    re.IGNORECASE,
)


def _drop_unreadable_reasoning(request: dict[str, Any]) -> int:
    """Strip reasoning the new upstream cannot decrypt.

    The item is removed rather than emptied: without ``id`` or
    ``encrypted_content`` a reasoning item carries nothing the protocol
    accepts as input. The probes, tool results and answers around it are
    what the run is scored on and they stay intact, so the turn loses the
    model's scratchpad but not its evidence.
    """

    items = request.get("input")
    if not isinstance(items, list):
        return 0
    kept = [
        item for item in items
        if not (
            isinstance(item, dict)
            and item.get("type") == "reasoning"
            and item.get("encrypted_content")
        )
    ]
    dropped = len(items) - len(kept)
    if dropped:
        request["input"] = kept
    return dropped


def _is_content_free(result: ProviderResult) -> bool:
    """True when a 2xx turn carries nothing a caller could act on.

    Doubao ends a `completed` response with a reasoning item and no message
    once in a while; the tokens are spent but the turn says nothing, so the
    grader sees a blank answer.
    """

    return not (result.text or "").strip() and not result.tool_calls


def _is_throttle_message(message: str) -> bool:
    """Whether a response body says "slow down" whatever its status line says.

    Not every provider throttles with 429. Ali returns HTTP 400 carrying
    ``{'code': 'Throttling.Concurrency'}``, which read as an ordinary bad
    request and so was never retried: three qwen3.8-max control runs scored
    near zero because over a thousand graded calls were dropped on the first
    attempt.
    """

    lowered = message.lower()
    return any(
        marker in lowered
        for marker in (
            "throttling",
            "rate limit",
            "too many requests",
            "resource exhausted",
            "resourceexhausted",
            "<429>",
        )
    )


# A wrong key fails every call, so an auth failure on a key that has already
# been accepted is the auth service faltering rather than a verdict on the
# credentials. GatewayA returned two of these across 187 calls of one run, each
# costing a graded question, while the same key served the other 185.
def _is_proven_auth_failure(status_code: int, auth_proven: bool) -> bool:
    return status_code == 401 and auth_proven


def is_retryable_exception(error: BaseException) -> bool:
    override = getattr(error, "retryable", None)
    if isinstance(override, bool):
        return override
    if isinstance(error, AgentEmptyCompletionError):
        return True
    # A turn whose tool-call arguments will not parse is the model's mistake,
    # not the transport's, and it is exactly the kind of thing a re-ask fixes.
    # Letting it propagate ends the whole evaluation over one bad turn.
    if isinstance(error, MalformedTurnError):
        return True
    status = getattr(error, "status_code", None)
    if status is None:
        status = getattr(
            getattr(error, "response", None), "status_code", None
        )
    if status is not None:
        try:
            code = int(status)
        except (TypeError, ValueError):
            code = 0
        if code == 403:
            return _is_transient_403(str(error))
        if code in {408, 409, 429} or 500 <= code < 600:
            return True
        return _is_throttle_message(str(error))
    if isinstance(
        error,
        (
            httpx.TimeoutException,
            httpx.TransportError,
            TimeoutError,
            ConnectionError,
        ),
    ):
        return True
    name = type(error).__name__.lower()
    if "timeout" in name or "connection" in name:
        return True
    message = str(error).lower()
    return _is_throttle_message(message) or any(
        marker in message
        for marker in (
            "502 bad gateway",
            "503 service unavailable",
            "504 gateway",
        )
    )


class AgentClient:
    """Provider-native agent client.

    A client owns transport, trace and usage aggregation. Each logical
    conversation is an AgentSession; forked sessions share the trace/ledger but
    have independent provider histories and continuation IDs.
    """

    def __init__(
        self,
        config: AgentClientConfig | None = None,
        *,
        model: str | None = None,
        transport: Transport | None = None,
        trace_store: JsonlTraceStore | None = None,
        usage_ledger: UsageLedger | None = None,
        **config_kwargs: Any,
    ) -> None:
        if config is None:
            if not model:
                raise ValueError("pass config or model")
            config = AgentClientConfig(model=model, **config_kwargs)
        elif model is not None or config_kwargs:
            raise ValueError("do not combine config with model/config kwargs")
        self.config = config
        self.adapter = _adapter_for(config)
        self.transport = transport or HttpxTransport()
        self.trace_store = trace_store or JsonlTraceStore(
            config.trace_path,
            snapshot_dir=config.snapshot_dir,
            fsync=config.trace_fsync,
        )
        self.usage_ledger = usage_ledger or UsageLedger()
        # Set once the gateway has accepted these credentials, which is what
        # makes a later auth failure a transient fault rather than a verdict.
        self.auth_proven = False

    def create_session(
        self,
        *,
        system: str | list[dict[str, Any]] | None = None,
        tools: Iterable[ToolDefinition | dict[str, Any]] | None = None,
        session_id: str | None = None,
    ) -> "AgentSession":
        session = AgentSession(
            client=self,
            system=json_copy(system),
            tools=normalize_tools(tools),
            session_id=session_id or new_id("session"),
        )
        self.trace_store.append({
            "record_type": "session_created",
            "provider": self.config.provider.value,
            "model": self.config.model,
            "session_id": session.session_id,
        })
        session.snapshot()
        return session

    @classmethod
    def from_snapshot(
        cls,
        path: str | Path,
        *,
        headers: dict[str, str] | None = None,
        transport: Transport | None = None,
    ) -> tuple["AgentClient", "AgentSession"]:
        import json

        with Path(path).open(encoding="utf-8") as source:
            state = SessionState.from_dict(json.load(source))
        config = AgentClientConfig.from_public_dict(
            state.config, headers=headers
        )
        client = cls(config, transport=transport)
        return client, client.restore_session(state)

    def restore_session(
        self,
        state: SessionState | dict[str, Any] | str | Path,
    ) -> "AgentSession":
        if isinstance(state, (str, Path)):
            import json

            with Path(state).open(encoding="utf-8") as source:
                state = json.load(source)
        if isinstance(state, dict):
            state = SessionState.from_dict(state)
        if state.provider != self.config.provider.value:
            raise ValueError(
                f"snapshot provider {state.provider!r} does not match "
                f"client provider {self.config.provider.value!r}"
            )
        if not same_model(state.model, self.config.model):
            raise ValueError(
                f"snapshot model {state.model!r} does not match "
                f"client model {self.config.model!r}"
            )
        session = AgentSession.from_state(self, state)
        usage_events = [
            event for event in session._history
            if event.kind == "usage"
        ]
        if usage_events:
            for index, event in enumerate(usage_events):
                self.usage_ledger.add(
                    UsageRecord.from_dict(event.payload),
                    session_id=session.session_id,
                    exchange_id=(
                        event.exchange_id
                        or f"restored_snapshot_{index + 1}"
                    ),
                )
        else:
            restored = UsageRecord.from_dict(state.usage)
            if any((
                restored.input_tokens,
                restored.output_tokens,
                restored.reasoning_tokens,
                restored.total_tokens,
                restored.cache_read_input_tokens,
                restored.cache_write_input_tokens,
                restored.cache_creation_input_tokens,
            )):
                self.usage_ledger.add(
                    restored,
                    session_id=session.session_id,
                    exchange_id="restored_snapshot",
                )
        return session

    def get_usage(self) -> dict[str, Any]:
        return self.usage_ledger.snapshot()

    def close(self) -> None:
        close = getattr(self.transport, "close", None)
        if callable(close):
            close()

    def __enter__(self) -> "AgentClient":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


class AgentSession:
    def __init__(
        self,
        *,
        client: AgentClient,
        system: str | list[dict[str, Any]] | None,
        tools: list[ToolDefinition],
        session_id: str,
        history: list[HistoryEvent] | None = None,
        provider_history: list[dict[str, Any]] | None = None,
        last_response_id: str | None = None,
        pending_tool_calls: list[ToolCall] | None = None,
        usage: UsageRecord | None = None,
        parent_session_id: str | None = None,
        forked_from_event_id: str | None = None,
    ) -> None:
        self.client = client
        self.system = json_copy(system)
        self.tools = normalize_tools(tools)
        self.session_id = session_id
        self.parent_session_id = parent_session_id
        self.forked_from_event_id = forked_from_event_id
        self._history = list(history or [])
        self._provider_history = json_copy(provider_history or [])
        self.last_response_id = last_response_id
        self._pending_tool_calls = list(pending_tool_calls or [])
        self._usage = usage or UsageRecord(
            provider=client.config.provider.value
        )
        self._last_response: AgentResponse | None = None
        self._last_meta: dict[str, Any] = {}
        self._lock = threading.RLock()
        # One task id per session, which in an evaluation means one per graded
        # question. The gateway keeps a task on the account it picked for it,
        # so a question's turns still share a prompt cache, while separate
        # questions land on separate accounts instead of queueing behind one.
        self.cache_task_id = new_cache_task_id()

    @classmethod
    def from_state(
        cls,
        client: AgentClient,
        state: SessionState,
    ) -> "AgentSession":
        return cls(
            client=client,
            system=state.system,
            tools=[
                ToolDefinition.from_value(value) for value in state.tools
            ],
            session_id=state.session_id,
            history=[
                HistoryEvent.from_dict(value) for value in state.history
            ],
            provider_history=state.provider_history,
            last_response_id=state.last_response_id,
            pending_tool_calls=[
                ToolCall(
                    call_id=str(value["call_id"]),
                    name=str(value["name"]),
                    arguments=json_copy(value.get("arguments", {})),
                    raw=json_copy(value.get("raw", {})),
                )
                for value in state.pending_tool_calls
            ],
            usage=UsageRecord.from_dict(state.usage),
            parent_session_id=state.parent_session_id,
            forked_from_event_id=state.forked_from_event_id,
        )

    @property
    def history(self) -> list[dict[str, Any]]:
        return [event.to_dict() for event in self._history]

    @property
    def provider_history(self) -> list[dict[str, Any]]:
        return json_copy(self._provider_history)

    @property
    def pending_tool_calls(self) -> list[ToolCall]:
        return [
            ToolCall(
                call_id=call.call_id,
                name=call.name,
                arguments=json_copy(call.arguments),
                raw=json_copy(call.raw),
            )
            for call in self._pending_tool_calls
        ]

    @property
    def last_response(self) -> AgentResponse | None:
        return self._last_response

    @property
    def last_call_meta(self) -> dict[str, Any]:
        return json_copy(self._last_meta)

    def send_user(
        self,
        content: Any,
        *,
        label: str = "",
        request_overrides: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> AgentResponse:
        with self._lock:
            if self._pending_tool_calls:
                pending = [
                    call.call_id for call in self._pending_tool_calls
                ]
                raise AgentClientError(
                    "submit tool results before sending another user "
                    f"message; pending={pending}"
                )
            prepared = self.client.adapter.prepare_user(content)
            return self._exchange(
                prepared,
                label=label,
                request_overrides=request_overrides,
                timeout=timeout,
            )

    def submit_tool_results(
        self,
        results: Iterable[ToolResult | dict[str, Any]],
        *,
        label: str = "",
        request_overrides: dict[str, Any] | None = None,
    ) -> AgentResponse:
        normalized = [ToolResult.from_value(value) for value in results]
        with self._lock:
            expected = {call.call_id for call in self._pending_tool_calls}
            received = {result.call_id for result in normalized}
            if not expected:
                raise AgentClientError("there are no pending tool calls")
            if len(received) != len(normalized):
                raise AgentClientError("duplicate tool result call_id")
            if received != expected:
                missing = sorted(expected - received)
                extra = sorted(received - expected)
                raise AgentClientError(
                    f"tool results must exactly match pending calls; "
                    f"missing={missing}, extra={extra}"
                )
            by_id = {result.call_id: result for result in normalized}
            normalized = [
                by_id[call.call_id] for call in self._pending_tool_calls
            ]
            call_names = {
                call.call_id: call.name for call in self._pending_tool_calls
            }
            prepared = self.client.adapter.prepare_tool_results(
                normalized, call_names=call_names
            )
            return self._exchange(
                prepared,
                label=label,
                request_overrides=request_overrides,
            )

    @contextmanager
    def using_tools(
        self,
        tools: Iterable[ToolDefinition | dict[str, Any]] | None,
    ) -> Iterator["AgentSession"]:
        """Change what this session may call, for the duration of the block.

        An evaluation often opens the interpreter for one exploration phase
        while every graded phase on the same conversation stays closed-book.
        `fork` cannot express that: the exploration has to accumulate on the
        main line, and a branch would strand it. Leaving the block with tool
        calls outstanding is refused, since the next turn would carry tools the
        pending calls were not made against.
        """

        previous = self.tools
        self.tools = normalize_tools(tools)
        unwinding = False
        try:
            yield self
        except BaseException:
            unwinding = True
            raise
        finally:
            self.tools = previous
            if self._pending_tool_calls and not unwinding:
                pending = [
                    call.call_id for call in self._pending_tool_calls
                ]
                raise AgentClientError(
                    "submit tool results before leaving using_tools; "
                    f"pending={pending}"
                )

    def _with_session_cache_key(
        self, request_overrides: dict[str, Any] | None
    ) -> dict[str, Any]:
        """Give this session its own OpenAI cache key, where the route wants one.

        Gateway account assignment is ``cache_task_id`` in the auth query and
        is already one id per session. Azure also keys OpenAI's prefix cache
        off the body's ``prompt_cache_key``; a run that sets that once still
        hands every concurrent question the same OpenAI key, so the run's
        key stays as the prefix and the session id separates the questions
        inside it.

        A caller that named the key for this call meant that key, so an
        explicit override is left alone.

        The assembled value is clamped, because this is where its length is
        finally known: OpenAI rejects anything past 64 characters with a
        400, and a run id long enough to overrun only appears in some
        cohorts. A control-arm id took `aliencode:gpt-5.6-sol-max:...` to
        sixty-five and every call of three runs failed on the first turn.
        Clamping by digest rather than truncation keeps two ids that share
        a prefix in separate cache lanes.
        """

        overrides = dict(request_overrides or {})
        if not self.client.config.cache_key_spread:
            return overrides
        if "prompt_cache_key" in overrides:
            return overrides
        base = str(self.client.config.settings.get("prompt_cache_key") or "")
        if not base:
            return overrides
        overrides["prompt_cache_key"] = _clamp_cache_key(
            f"{base}:{self.cache_task_id[:12]}")
        return overrides

    def fork(
        self,
        *,
        session_id: str | None = None,
        tools: Iterable[ToolDefinition | dict[str, Any]] | None = None,
    ) -> "AgentSession":
        """Branch the conversation, preserving provider-native continuation.

        ``tools`` defaults to inheriting the parent's, but pass an explicit
        (possibly empty) list to change what the branch may call. Handing a
        branch fewer tools than its parent is how a caller keeps an evaluation
        closed-book: the shared history stays intact while the fork loses the
        ability to act.
        """

        with self._lock:
            if self._pending_tool_calls:
                pending = [
                    call.call_id for call in self._pending_tool_calls
                ]
                raise AgentClientError(
                    "submit tool results before forking the session; "
                    f"pending={pending}"
                )
            anchor = (
                self._history[-1].event_id if self._history else None
            )
            child = AgentSession(
                client=self.client,
                system=self.system,
                tools=(
                    self.tools
                    if tools is None
                    else [ToolDefinition.from_value(tool) for tool in tools]
                ),
                session_id=session_id or new_id("session"),
                history=[
                    HistoryEvent.from_dict(event.to_dict())
                    for event in self._history
                ],
                provider_history=self._provider_history,
                last_response_id=self.last_response_id,
                pending_tool_calls=self._pending_tool_calls,
                usage=UsageRecord.from_dict(self._usage.to_dict()),
                parent_session_id=self.session_id,
                forked_from_event_id=anchor,
            )
        self.client.trace_store.append({
            "record_type": "session_fork",
            "provider": self.client.config.provider.value,
            "model": self.client.config.model,
            "parent_session_id": self.session_id,
            "session_id": child.session_id,
            "forked_from_event_id": anchor,
            "previous_response_id": child.last_response_id,
            # Recorded so an audit can prove which branches could act and
            # which were closed-book.
            "tools": [tool.name for tool in child.tools],
            "parent_tools": [tool.name for tool in self.tools],
            "continuation_signatures": _continuation_signatures(
                child._provider_history,
                self.client.config.provider,
            ),
        })
        child.snapshot()
        return child

    def state(self) -> SessionState:
        with self._lock:
            return SessionState(
                provider=self.client.config.provider.value,
                model=self.client.config.model,
                system=json_copy(self.system),
                tools=[tool.to_dict() for tool in self.tools],
                history=self.history,
                provider_history=json_copy(self._provider_history),
                last_response_id=self.last_response_id,
                pending_tool_calls=[
                    call.to_dict() for call in self._pending_tool_calls
                ],
                usage=self._usage.to_dict(),
                config=self.client.config.public_dict(),
                session_id=self.session_id,
                parent_session_id=self.parent_session_id,
                forked_from_event_id=self.forked_from_event_id,
            )

    def snapshot(self) -> Path | None:
        return self.client.trace_store.save_snapshot(self.state())

    def restore_in_place(
        self,
        state: SessionState | dict[str, Any],
        *,
        preserve_usage: UsageRecord | None = None,
    ) -> None:
        """Roll back conversation state while retaining an attempted call's cost."""

        if isinstance(state, dict):
            state = SessionState.from_dict(state)
        if state.session_id != self.session_id:
            raise ValueError("cannot restore a different session in place")
        restored = AgentSession.from_state(self.client, state)
        with self._lock:
            discarded_response_id = self.last_response_id
            discarded_exchange_id = (
                self._last_response.exchange_id
                if self._last_response is not None
                else None
            )
            self.system = restored.system
            self.tools = restored.tools
            self._history = restored._history
            self._provider_history = restored._provider_history
            self.last_response_id = restored.last_response_id
            self._pending_tool_calls = restored._pending_tool_calls
            self._usage = restored._usage
            if preserve_usage is not None:
                self._usage.add(preserve_usage)
            self._last_response = None
            self._last_meta = {}
            self.client.trace_store.append({
                "record_type": "session_restore",
                "provider": self.client.config.provider.value,
                "model": self.client.config.model,
                "session_id": self.session_id,
                "discarded_exchange_id": discarded_exchange_id,
                "discarded_response_id": discarded_response_id,
                "restored_response_id": self.last_response_id,
                "restored_to_event_id": (
                    self._history[-1].event_id
                    if self._history
                    else None
                ),
                "continuation_signatures": _continuation_signatures(
                    self._provider_history,
                    self.client.config.provider,
                ),
                "pending_tool_calls": [
                    call.to_dict() for call in self._pending_tool_calls
                ],
                "preserved_usage": (
                    preserve_usage.to_dict()
                    if preserve_usage is not None
                    else None
                ),
            })
            self.snapshot()

    def _exchange(
        self,
        prepared: PreparedInput,
        *,
        label: str,
        request_overrides: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> AgentResponse:
        # A logical session is sequential. Fork before using it concurrently.
        with self._lock:
            exchange_id = new_id("exchange")
            previous_response_id = self.last_response_id
            overrides = self._with_session_cache_key(request_overrides)
            request = self.client.adapter.build_request(
                system=self.system,
                tools=self.tools,
                provider_history=self._provider_history,
                input_items=prepared.provider_items,
                last_response_id=previous_response_id,
                request_overrides=dict(overrides),
            )
            result = None
            successful_response: TransportResponse | None = None
            retries = 0
            empty_retries = 0
            started = time.monotonic()
            for attempt in range(self.client.config.max_retries + 1):
                response: TransportResponse | None = None
                try:
                    # Only the call itself takes a slot; a backoff sleep must
                    # not keep one from a sibling run.
                    with hold_slot():
                        response = self.client.transport.post(
                            str(self.client.config.endpoint),
                            headers=self.client.config.auth_headers(
                                self.cache_task_id
                            ),
                            payload=request,
                            timeout=self.client.config.http_timeout_for(
                                timeout),
                        )
                    if (
                        previous_response_id
                        and response.status_code >= 500
                        and _is_no_account(_response_message(response))
                    ):
                        # Asked before the retry classifier, which would
                        # otherwise read this as a transient outage and keep
                        # re-sending a continuation nothing can serve.
                        self.client.trace_store.append({
                            "record_type": "continuation_dropped",
                            "session_id": self.session_id,
                            "exchange_id": exchange_id,
                            "label": label,
                            "dropped_response_id": previous_response_id,
                            "reason": "no_available_account",
                        })
                        previous_response_id = None
                        self.last_response_id = None
                        request = self.client.adapter.build_request(
                            system=self.system,
                            tools=self.tools,
                            provider_history=self._provider_history,
                            input_items=prepared.provider_items,
                            last_response_id=None,
                            request_overrides=dict(overrides),
                        )
                        continue
                    if _retryable_status(
                        response.status_code,
                        _response_message(response),
                        auth_proven=self.client.auth_proven,
                    ):
                        raise AgentHTTPError(
                            response.status_code,
                            _response_message(response),
                            response=response.data,
                            retryable=True,
                        )
                    if (
                        response.status_code == 400
                        and previous_response_id
                        and _is_orphaned_continuation(
                            _response_message(response)
                        )
                    ):
                        # The gateway load-balances across Azure resources and
                        # a stored response only exists on the one that made
                        # it. Rebuilding without the id replays the history
                        # inline, which costs tokens but saves the session.
                        self.client.trace_store.append({
                            "record_type": "continuation_dropped",
                            "session_id": self.session_id,
                            "exchange_id": exchange_id,
                            "label": label,
                            "dropped_response_id": previous_response_id,
                        })
                        previous_response_id = None
                        self.last_response_id = None
                        request = self.client.adapter.build_request(
                            system=self.system,
                            tools=self.tools,
                            provider_history=self._provider_history,
                            input_items=prepared.provider_items,
                            last_response_id=None,
                            request_overrides=dict(overrides),
                        )
                        continue
                    if (
                        response.status_code == 400
                        and _UNREADABLE_REASONING.search(
                            _response_message(response)
                        )
                    ):
                        unreadable = _drop_unreadable_reasoning(request)
                        if unreadable:
                            self.client.trace_store.append({
                                "record_type": "encrypted_reasoning_dropped",
                                "session_id": self.session_id,
                                "exchange_id": exchange_id,
                                "label": label,
                                "dropped_items": unreadable,
                            })
                            continue
                    if response.status_code == 400:
                        room = _context_overflow_budget(
                            _response_message(response)
                        )
                        present = [
                            key for key in _BUDGET_KEYS if key in request
                        ]
                        if room and present and room < min(
                            int(request[key]) for key in present
                        ):
                            # Shrinking the answer is the only move that keeps
                            # the episode: the history is what it is, and
                            # ending the run here would lose every milestone
                            # still ahead of it.
                            self.client.trace_store.append({
                                "record_type": "completion_budget_reduced",
                                "session_id": self.session_id,
                                "exchange_id": exchange_id,
                                "label": label,
                                "budget": room,
                            })
                            for key in present:
                                request[key] = room
                            continue
                        dropped = _drop_oldest_reasoning(
                            request, _response_message(response)
                        )
                        if dropped:
                            # No completion budget is small enough here: the
                            # replayed history alone is over the window.
                            self.client.trace_store.append({
                                "record_type": "history_reasoning_dropped",
                                "session_id": self.session_id,
                                "exchange_id": exchange_id,
                                "label": label,
                                "dropped_items": dropped,
                            })
                            continue
                    if response.status_code >= 400:
                        raise AgentHTTPError(
                            response.status_code,
                            _response_message(response),
                            response=response.data,
                        )
                    if not isinstance(response.data, dict):
                        raise AgentClientError(
                            "provider returned a non-object JSON success "
                            "response"
                        )
                    provider_error = self.client.adapter.response_error(
                        response.data
                    )
                    if provider_error:
                        raise AgentClientError(provider_error)
                    result = self.client.adapter.parse_response(
                        response.data,
                        previous_response_id=previous_response_id,
                    )
                    call_ids = [
                        call.call_id for call in result.tool_calls
                    ]
                    if len(call_ids) != len(set(call_ids)):
                        raise AgentClientError(
                            "provider returned duplicate tool call IDs"
                        )
                    empty_budget = self.client.config.max_empty_retries
                    if (
                        _is_content_free(result)
                        and empty_retries < empty_budget
                        # An empty turn scores zero but an exception ends the
                        # task, so silence is only worth re-asking while an
                        # attempt is left to spend on it.
                        and attempt < self.client.config.max_retries
                    ):
                        empty_retries += 1
                        raise AgentEmptyCompletionError(
                            "provider returned no message and no tool call "
                            f"(empty retry {empty_retries}/{empty_budget})"
                        )
                    retries = attempt
                    successful_response = response
                    self.client.auth_proven = True
                    self.client.config.pin_upstream_account(response.headers)
                    break
                except Exception as exc:
                    retryable = is_retryable_exception(exc)
                    self._trace_failed_attempt(
                        exchange_id=exchange_id,
                        label=label,
                        attempt=attempt,
                        request=request,
                        response=response,
                        error=exc,
                        previous_response_id=previous_response_id,
                    )
                    if (
                        not retryable
                        or attempt >= self.client.config.max_retries
                    ):
                        raise
                    delay = min(
                        self.client.config.base_retry_delay * (2 ** attempt)
                        + random.uniform(0, 1),
                        self.client.config.max_retry_delay,
                    )
                    if isinstance(exc, AgentEmptyCompletionError):
                        # Nothing upstream is overloaded, so backing off buys
                        # nothing; re-ask right away.
                        delay = self.client.config.base_retry_delay
                    if response is not None:
                        try:
                            retry_after = float(
                                response.headers.get("retry-after", "0")
                            )
                        except (TypeError, ValueError):
                            retry_after = 0
                        delay = max(delay, retry_after)
                    time.sleep(delay)

            if result is None:
                raise AgentClientError("provider call completed without result")
            if successful_response is None:
                raise AgentClientError(
                    "provider call has no successful transport response"
                )
            for event in result.history_events:
                if event.role == "assistant" and event.kind == "message":
                    event.payload.setdefault(
                        "usage", result.usage.to_dict()
                    )
            events = prepared.history_events + result.history_events
            usage_event = HistoryEvent(
                kind="usage",
                role=None,
                payload=result.usage.to_dict(),
            )
            events.append(usage_event)
            for event in events:
                event.exchange_id = exchange_id

            self.client.trace_store.append({
                "record_type": "api_exchange",
                "status": "success",
                "session_id": self.session_id,
                "parent_session_id": self.parent_session_id,
                "exchange_id": exchange_id,
                "label": label,
                "provider": self.client.config.provider.value,
                "model": self.client.config.model,
                "endpoint": self.client.config.endpoint,
                "attempt": retries,
                "http_status": successful_response.status_code,
                "response_headers": _safe_response_headers(
                    successful_response.headers
                ),
                "request": _slim_request(request),
                # Kept as it came off the wire. The protocol audit checks our
                # replay against this rather than against our own normalized
                # copy of it, which would make the check circular.
                "response": successful_response.data,
                "response_text": None,
                "usage": result.usage.to_dict(),
                # The turn's history delta lives on the `trajectory` record
                # below; a second copy here doubled the trace for nothing.
                "previous_response_id": previous_response_id,
                "response_id": result.response_id,
                # Whether this route carries state server-side. An upstream
                # that cannot replays the history inline instead, so the audit
                # has to check the replay rather than the id chain.
                "response_id_continuation": bool(
                    self.client.config.settings.get(
                        "response_id_continuation", True
                    )
                ),
            })

            self._history.extend(events)
            self._provider_history.extend(
                json_copy(prepared.provider_items)
            )
            self._provider_history.extend(
                json_copy(result.provider_history_delta)
            )
            if result.response_id and self.client.config.settings.get(
                "response_id_continuation", True
            ):
                # A route that cannot continue server-side keeps no anchor at
                # all: holding one it will never send would leave the trace
                # recording a continuation the request contradicts.
                self.last_response_id = result.response_id
            self._pending_tool_calls = result.tool_calls
            self._usage.add(result.usage)
            self.client.usage_ledger.add(
                result.usage,
                session_id=self.session_id,
                exchange_id=exchange_id,
            )

            response_obj = AgentResponse(
                text=result.text,
                tool_calls=self.pending_tool_calls,
                reasoning=result.reasoning,
                usage=result.usage,
                raw_response=json_copy(result.raw_response),
                response_id=result.response_id,
                previous_response_id=previous_response_id,
                exchange_id=exchange_id,
                stop_reason=result.stop_reason,
            )
            self._last_response = response_obj
            self._last_meta = {
                "transport_retry_count": retries,
                "effective_max_tokens": _effective_max_tokens(request),
                "api_protocol": self.client.config.provider.value,
                "response_id": result.response_id,
                "previous_response_id": previous_response_id,
                "cache_hit_ratio": result.usage.cache_hit_ratio,
                "elapsed_seconds": time.monotonic() - started,
                "trace_path": (
                    str(self.client.trace_store.path)
                    if self.client.trace_store.path
                    else None
                ),
            }
            self.client.trace_store.append({
                "record_type": "trajectory",
                "session_id": self.session_id,
                "parent_session_id": self.parent_session_id,
                "exchange_id": exchange_id,
                "label": label,
                "provider": self.client.config.provider.value,
                "model": self.client.config.model,
                "prompt": (
                    prepared.history_events[-1].payload.get("content")
                    if prepared.history_events
                    else None
                ),
                "response": result.text,
                # Without this a refused or truncated turn is indistinguishable
                # from the model simply answering with nothing.
                "stop_reason": result.stop_reason,
                "stop_details": (
                    result.raw_response.get("stop_details")
                    if isinstance(result.raw_response, dict)
                    else None
                ),
                "reasoning": [
                    artifact.to_dict() for artifact in result.reasoning
                ],
                "reasoning_len": sum(
                    len(artifact.text or "")
                    for artifact in result.reasoning
                ),
                "usage": result.usage.to_dict(),
                "history_delta": [
                    event.to_dict() for event in events
                ],
                "provider_history_delta": (
                    json_copy(prepared.provider_items)
                    + json_copy(result.provider_history_delta)
                ),
                "previous_response_id": previous_response_id,
                "response_id": result.response_id,
            })
            self.snapshot()
            return response_obj

    def _trace_failed_attempt(
        self,
        *,
        exchange_id: str,
        label: str,
        attempt: int,
        request: dict[str, Any],
        response: TransportResponse | None,
        error: BaseException | None,
        previous_response_id: str | None,
    ) -> None:
        self.client.trace_store.append({
            "record_type": "api_exchange",
            "status": "failed",
            "session_id": self.session_id,
            "parent_session_id": self.parent_session_id,
            "exchange_id": exchange_id,
            "label": label,
            "provider": self.client.config.provider.value,
            "model": self.client.config.model,
            "endpoint": self.client.config.endpoint,
            "attempt": attempt,
            "http_status": response.status_code if response else None,
            "response_headers": (
                _safe_response_headers(response.headers)
                if response
                else {}
            ),
            "request": _slim_request(request),
            # One capped copy of the body. An upstream error page can run to
            # hundreds of KB and used to be stored twice (parsed and raw); what
            # anyone ever reads is the opening lines, and `error` below already
            # holds the type and message.
            "response_text": _clip(response.text if response else None),
            "previous_response_id": previous_response_id,
            "error": (
                {
                    "type": type(error).__name__,
                    "message": str(error),
                    "retryable": is_retryable_exception(error),
                }
                if error
                else None
            ),
        })


def _retryable_status(
    status_code: int,
    message: str = "",
    *,
    auth_proven: bool = False,
) -> bool:
    if _is_proven_auth_failure(status_code, auth_proven):
        return True
    if status_code == 403:
        return _is_transient_403(message)
    return status_code in {408, 409, 429} or 500 <= status_code < 600


_CLIP_AT = 4000


def _clip(text: str | None) -> str | None:
    """A failure body, short enough to keep but long enough to diagnose."""

    if text is None or len(text) <= _CLIP_AT:
        return text
    return f"{text[:_CLIP_AT]}… [截断，原长 {len(text)} 字符]"


# What a replayed request is recorded for: proof that the continuation context
# went back out. Signatures and ids answer that; the prose does not, and every
# call carries the whole conversation again, so the prose is the same text
# written once per turn that follows it.
_REQUEST_KEEP_WHOLE = frozenset({
    "signature", "data", "thoughtSignature", "encrypted_content",
    "id", "tool_use_id", "tool_call_id", "call_id", "name", "type", "role",
    "previous_response_id", "response_id", "model", "prompt_cache_key",
})
_REQUEST_CLIP_AT = 200


def _slim_request(value: Any, key: str | None = None) -> Any:
    """The request as evidence: structure and signatures, not a second copy."""

    if isinstance(value, dict):
        return {k: _slim_request(v, k) for k, v in value.items()}
    if isinstance(value, list):
        return [_slim_request(item, key) for item in value]
    if (
        isinstance(value, str)
        and key not in _REQUEST_KEEP_WHOLE
        and len(value) > _REQUEST_CLIP_AT
    ):
        return f"{value[:_REQUEST_CLIP_AT]}… [省略，原长 {len(value)} 字符]"
    return value


def _response_message(response: TransportResponse) -> str:
    if response.data is not None:
        return str(response.data)[:1000]
    return (response.text or "<empty response>")[:1000]


def _safe_response_headers(
    headers: dict[str, str],
) -> dict[str, str]:
    allowed_exact = {
        "age",
        "cf-cache-status",
        "date",
        "request-id",
        "server-timing",
        "x-cache",
        "x-request-id",
    }
    return {
        key: value
        for key, value in headers.items()
        if key.lower() in allowed_exact
        or key.lower().startswith("x-ratelimit-")
    }


def _effective_max_tokens(request: dict[str, Any]) -> int | None:
    for key in ("max_output_tokens", "max_tokens", "max_completion_tokens"):
        value = request.get(key)
        if isinstance(value, int):
            return value
    generation = request.get("generationConfig")
    if isinstance(generation, dict):
        value = generation.get("maxOutputTokens")
        if isinstance(value, int):
            return value
    return None


def fingerprint(signature: str) -> str:
    """A signature, short enough to record for every fork.

    Forks are recorded so the audit can check that the child replayed the
    parent's continuation signatures. Comparing fingerprints answers that just
    as well as comparing the blobs, and a run forks once per graded question --
    at a few hundred KB of signatures each, the blobs were a quarter of a trace.
    """

    return f"sha256:{hashlib.sha256(signature.encode()).hexdigest()[:16]}"


def _continuation_signatures(
    provider_history: list[dict[str, Any]],
    provider: Provider,
) -> list[str]:
    signatures: list[str] = []
    for value in _walk_values(provider_history):
        if not isinstance(value, dict):
            continue
        if provider is Provider.ANTHROPIC and value.get("type") in {
            "thinking",
            "redacted_thinking",
        }:
            candidate = value.get("signature", value.get("data"))
        elif provider is Provider.GEMINI:
            candidate = value.get("thoughtSignature")
        else:
            candidate = None
        if isinstance(candidate, str) and candidate:
            signatures.append(fingerprint(candidate))
    return signatures


def _walk_values(value: Any):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk_values(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_values(child)
