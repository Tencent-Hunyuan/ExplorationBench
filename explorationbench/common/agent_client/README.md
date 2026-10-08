# AgentClient

What each vendor's API needs and where it bites. The process around it -- which
five places a new model has to be declared, in what order to verify it, and what
misreads a score -- is in `../../EVALUATION.md`; how to launch a run and what it
leaves behind is in `../../RUNBOOK.md`.

`AgentClient` keeps provider-native reasoning state instead of flattening every
turn into OpenAI-style text messages:

- Claude: the complete assistant content block list is replayed, including
  `thinking`, `redacted_thinking`, and `signature`.
- Gemini: the complete model `Content` is replayed, preserving each
  `thoughtSignature` on its original `Part`.
- OpenAI Responses: each turn sends only new input with
  `previous_response_id`; raw reasoning output items remain in the local audit
  history while OpenAI hosts continuation state.
- Other models: OpenAI-compatible chat-completions history is replayed.

Tools are declared once per session in a single shape and translated per
protocol. Every attempt's `usage` is recorded verbatim, including the cache
counters, so a run can be checked for cache hits and for reasoning actually
being carried rather than silently dropped.

```python
from common.agent_client import AgentClient, ToolResult

client = AgentClient(model="api_azure_openai_gpt-5.6-sol")
session = client.create_session(
    system="Use tools when needed.",
    tools=[{
        "name": "lookup",
        "description": "Look up a value",
        "input_schema": {
            "type": "object",
            "properties": {"key": {"type": "string"}},
            "required": ["key"],
        },
    }],
)

turn = session.send_user("Look up alpha")
if turn.tool_calls:
    turn = session.submit_tool_results([
        ToolResult(
            call_id=turn.tool_calls[0].call_id,
            output={"value": 42},
        )
    ])
```

## Routes

The model id picks the protocol and the upstream. Endpoints below are paths on
one of three bases: `MODEL_EVAL_GATEWAY_A_BASE_URL` (GatewayA),
`MODEL_EVAL_BASE_URL` (platform, `.../v1`) or `MODEL_EVAL_GEMINI_BASE_URL`.
The top tier is what the harness asks for by default.

| model id | protocol | endpoint | top tier |
| --- | --- | --- | --- |
| `api_aws_third_*` (claude) | Anthropic Messages | GatewayA `/model/{model}/invoke` | adaptive + `output_config.effort=max` |
| `api_google_*` (claude) | Anthropic Messages | GatewayA `/v1/publishers/anthropic/models/{model}:streamRawPredict` | adaptive + `output_config.effort=max` |
| `messages/*` (deepseek) | Anthropic Messages | platform `/messages` | `thinking.budget_tokens=60000` |
| `*gemini*` | generateContent | gemini `/v1beta/models/{model}:generateContent` | `thinkingLevel=high` |
| `*gpt-*`, `responses/*` | OpenAI Responses | platform `/responses` | `reasoning.effort=max` |
| `api_ali_*` (qwen) | OpenAI Responses | GatewayA `/compatible-mode/v1/responses` | `reasoning.effort=max` |
| `api_doubao_*` | OpenAI Responses | GatewayA `/api/v3/responses` | `thinking.enabled` + `effort=high` |
| `api_moonshot_*` (kimi) | chat/completions | GatewayA `/v1/chat/completions` | `reasoning_effort=max` |
| `api_xai_*` (grok) | chat/completions | GatewayA `/v1/chat/completions` | `reasoning_effort=high` |
| `gateway_a/*` (deepseek v4.1) | chat/completions | `<redacted-host>/standard/v1/chat/completions` | `reasoning_effort=max` |
| everything else | chat/completions | platform `/chat/completions` | vendor default |

Every GatewayA passthrough routes on the auth query
(`?provider=...&model=...&timeout=...`), not the path; the Anthropic ones also
drop the model from the body, since the path already names it. The `gateway_a/`
prefix is the exception: it is GatewayA's own front door rather than the
evaluation gateway, so it authenticates with `GATEWAY_A_API_KEY` and carries no
query at all.

## Per-vendor gotchas

Each of these was a real failure, not a precaution:

- **Bedrock vs Vertex.** `anthropic-version` differs (`bedrock-2023-05-31` vs
  `vertex-2023-10-16`) and Vertex additionally needs `?api=claude_api`.
- **`messages/` is not a passthrough.** It is the platform's own
  `/v1/messages`, which serves the Anthropic protocol for other vendors. On
  deepseek `cache_control` must be a bare `{"type": "ephemeral"}` -- adding
  Anthropic's `ttl` is rejected -- and the gateway deadline rides in the auth
  query rather than a header.
- **Gemini 3.x takes named thinking levels only.** The `max` every other vendor
  accepts is a 400, and `temperature`/`topP`/`topK` are deprecated fields the
  vendor warns will start failing, so the harness sends neither.
- **Responses continuations can be orphaned.** A `previous_response_id` the
  gateway cannot reach comes back 400; the client rebuilds the full history and
  retries once so the episode survives.
- **doubao and its prompt cache.** Setting `instructions` makes it read no
  cache at all, so the system prompt travels as the leading `input` item to
  stay a cacheable prefix. Its explicit `caching` also refuses any request that
  both continues a cached response and declares `tools`, which is every turn of
  a tool-mode episode, so the run relies on the implicit prefix cache instead.
  Its deadline needs room: a single turn answers in about 200s, but graded
  questions run in parallel and queue on the upstream, so at a 300s cap half of
  one run's calls died at the cap with `upstream timeout ... awaiting headers`.
  The route allows 600; at 1800 the gateway drops the socket instead of waiting.
- **qwen at `max` on long contexts.** The vendor enforces a hard deadline
  around ten minutes and answers 504; no client or gateway timeout overrides
  it, so long episodes need a lower tier.
- **kimi hosts no thinking of its own.** Reasoning comes back as
  `message.reasoning_content` and is only in scope next turn if the assistant
  message goes back byte-for-byte, that field included -- which the chat
  adapter does by replaying the message it parsed. Sampling is fixed upstream
  (`temperature` 1.0, `top_p` 0.95, `n` 1, both penalties 0) and any other
  value is a 400, so the harness sends none of them; `max_tokens` is retired in
  favour of `max_completion_tokens`, one budget covering reasoning and answer.
  Its account pool at the gateway also runs dry under load and answers
  `HTTP 500 无可用账号` until one frees up -- 11 attempts in a row during one
  check -- so the orchestrator gives this model fewer workers.
- **The gateway does not stock every model.** DeepSeek v4.1 Flash answers
  `PlatformNoAvailableAccount` on all four of the gateway's routes -- Responses
  passthrough, chat passthrough, standard chat and standard Responses -- for
  both credential pairs, while v4-flash and v4-pro serve fine from the same
  account. GatewayA's own endpoint has it, so the `gateway_a/` prefix goes there with
  that platform's key. Its Responses passthrough works too but is unusable
  here: the vendor accepts `previous_response_id` with a 200 and then ignores
  it, which would drop the entire explored history without an error. Chat
  Completions replays the history in full, so that is the route. One budget
  covers the thinking and the answer -- at the 4096 default the model spends
  the whole thing thinking and returns nothing at all, which was 3 blank turns
  out of 4 in the dataset check -- and `json_schema` and `parallel_tool_calls`
  are not honoured.
- **grok rejects three sampling knobs outright.** `presence_penalty`,
  `frequency_penalty` and `stop` are a 400 on a reasoning model, and reasoning
  cannot be turned off, so they are a 400 always. Its tiers stop at `high` --
  both `max` and `none` are 400s -- and `max_tokens` is retired for
  `max_completion_tokens`, which caps the visible answer without the reasoning
  spending it. `temperature` above about 1.6 makes the vendor take minutes and
  time out, so the harness pins 1.0. Nothing hosts the thinking here either;
  affinity is the body's `prompt_cache_key`, not a gateway query. Reading its
  usage is another reason to stay on the passthrough: the standard protocol's
  gateway counts the reasoning into `completion_tokens`, inflating it by one to
  two orders of magnitude, while the passthrough reports it separately.
- **A `completed` turn that says nothing.** doubao ends a 200 with a lone
  reasoning item and no message often enough to matter: 8% of one run's graded
  turns, each scored as a blank answer. The client now re-asks such a turn up to
  `max_empty_retries` times (default 2) and only hands over the silence once
  that budget is spent, since a blank answer scores zero but an exception would
  end the episode.
- **A tool call the request never offered.** In a closed-book turn -- no tools,
  but a history full of them -- doubao reissued `run_aliencode` in 35% of one
  run's graded turns, with the answer sitting in the arguments. The protocol
  will not take another turn while that call hangs, so the harness answers it
  with a refusal and asks again; `scripts/check_closed_book.py` replays a real
  session snapshot to check that recovery.

## Persistence

Configure `trace_path` and `snapshot_dir` on `AgentClientConfig` to persist:

- an append-only JSONL record for every HTTP attempt and normalized turn;
- an atomic JSON snapshot containing canonical history, exact provider-native
  history, tool calls/results, reasoning artifacts, continuation IDs, and raw
  usage counters.

Use `AgentSession.fork()` for independent held-out tests. Forks retain the
provider-native history or `previous_response_id` without sharing mutable
conversation state. A session with pending tool calls must first submit every
matching tool result; sending a new user turn or forking at that point is
rejected so the provider continuation cannot be corrupted.

## Verifying a route

Before a full run, both checks below are worth doing on a newly added model.
The first is offline-cheap and covers the protocol; the second uses real
dataset instances, so it is the one that shows whether the cache is actually
being hit.

```bash
python dev/scripts/check_protocol.py --model <model-id>
python dev/scripts/check_dataset_cache.py --model <model-id> --cache --tool-mode
```

Two more, once a run has produced traces. `check_closed_book.py` walks the shape
that costs the most points -- explore with the tool, fork without it, ask for an
answer -- and `--from-snapshot` replays a real fork, which a short synthetic
history rarely provokes. `probe_timeout.py` sends the same demanding turn at
several deadlines, which is how a route's `max_timeout` gets chosen.

```bash
python dev/scripts/check_closed_book.py --model <model-id> [--from-snapshot <session.json>]
python dev/scripts/probe_timeout.py --model <model-id> --from-snapshot <session.json> \
    --timeouts 300,600,900
```

`check_protocol.py` asserts reasoning is returned, the continuation carries
context across turns, a tool call round trips, and the recorded history keeps
message/reasoning/tool_call/tool_result/usage. It writes the trace, a
`messages_history` snapshot and an HTML replay under `logs/protocol_tests/`.

Audit a completed trace:

```bash
python dev/scripts/audit_agent_trace.py --strict path/to/agent_trace.jsonl
```

An opt-in live tool round trip is available in
`dev/scripts/smoke_agent_client.py`; it requires the model-evaluation
credentials in the environment. The Claude Opus 5 passthrough check is:

```bash
export MODEL_EVAL_API_ID=...
export MODEL_EVAL_API_KEY=...
dev/run_opus5_protocol_smoke.sh
```

It enables adaptive thinking, performs a real `tool_use`/`tool_result`
round trip, requires a signed first response, writes the full trace and
`messages_history` snapshot under `logs/protocol_tests/`, and runs the strict
trace auditor automatically.
