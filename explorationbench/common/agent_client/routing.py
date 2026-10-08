from __future__ import annotations

import hashlib
import os
import copy
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlencode

from .types import Provider


# The platform serves the Anthropic Messages protocol for vendors that are not
# Anthropic. This prefix asks for that endpoint specifically: it is the
# platform's own /v1/messages, never the GatewayA passthrough, and the marker
# behind it is what the request body must name.
STANDARD_MESSAGES_PREFIX = "messages/"

# GatewayA's own standard endpoint, which is a different door from the evaluation
# gateway: it authenticates with that platform's `sk-` key instead of the
# id:key pair, speaks plain Chat Completions, and names the model in the body,
# so nothing rides in the auth query. Models the gateway has no account for are
# still reachable here.
GATEWAY_A_STANDARD_PREFIX = "gateway_a/"

# Behind the same key GatewayA also fronts each vendor's native endpoint. A model
# id naming one of these vendors after the prefix -- `gateway_a/ali/qwen3.8-max` --
# goes there instead of the standard door: a different path, and a provider
# query the standard endpoint must not carry.
#
# ``protocol`` names the dialect that path speaks. It is omitted where the
# vendor endpoint is Chat Completions, which is what the standard door serves
# and therefore the assumed default.
# GatewayA's docs give ?timeout=N a 1..600 range, but the gateway accepts and
# honours larger values: probing the standard door with 600, 1200 and 1800 all
# returned HTTP 200. At 600 a Qwen xhigh turn was overrunning the deadline and
# failing nine requests in ten, each burning the full ten minutes before the
# retry, so the ceiling is raised to the largest value seen to work. The 600s
# figure stays the default for routes that never needed more.
GATEWAY_A_STANDARD_MAX_TIMEOUT = 1200.0

GATEWAY_A_VENDOR_ROUTES: dict[str, dict[str, Any]] = {
    "ali": {
        "path": "/compatible-mode/v1/chat/completions",
        "gateway_provider": "ali",
        # The vendor's own deadline, and the one place a longer client timeout
        # buys nothing: past it the upstream answers 504 rather than waiting.
        "max_timeout": 1200.0,
    },
    "deepseek": {
        # The evaluation gateway stocks no account for this vendor and answers
        # PlatformNoAvailableAccount, so its passthrough is reachable only
        # through GatewayA's own door.
        "path": "/v1/responses",
        "gateway_provider": "deepseek",
        "protocol": Provider.OPENAI_RESPONSES,
        "max_timeout": 1200.0,
        # It accepts previous_response_id and then rejects the tool output
        # that follows, so the history has to travel inline for tool use to
        # survive a turn.
        "settings": {"response_id_continuation": False},
    },
    "gateway_b": {
        # A second GatewayB door, chat-completions shaped but a vendor
        # passthrough rather than GatewayA's aggregation endpoint.
        "path": "/openapi/v2/chat/completions",
        "gateway_provider": "gateway_b",
        "max_timeout": 1200.0,
    },
    "azure": {
        # Azure OpenAI passthrough. GatewayA also serves this model from its own
        # aggregation door at /standard/v1/chat/completions, but that flattens
        # the exchange into chat; the vendor's Responses endpoint is the one
        # the reported GPT cohort used, so it is the comparable route.
        "path": "/openai/v1/responses",
        "gateway_provider": "azure",
        "protocol": Provider.OPENAI_RESPONSES,
        "max_timeout": 1200.0,
        # Deliberately no cache_task_id, even though affinity is on by
        # default and spreading looked like the cure for an exhausted pool.
        # This route replays reasoning inline, and that reasoning is
        # ciphertext only the resource that wrote it can read. A task id
        # moves the request to a different account, which answers 400
        # invalid_encrypted_content: the same replay succeeded under the
        # app's default affinity and under account_affinity=disable, and
        # failed under both a stable and a random task id.
        #
        # Losing the reasoning is not a degraded retry, it is a different
        # experiment -- the model would answer a graded question without the
        # thinking the run built up. So the account the ciphertext belongs to
        # is the one to stay on.
        #
        # The body's prompt_cache_key does not move the account: run-level,
        # per-session and absent all replayed the same ciphertext fine. It is
        # OpenAI's prefix cache, nothing more.
        "cache_key_spread": True,
    },
    "aws_third": {
        # Bedrock InvokeModel, same shape as the evaluation gateway's own
        # Bedrock passthrough but behind GatewayA's key. The gateway stocks no
        # account for the newest Claude, which is only reachable here.
        "path": "/model/{model}/invoke",
        "gateway_provider": "aws_third",
        "protocol": Provider.ANTHROPIC,
        "anthropic_version": "bedrock-2023-05-31",
        "max_timeout": 1200.0,
        # Without an affinity key the gateway has nothing to spread requests
        # by, and thirty graded questions at once drained the account pool:
        # Opus 5 answered HTTP 500 PlatformNoAvailableAccount on sixty-nine
        # attempts in one run. Each session already carries its own id, so
        # sending it puts one question on one account and different questions
        # on different ones.
        "cache_task_id": True,
    },
    "anthropic": {
        # Anthropic's own Messages endpoint, fronted by GatewayA. The Bedrock
        # vendor above is stocked per model and Opus 5 has no account there --
        # every request answers PlatformNoAvailableAccount, concurrent or not,
        # while Opus 4.8 on the same key and route answers fine. This door
        # serves it.
        #
        # It is shaped like Anthropic's, not like Bedrock's: the model travels
        # in the body, and the version rides in a header rather than the body.
        "path": "/v1/messages",
        "gateway_provider": "anthropic",
        "protocol": Provider.ANTHROPIC,
        "anthropic_version_header": "2023-06-01",
        "max_timeout": 1200.0,
        "cache_task_id": True,
    },
}

GATEWAY_A_STANDARD_PATH = "/standard/v1/chat/completions"


def gateway_a_vendor(model: str) -> str | None:
    """The GatewayA upstream a model id names, or None for the standard door."""

    if not model.startswith(GATEWAY_A_STANDARD_PREFIX):
        return None
    vendor, _, rest = model[len(GATEWAY_A_STANDARD_PREFIX):].partition("/")
    return vendor if rest and vendor in GATEWAY_A_VENDOR_ROUTES else None


def gateway_a_protocol(model: str) -> "Provider | None":
    """The dialect an GatewayA vendor endpoint speaks, if it is not chat."""

    vendor = gateway_a_vendor(model)
    if not vendor:
        return None
    return GATEWAY_A_VENDOR_ROUTES[vendor].get("protocol")


def gateway_a_path(model: str, wire_model: str = "") -> str:
    """The GatewayA path a model id resolves to: vendor endpoint or standard.

    Bedrock names the model in the URL rather than the body, so the vendor
    path may carry a ``{model}`` placeholder for the wire name.
    """

    vendor = gateway_a_vendor(model)
    if not vendor:
        return GATEWAY_A_STANDARD_PATH
    return GATEWAY_A_VENDOR_ROUTES[vendor]["path"].format(model=wire_model)


def infer_provider(model: str) -> Provider:
    name = model.lower()
    if name.startswith("responses/"):
        return Provider.OPENAI_RESPONSES
    if name.startswith(STANDARD_MESSAGES_PREFIX):
        return Provider.ANTHROPIC
    if name.startswith(GATEWAY_A_STANDARD_PREFIX):
        return gateway_a_protocol(model) or Provider.LEGACY_CHAT
    if name.startswith("openrouter/"):
        return Provider.LEGACY_CHAT
    if responses_route_from_model(model):
        return Provider.OPENAI_RESPONSES
    if "anthropic" in name or "claude" in name:
        return Provider.ANTHROPIC
    if "gemini" in name:
        return Provider.GEMINI
    if (
        "openai" in name
        or "gpt-" in name
        or name.startswith(("gpt", "o1", "o3", "o4"))
    ):
        return Provider.OPENAI_RESPONSES
    return Provider.LEGACY_CHAT


BEDROCK_ROUTE = "bedrock"
VERTEX_ROUTE = "vertex"

# GatewayA fronts the Anthropic Messages protocol on two upstreams. Both keep the
# model out of the body, carrying it in the URL path and the auth query
# instead, but they disagree on the URL shape, the provider name, and the
# anthropic_version the body must declare.
PASSTHROUGH_ROUTES: dict[str, dict[str, Any]] = {
    BEDROCK_ROUTE: {
        "prefix": "api_aws_third_",
        "gateway_provider": "aws_third",
        "anthropic_version": "bedrock-2023-05-31",
        "path": "/model/{model}/invoke",
        "extra_query": {},
    },
    VERTEX_ROUTE: {
        "prefix": "api_google_",
        "gateway_provider": "google",
        "anthropic_version": "vertex-2023-10-16",
        "path": "/v1/publishers/anthropic/models/{model}:streamRawPredict",
        "extra_query": {"api": "claude_api"},
    },
}

# GatewayA also fronts vendors that speak the OpenAI Responses protocol natively.
# Unlike the Anthropic passthrough the model still travels in the body; only
# the endpoint path and the auth query identify the upstream.
# ``cache_task_id`` keys the gateway's own cache affinity. Upstreams that cache
# by prefix on their own side have no use for it and may reject the unknown
# query, so it is opt-in per route.
RESPONSES_ROUTES: dict[str, dict[str, Any]] = {
    "ali": {
        "prefix": "api_ali_",
        "gateway_provider": "ali",
        "path": "/compatible-mode/v1/responses",
        "cache_task_id": True,
    },
    "doubao": {
        "prefix": "api_doubao_",
        "gateway_provider": "doubao",
        "path": "/api/v3/responses",
        "cache_task_id": False,
        # A 1800s deadline makes the gateway drop the socket instead of waiting,
        # which costs a whole client timeout per attempt; 600 is honoured. It
        # needs to be this generous because parallel graded questions queue on
        # the upstream, and at 300 half of one run's calls died at the cap.
        "max_timeout": 1200.0,
    },
}


# And it fronts vendors that speak plain OpenAI Chat Completions. Same shape as
# the Responses table -- the body names the model, the auth query names the
# upstream -- just a different protocol on the other side.
CHAT_ROUTES: dict[str, dict[str, Any]] = {
    "moonshot": {
        "prefix": "api_moonshot_",
        "gateway_provider": "moonshot",
        "path": "/v1/chat/completions",
        "cache_task_id": True,
    },
    "xai": {
        "prefix": "api_xai_",
        "gateway_provider": "xai",
        "path": "/v1/chat/completions",
        # This upstream keys its own cache affinity off the body's
        # prompt_cache_key, and the gateway query is not documented to take an
        # affinity id at all, so sending one risks a rejected unknown param.
        "cache_task_id": False,
        # Well inside what the route accepts (1000s answers fine); a 500k-token
        # prompt returns in about 80s, so nothing here should come close.
        "max_timeout": 1200.0,
    },
}


def responses_route_from_model(model: str) -> str | None:
    """Pick the GatewayA Responses upstream from the model name prefix."""

    for name, spec in RESPONSES_ROUTES.items():
        if model.startswith(spec["prefix"]):
            return name
    return None


def chat_route_from_model(model: str) -> str | None:
    """Pick the GatewayA Chat Completions upstream from the model name prefix."""

    for name, spec in CHAT_ROUTES.items():
        if model.startswith(spec["prefix"]):
            return name
    return None


BEDROCK_ANTHROPIC_VERSION = PASSTHROUGH_ROUTES[BEDROCK_ROUTE][
    "anthropic_version"]
VERTEX_ANTHROPIC_VERSION = PASSTHROUGH_ROUTES[VERTEX_ROUTE][
    "anthropic_version"]
BEDROCK_GATEWAY_PROVIDER = PASSTHROUGH_ROUTES[BEDROCK_ROUTE][
    "gateway_provider"]


def passthrough_base() -> str:
    return os.environ.get("MODEL_EVAL_GATEWAY_A_BASE_URL", "").strip().rstrip("/")


def route_from_model(model: str) -> str | None:
    """Pick the upstream from the GatewayA prefix the model name carries."""

    for name, spec in PASSTHROUGH_ROUTES.items():
        if model.startswith(spec["prefix"]):
            return name
    return None


def route_from_endpoint(endpoint: str | None) -> str | None:
    """Recognise an upstream from a URL the caller supplied directly."""

    if not endpoint:
        return None
    if ":streamRawPredict" in endpoint:
        return VERTEX_ROUTE
    if "/model/" in endpoint and "/invoke" in endpoint:
        return BEDROCK_ROUTE
    return None


def passthrough_model_name(model: str, route: str | None) -> str:
    """Strip the GatewayA route prefix to get the upstream model id."""

    spec = PASSTHROUGH_ROUTES.get(route or "")
    prefix = spec["prefix"] if spec else ""
    if prefix and model.startswith(prefix):
        return model[len(prefix):]
    return model


#: Ids that name the same upstream model behind different doors. A snapshot
#: records the id its run was launched with, and moving a model to another
#: route would otherwise strand every snapshot it had already written -- which
#: is the whole exploration, the part that cannot be re-created. Membership
#: here is a claim that the two ids reach the same weights, so it is written
#: out one pair at a time rather than inferred from the names.
MODEL_ALIASES: tuple[frozenset[str], ...] = (
    frozenset({
        "gateway_a/aws_third/anthropic.claude-opus-5",
        "gateway_a/anthropic/claude-opus-5",
    }),
)


def same_model(left: str, right: str) -> bool:
    """Whether two model ids name the same upstream, route aside."""

    if left == right:
        return True
    return any({left, right} <= group for group in MODEL_ALIASES)


def new_cache_task_id() -> str:
    seed = f"{time.time()}{os.environ.get('MODEL_EVAL_API_ID', '')}"
    return hashlib.md5(seed.encode("utf-8")).hexdigest()


def stable_cache_task_id(model: str) -> str:
    """An affinity key that is identical for every call against one model.

    Gemini's gateway caches a prompt prefix on whichever upstream served it, so
    the key has to be reproducible to get a hit -- unlike the per-run Anthropic
    id, which only needs to be stable within a single run.
    """

    seed = f"{os.environ.get('MODEL_EVAL_API_ID', '')}_{model}"
    return hashlib.md5(seed.encode("utf-8")).hexdigest()


def wire_model_name(model: str, provider: Provider) -> str:
    if provider is Provider.ANTHROPIC and model.startswith(
        STANDARD_MESSAGES_PREFIX
    ):
        return model[len(STANDARD_MESSAGES_PREFIX):]
    if model.startswith(GATEWAY_A_STANDARD_PREFIX):
        # The vendor segment picks the endpoint, so only the trailing id is
        # the name the body carries. This is protocol-independent: GatewayA
        # fronts Responses and Chat behind the same prefix.
        wire = model[len(GATEWAY_A_STANDARD_PREFIX):]
        vendor = gateway_a_vendor(model)
        return wire[len(vendor) + 1:] if vendor else wire
    if provider is Provider.OPENAI_RESPONSES:
        if model.startswith("responses/"):
            return model[len("responses/"):]
        route = responses_route_from_model(model)
        if route:
            return model[len(RESPONSES_ROUTES[route]["prefix"]):]
    if provider is Provider.GEMINI:
        for prefix in (
            "api_naci_default_",
            "api_google_",
            "google/",
        ):
            if model.startswith(prefix):
                return model[len(prefix):]
    if provider is Provider.LEGACY_CHAT:
        route = chat_route_from_model(model)
        if route:
            return model[len(CHAT_ROUTES[route]["prefix"]):]
        if model.startswith("openrouter/"):
            return model[len("openrouter/"):]
        if model.startswith("volc_"):
            return model.rsplit("_", 1)[-1]
    return model


@dataclass(slots=True)
class AgentClientConfig:
    model: str
    provider: Provider | str | None = None
    endpoint: str | None = None
    wire_model: str | None = None
    timeout: float = 3600.0
    max_retries: int = 8
    base_retry_delay: float = 2.0
    max_retry_delay: float = 60.0
    # Some reasoning models answer a valid request with a thinking block and
    # no message at all. Nothing downstream can use that turn, so ask again
    # before handing the caller an empty answer.
    max_empty_retries: int = 2
    headers: dict[str, str] = field(default_factory=dict)
    settings: dict[str, Any] = field(default_factory=dict)
    trace_path: str | None = None
    snapshot_dir: str | None = None
    trace_fsync: bool = False
    #: What the upstream will actually honour, resolved once from the route.
    #: A caller raising the deadline for one call is still bound by it.
    route_timeout_cap: float | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if not self.model:
            raise ValueError("model must be non-empty")
        if self.provider is None:
            self.provider = infer_provider(self.model)
        elif not isinstance(self.provider, Provider):
            self.provider = Provider(self.provider)
        self.settings["passthrough_route"] = self._resolve_passthrough()
        self.settings["responses_route"] = self._resolve_responses_route()
        self.settings["chat_route"] = self._resolve_chat_route()
        if self.wire_model is None:
            self.wire_model = (
                passthrough_model_name(self.model, self.passthrough_route)
                if self.anthropic_passthrough
                else wire_model_name(self.model, self.provider)
            )
        if self.endpoint is None:
            if self.anthropic_passthrough:
                path = PASSTHROUGH_ROUTES[self.passthrough_route]["path"]
                self.endpoint = (
                    passthrough_base() + path.format(model=self.wire_model)
                )
            elif self.responses_route:
                self.endpoint = (
                    passthrough_base()
                    + RESPONSES_ROUTES[self.responses_route]["path"]
                )
            elif self.chat_route:
                self.endpoint = (
                    passthrough_base()
                    + CHAT_ROUTES[self.chat_route]["path"]
                )
            else:
                self.endpoint = default_endpoint(
                    self.provider, self.wire_model, self.model
                )
        if self.max_retries < 0:
            raise ValueError("max_retries cannot be negative")
        if self.max_empty_retries < 0:
            raise ValueError("max_empty_retries cannot be negative")
        cap = None
        if self.responses_route:
            cap = RESPONSES_ROUTES[self.responses_route].get("max_timeout")
        elif self.chat_route:
            cap = CHAT_ROUTES[self.chat_route].get("max_timeout")
        elif gateway_a_vendor(self.model):
            vendor = gateway_a_vendor(self.model)
            spec = GATEWAY_A_VENDOR_ROUTES[vendor]
            cap = spec.get("max_timeout")
            # GatewayA can serve one vendor's model from another supplier's
            # accounts behind the same door. The operator's choice overrides
            # whatever a restored config carried, and lands in settings so the
            # record says who served the run.
            supplier = os.environ.get(
                f"GATEWAY_A_PROVIDER_{vendor.upper()}", "").strip()
            if supplier:
                self.settings["gateway_provider"] = supplier
            # A caller's explicit setting still wins; these only describe what
            # the upstream is known to require.
            for key, value in (spec.get("settings") or {}).items():
                self.settings.setdefault(key, value)
        self.route_timeout_cap = cap
        if cap is None and legacy_route(self.model) == "gateway_a_standard":
            # GatewayA documents its deadline as ?timeout=N with N in 1..600, and
            # abandons the request itself past that. Without a cap the standard
            # door inherited the hour-long default, so a request the gateway had
            # already given up on kept a worker parked for the rest of that
            # hour, and with retries a single question could hold one for most
            # of a day. Two runs wedged that way before this was found.
            cap = GATEWAY_A_STANDARD_MAX_TIMEOUT
            self.route_timeout_cap = cap
        if cap:
            self.timeout = min(self.timeout, float(cap))
        if self.anthropic_passthrough:
            # Account affinity only works when every request in a run carries
            # the same id, so resolve it once instead of per request.
            self.settings.setdefault(
                "cache_task_id",
                os.environ.get("MODEL_EVAL_CACHE_TASK_ID", "").strip()
                or new_cache_task_id(),
            )
        elif (
            self.provider is Provider.GEMINI
            or self.responses_route
            or self.chat_route
        ):
            self.settings.setdefault(
                "cache_task_id",
                os.environ.get("MODEL_EVAL_CACHE_TASK_ID", "").strip()
                or stable_cache_task_id(self.wire_model),
            )

    def _resolve_passthrough(self) -> str | None:
        """Decide which GatewayA upstream, if any, serves this model.

        An explicitly supplied endpoint wins, because the URL is what dictates
        the request shape. Otherwise the GatewayA prefix on the model name picks
        the route, and a bare Anthropic model with a configured gateway falls
        back to Bedrock.
        """

        override = self.settings.get("passthrough_route")
        if override is not None:
            route = str(override) or None
            if route and route not in PASSTHROUGH_ROUTES:
                raise ValueError(
                    f"unknown passthrough route {route!r}; "
                    f"known: {', '.join(sorted(PASSTHROUGH_ROUTES))}"
                )
        elif self.provider is not Provider.ANTHROPIC:
            route = None
        elif self.standard_messages or gateway_a_vendor(self.model):
            # The prefix already named its own endpoint -- the platform's
            # /v1/messages, or GatewayA's door -- so a configured gateway must
            # not pull the request onto Bedrock.
            route = None
        elif self.endpoint:
            route = route_from_endpoint(self.endpoint)
        elif passthrough_base():
            # A configured gateway is what enables passthrough at all; the
            # model prefix only chooses which upstream serves it.
            route = route_from_model(self.model) or BEDROCK_ROUTE
        else:
            route = None
        if route and self.provider is not Provider.ANTHROPIC:
            raise ValueError(
                "the GatewayA passthrough only applies to the Anthropic protocol"
            )
        if route and not self.endpoint and not passthrough_base():
            raise RuntimeError(
                "MODEL_EVAL_GATEWAY_A_BASE_URL is required for the "
                f"{route} passthrough"
            )
        return route

    def _resolve_responses_route(self) -> str | None:
        """Decide whether GatewayA fronts this model's Responses endpoint."""

        override = self.settings.get("responses_route")
        if override is not None:
            route = str(override) or None
        elif self.provider is not Provider.OPENAI_RESPONSES:
            route = None
        else:
            route = responses_route_from_model(self.model)
        if route and route not in RESPONSES_ROUTES:
            raise ValueError(
                f"unknown responses route {route!r}; "
                f"known: {', '.join(sorted(RESPONSES_ROUTES))}"
            )
        if route and self.provider is not Provider.OPENAI_RESPONSES:
            raise ValueError(
                "the GatewayA Responses passthrough only applies to the "
                "OpenAI Responses protocol"
            )
        if route and not self.endpoint and not passthrough_base():
            raise RuntimeError(
                "MODEL_EVAL_GATEWAY_A_BASE_URL is required for the "
                f"{route} Responses passthrough"
            )
        return route

    def _resolve_chat_route(self) -> str | None:
        """Decide whether GatewayA fronts this model's Chat Completions endpoint."""

        override = self.settings.get("chat_route")
        if override is not None:
            route = str(override) or None
        elif self.provider is not Provider.LEGACY_CHAT:
            route = None
        else:
            route = chat_route_from_model(self.model)
        if route and route not in CHAT_ROUTES:
            raise ValueError(
                f"unknown chat route {route!r}; "
                f"known: {', '.join(sorted(CHAT_ROUTES))}"
            )
        if route and self.provider is not Provider.LEGACY_CHAT:
            raise ValueError(
                "the GatewayA Chat passthrough only applies to the Chat "
                "Completions protocol"
            )
        if route and not self.endpoint and not passthrough_base():
            raise RuntimeError(
                "MODEL_EVAL_GATEWAY_A_BASE_URL is required for the "
                f"{route} Chat passthrough"
            )
        return route

    @property
    def passthrough_route(self) -> str | None:
        return self.settings.get("passthrough_route") or None

    @property
    def anthropic_passthrough(self) -> bool:
        return self.passthrough_route is not None

    @property
    def gateway_a_anthropic(self) -> bool:
        """GatewayA's own Bedrock door, which the gateway routes cannot reach."""

        return gateway_a_protocol(self.model) is Provider.ANTHROPIC

    @property
    def anthropic_url_addressed(self) -> bool:
        """The URL names the model, so the body must not.

        Both Bedrock doors work this way and both require the body to declare
        an ``anthropic_version`` instead. GatewayA also fronts Anthropic's own
        Messages endpoint, which is shaped the ordinary way, so the test is
        whether the path has a slot for the model rather than which protocol
        the upstream speaks.
        """

        if self.anthropic_passthrough:
            return True
        vendor = gateway_a_vendor(self.model)
        if not (vendor and self.gateway_a_anthropic):
            return False
        return "{model}" in str(GATEWAY_A_VENDOR_ROUTES[vendor]["path"])

    @property
    def cache_key_spread(self) -> bool:
        """Whether each session should carry its own ``prompt_cache_key``.

        Opt-in per route, like ``cache_task_id``. A vendor that keys its own
        cache off the body and is served from a pool of accounts wants one
        key per question; one that caches by prefix on a single upstream
        wants the run's key left alone.

        Keyed on the GatewayA vendor rather than ``passthrough_route``. That
        one names only the Anthropic doors -- ``_resolve_passthrough``
        returns None for everything else -- while the door this matters for
        is reached by the prefix on the model id.
        """

        vendor = gateway_a_vendor(self.model)
        if not vendor:
            return False
        return bool(GATEWAY_A_VENDOR_ROUTES.get(vendor, {}).get("cache_key_spread"))

    @property
    def responses_route(self) -> str | None:
        return self.settings.get("responses_route") or None

    @property
    def chat_route(self) -> str | None:
        return self.settings.get("chat_route") or None

    @property
    def standard_messages(self) -> bool:
        """Anthropic Messages on the platform endpoint, not via GatewayA."""

        return str(self.model).startswith(STANDARD_MESSAGES_PREFIX)

    @property
    def passthrough_anthropic_version(self) -> str:
        """The ``anthropic_version`` the upstream requires in the body."""

        route = self.passthrough_route
        if route:
            return str(PASSTHROUGH_ROUTES[route]["anthropic_version"])
        vendor = gateway_a_vendor(self.model)
        if vendor and self.gateway_a_anthropic:
            return str(GATEWAY_A_VENDOR_ROUTES[vendor]["anthropic_version"])
        raise RuntimeError("not an Anthropic passthrough route")

    def pin_upstream_account(self, headers: dict[str, str] | None) -> None:
        """Deliberately does nothing; kept so callers need not branch.

        Naming an ``account_id`` used to hold a run on the upstream that served
        its first call, so a prompt cache written there would be read back. The
        cost is that the run then depends on one account staying up: when 256602
        went out of service every request of two Opus runs failed with "指定账号
        不可用" rather than moving elsewhere.

        ``cache_task_id`` buys the same locality without the single point of
        failure. The gateway assigns each task id whichever account is healthiest
        and keeps that task there, so cache still hits within a task while
        different tasks spread across accounts.
        """

        del headers

    @property
    def http_timeout(self) -> float:
        # The gateway enforces its own upstream deadline and answers with 504,
        # so keep the socket open long enough to read that answer. Every GatewayA
        # door does this, not just the ones reached over Responses: on the
        # legacy chat door the socket was closing at the same instant the
        # gateway gave up, so a run saw a bare timeout instead of the 504 that
        # says which side failed, and could not tell a slow turn from a dead
        # endpoint.
        gateway = (self.anthropic_passthrough
                   or bool(self.responses_route)
                   or legacy_route(self.model) == "gateway_a_standard")
        return self.timeout + 60 if gateway else self.timeout

    def http_timeout_for(self, seconds: float | None) -> float:
        """The socket deadline for one call, when it differs from the run's.

        Graded questions and the turns that build the trajectory are not the
        same kind of work. A question that runs out of time is an answer --
        the wrong one -- and holding the socket longer only delays scoring it.
        A milestone summary that runs out of time is not an answer at all: the
        run loses the rule report that milestone exists to collect, and no
        amount of re-asking the held-out set recovers it. So the caller can
        raise the deadline for the second kind without loosening the first.
        """
        if seconds is None:
            return self.http_timeout
        gateway = (self.anthropic_passthrough
                   or bool(self.responses_route)
                   or legacy_route(self.model) == "gateway_a_standard")
        ceiling = self.route_timeout_cap
        capped = min(float(seconds), float(ceiling)) if ceiling else float(seconds)
        return capped + 60 if gateway else capped

    def auth_headers(
        self, cache_task_id: str | None = None
    ) -> dict[str, str]:
        """Headers for one call, optionally under a caller's own task id.

        A session passes its own id so that each graded question is its own
        task: the gateway keeps a question's turns on one account for cache
        locality and spreads different questions across accounts, which is what
        keeps a run alive when one account goes out of service.
        """

        if self.headers:
            headers = {"Content-Type": "application/json"}
            if (self.provider is Provider.ANTHROPIC
                    and not self.anthropic_passthrough):
                headers["anthropic-version"] = str(
                    self.settings.get(
                        "anthropic_version", "2023-06-01"
                    )
                )
            headers.update(self.headers)
            return headers
        route = legacy_route(self.model)
        if route == "gateway_a_standard":
            key = os.environ.get("GATEWAY_A_API_KEY", "").strip()
            if not key:
                raise RuntimeError("GATEWAY_A_API_KEY is required")
            vendor = gateway_a_vendor(self.model)
            if vendor:
                # The vendor endpoints route on this and reject a request
                # without it; the standard one rejects a request with it.
                # The deadline rides here too, and defaults to 60s, which a
                # top-tier reasoning turn overruns.
                spec = GATEWAY_A_VENDOR_ROUTES[vendor]
                query = {
                    "provider": str(self.settings.get(
                        "gateway_provider", spec["gateway_provider"],
                    )),
                    "timeout": str(int(self.timeout)),
                }
                task_id = (cache_task_id or "").strip() or str(
                    self.settings.get("cache_task_id", "")
                ).strip()
                # Opt-in per vendor, like the other route tables: an upstream
                # that caches by prefix on its own side has no use for the key
                # and may reject the unknown query.
                if task_id and spec.get("cache_task_id"):
                    query["cache_task_id"] = task_id
                key += "?" + urlencode(query)
                version = spec.get("anthropic_version_header")
                if version:
                    # Anthropic's own door reads the version from a header;
                    # the Bedrock one reads it from the body and rejects it
                    # here, so this is per vendor rather than per protocol.
                    return {
                        "Content-Type": "application/json",
                        "anthropic-version": str(version),
                        "Authorization": f"Bearer {key}",
                    }
            return {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {key}",
            }
        if route == "openrouter":
            key = os.environ.get("OPENROUTER_API_KEY", "").strip()
            if not key:
                raise RuntimeError("OPENROUTER_API_KEY is required")
            return {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {key}",
            }
        if route == "gateway_c":
            key = os.environ.get("HY_GATEWAY_C_API_KEY", "").strip()
            if not key:
                raise RuntimeError("HY_GATEWAY_C_API_KEY is required")
            return {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {key}",
            }
        if route == "hy3":
            key = os.environ.get("HY3_API_TOKEN", "").strip()
            if not key:
                raise RuntimeError("HY3_API_TOKEN is required")
            return {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {key}",
            }
        if route == "hunyuan":
            key = os.environ.get("HUNYUAN_API_TOKEN", "").strip()
            if not key:
                raise RuntimeError("HUNYUAN_API_TOKEN is required")
            return {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {key}",
            }
        if route == "volc":
            key = os.environ.get("VOLC_ARK_API_KEY", "").strip()
            if not key:
                raise RuntimeError("VOLC_ARK_API_KEY is required")
            return {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {key}",
            }
        api_id = os.environ.get("MODEL_EVAL_API_ID", "").strip()
        api_key = os.environ.get("MODEL_EVAL_API_KEY", "").strip()
        if not api_id or not api_key:
            raise RuntimeError(
                "MODEL_EVAL_API_ID and MODEL_EVAL_API_KEY are required "
                "unless explicit headers are supplied"
            )
        token = f"{api_id}:{api_key}"
        if self.anthropic_passthrough:
            # The model travels in the URL path and this query, never in the
            # body. Bedrock and Vertex differ in the provider name, and Vertex
            # additionally wants api=claude_api.
            spec = PASSTHROUGH_ROUTES[self.passthrough_route]
            query = {
                "provider": str(
                    self.settings.get(
                        "gateway_provider", spec["gateway_provider"]
                    )
                ),
                "model": str(self.wire_model),
            }
            query.update(spec["extra_query"])
            query["timeout"] = str(int(self.timeout))
            task_id = (cache_task_id or "").strip() or str(
                self.settings.get("cache_task_id", "")
            ).strip()
            if task_id:
                query["cache_task_id"] = task_id
            max_retry = self.settings.get("gateway_max_retry")
            if max_retry is not None:
                query["max_retry"] = str(int(max_retry))
            token += "?" + urlencode(query)
            return {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {token}",
            }
        if self.responses_route or self.chat_route:
            # These bodies still name the model, but the gateway routes on this
            # query and caches per account, so the affinity id has to stay put
            # for as long as one task runs.
            spec = (
                RESPONSES_ROUTES[self.responses_route]
                if self.responses_route
                else CHAT_ROUTES[self.chat_route]
            )
            query = {
                "provider": str(
                    self.settings.get(
                        "gateway_provider", spec["gateway_provider"]
                    )
                ),
                "model": str(self.wire_model),
                "timeout": str(int(self.timeout)),
            }
            task_id = (cache_task_id or "").strip() or str(
                self.settings.get("cache_task_id", "")
            ).strip()
            if task_id and spec.get("cache_task_id"):
                query["cache_task_id"] = task_id
            token += "?" + urlencode(query)
            return {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {token}",
            }
        if self.provider is Provider.GEMINI:
            route = str(
                self.settings.get(
                    "gateway_provider",
                    os.environ.get(
                        "MODEL_EVAL_GEMINI_PROVIDER", "naci_default"
                    ),
                )
            )
            query = {
                "provider": route,
                "timeout": str(int(self.timeout)),
                "model": str(self.wire_model),
            }
            if route == "google":
                # The native generateContent passthrough. Without api and
                # dev_mode the gateway will not hand the request to Gemini in
                # its own dialect; usage asks for the token counts we score
                # cache hits from; and cache_task_id is the affinity key, so it
                # has to stay the same across the run for a prefix to be
                # reused at all.
                query["api"] = str(
                    self.settings.get("gemini_api", "gemini_api")
                )
                query["dev_mode"] = "true"
                query["usage"] = "1"
                query["cache_task_id"] = str(
                    self.settings.get("cache_task_id", "")
                )
            token += "?" + urlencode(query)
        elif self.standard_messages:
            # This endpoint takes its deadline from the token, and the default
            # is long enough that a slow turn would sit on a dead socket.
            token += "?" + urlencode({"timeout": str(int(self.timeout))})
        elif (
            self.provider is Provider.LEGACY_CHAT
            and self.model == "api_gateway_b_hy3"
        ):
            # HY3's standard Chat Completions route also reads its upstream
            # deadline from the Authorization token. Without this query the
            # platform silently falls back to 60 seconds.
            token += "?" + urlencode({"timeout": str(int(self.timeout))})
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        }
        if self.provider is Provider.ANTHROPIC:
            headers["anthropic-version"] = str(
                self.settings.get("anthropic_version", "2023-06-01")
            )
        return headers

    def public_dict(self) -> dict[str, Any]:
        """Serializable configuration with credentials deliberately omitted."""

        return {
            "model": self.model,
            "wire_model": self.wire_model,
            "provider": self.provider.value,
            "endpoint": self.endpoint,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "base_retry_delay": self.base_retry_delay,
            "max_retry_delay": self.max_retry_delay,
            "max_empty_retries": self.max_empty_retries,
            "settings": copy.deepcopy(self.settings),
            "trace_path": self.trace_path,
            "snapshot_dir": self.snapshot_dir,
            "trace_fsync": self.trace_fsync,
        }

    @classmethod
    def from_public_dict(
        cls,
        value: dict[str, Any],
        *,
        headers: dict[str, str] | None = None,
    ) -> "AgentClientConfig":
        return cls(
            model=str(value["model"]),
            wire_model=value.get("wire_model"),
            provider=value.get("provider"),
            endpoint=value.get("endpoint"),
            timeout=float(value.get("timeout", 3600)),
            max_retries=int(value.get("max_retries", 8)),
            base_retry_delay=float(value.get("base_retry_delay", 2)),
            max_retry_delay=float(value.get("max_retry_delay", 60)),
            max_empty_retries=int(value.get("max_empty_retries", 2)),
            settings=dict(value.get("settings", {})),
            trace_path=value.get("trace_path"),
            snapshot_dir=value.get("snapshot_dir"),
            trace_fsync=bool(value.get("trace_fsync", False)),
            headers=dict(headers or {}),
        )


def legacy_route(model: str) -> str:
    lower = model.lower()
    if lower.startswith(GATEWAY_A_STANDARD_PREFIX):
        return "gateway_a_standard"
    if lower.startswith("openrouter/"):
        return "openrouter"
    if (
        lower.startswith(("hy3-", "opd-"))
        or model in {
            value
            for value in os.environ.get(
                "HY_GATEWAY_C_MODELS", ""
            ).split(",")
            if value
        }
    ):
        return "gateway_c"
    if lower.startswith("hy3.0"):
        return "hy3"
    if lower.startswith("hunyuan-"):
        return "hunyuan"
    if lower.startswith("volc_"):
        return "volc"
    return "model_eval"


def default_endpoint(
    provider: Provider,
    wire_model: str,
    model: str,
) -> str:
    openai_base = os.environ.get(
        "MODEL_EVAL_BASE_URL", "https://<redacted-endpoint>"
    ).rstrip("/")
    if legacy_route(model) == "gateway_a_standard":
        # GatewayA is chosen by the model prefix rather than by protocol: the
        # same door fronts Responses for one vendor and Chat for another.
        base = os.environ.get(
            "GATEWAY_A_BASE_URL", "https://<redacted-endpoint>"
        ).rstrip("/")
        override = {
            Provider.OPENAI_RESPONSES: "GATEWAY_A_RESPONSES_URL",
            Provider.ANTHROPIC: "GATEWAY_A_MESSAGES_URL",
        }.get(provider, "GATEWAY_A_CHAT_COMPLETIONS_URL")
        return os.environ.get(override, base + gateway_a_path(model, wire_model))
    if provider is Provider.OPENAI_RESPONSES:
        return os.environ.get(
            "MODEL_EVAL_RESPONSES_URL", f"{openai_base}/responses"
        )
    if provider is Provider.LEGACY_CHAT:
        route = legacy_route(model)
        if route == "openrouter":
            return os.environ.get(
                "OPENROUTER_CHAT_URL",
                "https://openrouter.ai/api/v1/chat/completions",
            )
        if route == "gateway_c":
            base = os.environ.get(
                "HY_GATEWAY_C_BASE_URL",
                "https://<redacted-endpoint>",
            ).rstrip("/")
            return f"{base}/chat/completions"
        if route == "hy3":
            return os.environ.get(
                "HY3_CHAT_URL",
                "https://<redacted-endpoint>",
            )
        if route == "hunyuan":
            return os.environ.get(
                "HUNYUAN_CHAT_URL",
                "https://<redacted-endpoint>",
            )
        if route == "volc":
            base = os.environ.get(
                "VOLC_ARK_BASE_URL",
                "https://ark.cn-beijing.volces.com/api/v3",
            ).rstrip("/")
            return f"{base}/chat/completions"
        return os.environ.get(
            "MODEL_EVAL_CHAT_COMPLETIONS_URL",
            f"{openai_base}/chat/completions",
        )
    if provider is Provider.ANTHROPIC:
        return os.environ.get(
            "MODEL_EVAL_ANTHROPIC_MESSAGES_URL", f"{openai_base}/messages"
        )
    # The native generateContent passthrough lives on the GatewayA gateway, while
    # the older naci route is served from the standard host.
    gemini_route = os.environ.get("MODEL_EVAL_GEMINI_PROVIDER", "naci_default")
    gemini_base = os.environ.get(
        "MODEL_EVAL_GEMINI_BASE_URL",
        passthrough_base()
        if gemini_route == "google"
        else "https://<redacted-endpoint>",
    ).rstrip("/")
    return os.environ.get(
        "MODEL_EVAL_GEMINI_GENERATE_CONTENT_URL",
        f"{gemini_base}/v1beta/models/{wire_model}:generateContent",
    )
