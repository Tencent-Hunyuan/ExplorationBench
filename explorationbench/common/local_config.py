"""Load credentials and run parameters from a local TOML file.

The file is developer-owned and never committed. Anything it defines under
``[credentials]`` is exported into the process environment so the existing
env-var based routing in :mod:`common.agent_client.routing` keeps working
unchanged; ``[run]`` and ``[output]`` are returned to the caller.

Real environment variables always win over the file, so a one-off
``MODEL=... python3 ...`` still overrides the config.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


DEV_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = DEV_DIR.parent

CONFIG_FILENAME = "eval.local.toml"
EXAMPLE_FILENAME = "eval.local.example.toml"

# Config key -> environment variable consumed by the client/harness layer.
CREDENTIAL_ENV = {
    "model_eval_api_id": "MODEL_EVAL_API_ID",
    "model_eval_api_key": "MODEL_EVAL_API_KEY",
    # GatewayA's standard endpoint takes its own `sk-` key, not the id:key pair.
    "gateway_a_api_key": "GATEWAY_A_API_KEY",
    "openrouter_api_key": "OPENROUTER_API_KEY",
    "hy_gateway_c_api_key": "HY_GATEWAY_C_API_KEY",
    "hy3_api_token": "HY3_API_TOKEN",
    "hunyuan_api_token": "HUNYUAN_API_TOKEN",
    "volc_ark_api_key": "VOLC_ARK_API_KEY",
}

ENDPOINT_ENV = {
    "base_url": "MODEL_EVAL_BASE_URL",
    "responses_url": "MODEL_EVAL_RESPONSES_URL",
    "chat_completions_url": "MODEL_EVAL_CHAT_COMPLETIONS_URL",
    "gateway_a_chat_completions_url": "GATEWAY_A_CHAT_COMPLETIONS_URL",
    "anthropic_messages_url": "MODEL_EVAL_ANTHROPIC_MESSAGES_URL",
    # Set this to route Claude through the AWS Bedrock passthrough instead.
    "gateway_a_base_url": "MODEL_EVAL_GATEWAY_A_BASE_URL",
    "cache_task_id": "MODEL_EVAL_CACHE_TASK_ID",
    "gemini_base_url": "MODEL_EVAL_GEMINI_BASE_URL",
    "gemini_generate_content_url": "MODEL_EVAL_GEMINI_GENERATE_CONTENT_URL",
    "gemini_provider": "MODEL_EVAL_GEMINI_PROVIDER",
}


class LocalConfigError(RuntimeError):
    pass


@dataclass(slots=True)
class LocalConfig:
    path: Path | None
    run: dict[str, Any] = field(default_factory=dict)
    output: dict[str, Any] = field(default_factory=dict)
    exported_env: list[str] = field(default_factory=list)

    def get(self, key: str, default: Any = None) -> Any:
        return self.run.get(key, default)

    @property
    def output_dir(self) -> Path:
        configured = self.output.get("dir") or "logs/protocol_tests"
        path = Path(str(configured)).expanduser()
        return path if path.is_absolute() else REPO_ROOT / path


def default_config_path() -> Path:
    override = os.environ.get("EVAL_LOCAL_CONFIG", "").strip()
    if override:
        return Path(override).expanduser()
    return DEV_DIR / CONFIG_FILENAME


def example_config_path() -> Path:
    return DEV_DIR / EXAMPLE_FILENAME


def load(
    path: str | os.PathLike[str] | None = None,
    *,
    required: bool = False,
) -> LocalConfig:
    config_path = Path(path).expanduser() if path else default_config_path()
    if not config_path.is_file():
        if required:
            raise LocalConfigError(
                f"missing local config: {config_path}\n"
                f"copy {example_config_path()} and fill in your keys"
            )
        return LocalConfig(path=None)

    try:
        with config_path.open("rb") as source:
            data = tomllib.load(source)
    except tomllib.TOMLDecodeError as exc:
        raise LocalConfigError(f"{config_path}: invalid TOML: {exc}") from exc

    exported: list[str] = []
    exported += _export(data.get("credentials"), CREDENTIAL_ENV, config_path)
    exported += _export(data.get("endpoints"), ENDPOINT_ENV, config_path)
    exported += _export_raw(data.get("env"), config_path)

    return LocalConfig(
        path=config_path,
        run=_section(data, "run", config_path),
        output=_section(data, "output", config_path),
        exported_env=exported,
    )


def _section(data: dict[str, Any], name: str, path: Path) -> dict[str, Any]:
    value = data.get(name, {})
    if not isinstance(value, dict):
        raise LocalConfigError(f"{path}: [{name}] must be a table")
    return dict(value)


def _export(
    section: Any,
    mapping: dict[str, str],
    path: Path,
) -> list[str]:
    if section is None:
        return []
    if not isinstance(section, dict):
        raise LocalConfigError(f"{path}: expected a table, got {type(section)}")
    exported: list[str] = []
    for key, value in section.items():
        env_name = mapping.get(key)
        if env_name is None:
            raise LocalConfigError(
                f"{path}: unknown key {key!r}; "
                f"supported: {', '.join(sorted(mapping))}"
            )
        if _set_env(env_name, value):
            exported.append(env_name)
    return exported


def _export_raw(section: Any, path: Path) -> list[str]:
    """Escape hatch for env vars this module does not model explicitly."""

    if section is None:
        return []
    if not isinstance(section, dict):
        raise LocalConfigError(f"{path}: [env] must be a table")
    return [
        name
        for name, value in section.items()
        if _set_env(str(name), value)
    ]


def _set_env(name: str, value: Any) -> bool:
    text = "" if value is None else str(value).strip()
    if not text or os.environ.get(name, "").strip():
        return False
    os.environ[name] = text
    return True
