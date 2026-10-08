from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

from .types import SessionState, UsageRecord, json_copy, utc_now


class JsonlTraceStore:
    """Thread-safe, append-only wire trace plus atomic session snapshots."""

    def __init__(
        self,
        path: str | os.PathLike[str] | None = None,
        *,
        snapshot_dir: str | os.PathLike[str] | None = None,
        fsync: bool = False,
    ) -> None:
        self.path = Path(path).resolve() if path else None
        self.snapshot_dir = (
            Path(snapshot_dir).resolve() if snapshot_dir else None
        )
        self.fsync = fsync
        self._lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._handle: Any = None
        self._sequence = _last_sequence(self.path) if self.path else 0
        self._needs_separator = (
            _needs_separator(self.path) if self.path else False
        )
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.snapshot_dir:
            self.snapshot_dir.mkdir(parents=True, exist_ok=True)

    def append(self, record: dict[str, Any]) -> int:
        with self._lock:
            self._sequence += 1
            sequence = self._sequence
        if not self.path:
            return sequence
        # Encoding runs outside every lock: trace records carry whole
        # conversations, and serialising them under the writer lock turns a
        # wide worker pool into a single-file convoy.
        encoded = json.dumps(
            {
                "schema_version": 2,
                "sequence": sequence,
                "recorded_at": utc_now(),
                **record,
            },
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        )
        with self._write_lock:
            output = self._handle
            if output is None:
                output = self.path.open("a", encoding="utf-8")
                self._handle = output
            if self._needs_separator:
                output.write("\n")
                self._needs_separator = False
            output.write(encoded + "\n")
            output.flush()
            if self.fsync:
                os.fsync(output.fileno())
        return sequence

    def close(self) -> None:
        with self._write_lock:
            handle, self._handle = self._handle, None
        if handle is not None:
            try:
                handle.close()
            except OSError:
                pass

    def save_snapshot(self, state: SessionState) -> Path | None:
        if not self.snapshot_dir:
            return None
        target = self.snapshot_dir / f"{state.session_id}.json"
        tmp = target.with_name(
            f".{target.name}.{os.getpid()}.{threading.get_ident()}.tmp"
        )
        encoded = json.dumps(
            state.to_dict(), ensure_ascii=False, indent=2, default=str
        )
        with tmp.open("w", encoding="utf-8") as output:
            output.write(encoded + "\n")
            output.flush()
            if self.fsync:
                os.fsync(output.fileno())
        os.replace(tmp, target)
        return target


class UsageLedger:
    """Shared usage accumulator for a root session and all of its forks."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._records: list[dict[str, Any]] = []
        self._by_provider: dict[str, UsageRecord] = {}

    def add(
        self,
        usage: UsageRecord,
        *,
        session_id: str,
        exchange_id: str,
    ) -> None:
        with self._lock:
            aggregate = self._by_provider.get(usage.provider)
            if aggregate is None:
                aggregate = UsageRecord(provider=usage.provider)
                self._by_provider[usage.provider] = aggregate
            aggregate.add(usage)
            self._records.append({
                "session_id": session_id,
                "exchange_id": exchange_id,
                "usage": usage.to_dict(),
            })

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            by_provider = {
                key: value.to_dict()
                for key, value in self._by_provider.items()
            }
            prompt = sum(
                value.effective_input_tokens
                for value in self._by_provider.values()
            )
            output = sum(
                value.output_tokens for value in self._by_provider.values()
            )
            reasoning = sum(
                value.reasoning_tokens for value in self._by_provider.values()
            )
            total = sum(
                value.total_tokens for value in self._by_provider.values()
            )
            cache_read = sum(
                value.cache_read_input_tokens
                for value in self._by_provider.values()
            )
            cache_write = sum(
                value.cache_write_input_tokens
                + value.cache_creation_input_tokens
                for value in self._by_provider.values()
            )
            return {
                # Legacy names remain for existing result parsers.
                "prompt_tokens": prompt,
                "completion_tokens": output,
                "reasoning_tokens": reasoning,
                "total_tokens": total,
                "cache_read_input_tokens": cache_read,
                "cache_write_input_tokens": cache_write,
                "cache_hit_ratio": cache_read / prompt if prompt else 0.0,
                "by_provider": by_provider,
                "call_count": len(self._records),
            }

    def records(self) -> list[dict[str, Any]]:
        with self._lock:
            return json_copy(self._records)

    def reset(self) -> None:
        with self._lock:
            self._records.clear()
            self._by_provider.clear()

    def restore_legacy(self, snapshot: dict[str, Any]) -> None:
        """Restore old four-counter checkpoints into a synthetic provider."""

        with self._lock:
            self._records.clear()
            self._by_provider.clear()
            record = UsageRecord(
                provider="legacy_restored",
                input_tokens=int(snapshot.get("prompt_tokens", 0) or 0),
                output_tokens=int(snapshot.get("completion_tokens", 0) or 0),
                reasoning_tokens=int(
                    snapshot.get("reasoning_tokens", 0) or 0
                ),
                total_tokens=int(snapshot.get("total_tokens", 0) or 0),
                cache_read_input_tokens=int(
                    snapshot.get("cache_read_input_tokens", 0) or 0
                ),
                cache_write_input_tokens=int(
                    snapshot.get("cache_write_input_tokens", 0) or 0
                ),
            )
            self._by_provider[record.provider] = record

    def restore_snapshot(self, snapshot: dict[str, Any]) -> None:
        """Restore an aggregate returned by :meth:`snapshot`.

        Per-call records are represented by placeholders because the aggregate
        intentionally does not duplicate the append-only wire trace.
        """

        with self._lock:
            self._records.clear()
            self._by_provider.clear()
            by_provider = snapshot.get("by_provider")
            if isinstance(by_provider, dict) and by_provider:
                for provider, value in by_provider.items():
                    if not isinstance(value, dict):
                        continue
                    record = UsageRecord.from_dict({
                        **json_copy(value),
                        "provider": str(
                            value.get("provider") or provider
                        ),
                    })
                    self._by_provider[record.provider] = record
            else:
                record = UsageRecord(
                    provider="legacy_restored",
                    input_tokens=int(
                        snapshot.get("prompt_tokens", 0) or 0
                    ),
                    output_tokens=int(
                        snapshot.get("completion_tokens", 0) or 0
                    ),
                    reasoning_tokens=int(
                        snapshot.get("reasoning_tokens", 0) or 0
                    ),
                    total_tokens=int(
                        snapshot.get("total_tokens", 0) or 0
                    ),
                    cache_read_input_tokens=int(
                        snapshot.get("cache_read_input_tokens", 0) or 0
                    ),
                    cache_write_input_tokens=int(
                        snapshot.get("cache_write_input_tokens", 0) or 0
                    ),
                )
                self._by_provider[record.provider] = record

            try:
                call_count = max(
                    0, int(snapshot.get("call_count", 0) or 0)
                )
            except (TypeError, ValueError):
                call_count = 0
            self._records.extend({
                "session_id": "restored_snapshot",
                "exchange_id": f"restored_{index + 1}",
                "usage": {},
            } for index in range(call_count))


def _last_sequence(path: Path) -> int:
    if not path.exists() or path.stat().st_size == 0:
        return 0
    try:
        with path.open("rb") as source:
            end = source.seek(0, os.SEEK_END)
            position = end
            buffer = b""
            while position > 0:
                size = min(64 * 1024, position)
                position -= size
                source.seek(position)
                buffer = source.read(size) + buffer
                lines = buffer.splitlines()
                if position == 0 or len(lines) > 1:
                    # Concurrent writers encode outside the writer lock, so the
                    # tail can be slightly out of order; take the high-water
                    # mark rather than trusting the final line.
                    highest = 0
                    for line in reversed(lines):
                        if not line.strip():
                            continue
                        try:
                            value = json.loads(line)
                        except (json.JSONDecodeError, UnicodeDecodeError):
                            continue
                        highest = max(
                            highest, int(value.get("sequence", 0) or 0)
                        )
                    if highest:
                        return highest
                    break
    except (OSError, ValueError, TypeError):
        return 0
    return 0


def _needs_separator(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with path.open("rb") as source:
            source.seek(-1, os.SEEK_END)
            return source.read(1) not in {b"\n", b"\r"}
    except OSError:
        return False
