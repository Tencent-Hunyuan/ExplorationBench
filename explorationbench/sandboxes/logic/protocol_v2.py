"""AlienLogic v2 protocol: fixed calibration demos.

The counterpart to ``sandboxes/code/protocol_v2.py``, and for the same
reason. A free seed phase let every system walk into M0 carrying different
evidence: one model happened to probe the rule that mattered and another did
not, and the gap that produced was read afterwards as a difference in ability.
Fixing the evidence removes that confound -- what a run is measured on is what
it does with the same start.

Exploration itself stays free. After the demos the model writes its own
proofs and the verifier returns its real, opaque verdict on exactly what was
submitted; only the opening evidence is pinned.

The demos are the episode's own seed examples, rendered with the verdict the
verifier actually returns rather than a verdict written down beside them, so
the block cannot drift from the engine that scores the run.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Callable

from sandboxes.logic.protocol_data import REJECTED_DEMOS as _WORLD_REJECTED

PROTOCOL_VERSION = "alienlogic-v2-fixed-demos-2026-09-20"

#: Matched to the code sandbox: four exploration blocks, and a block wide
#: enough to hold a line of enquiry rather than a single shot. A probe here
#: is a whole proof and costs far more than one observed value does there, so
#: the two are aligned on how many independent experiments a block allows,
#: not on what each experiment is worth.
DEFAULT_PROBES_PER_BLOCK = 12
DEFAULT_EXPLORE_BLOCKS = 4


#: Proofs that the verifier turns down, and the whole reason the demo block
#: carries any information at all.
#:
#: The episode's own eight seed examples are all accepted, so a model reading
#: them learns that eight particular proofs are legal and nothing about what
#: is not. Every signal in this sandbox is in the refusals -- a reason_id is
#: the only handle on a hidden side condition -- so a block without one is a
#: block with no evidence in it. That is the likely reason AlienLogic starts
#: high and climbs little where AlienCode starts near zero and climbs far:
#: its opening evidence, unlike a printed output, says nothing.
#:
#: Written fresh rather than lifted from the held-out set, on different
#: propositions, so the block cannot leak an answer. Each one is checked
#: against the live verifier at build time and is only kept while it is still
#: refused for the reason it was written to expose.
REJECTED_DEMOS = _WORLD_REJECTED


def rejected_rows(check: Callable[[str], dict[str, Any]]
                  ) -> list[dict[str, Any]]:
    """The refused demos, dropping any the verifier has stopped refusing.

    A demo that starts being accepted is no longer evidence of anything, and
    silently showing it as a refusal would teach the wrong lesson. Rules move;
    this notices.
    """

    rows = []
    for demo_id, description, proof in REJECTED_DEMOS:
        diagnostic = check(proof.strip())
        if diagnostic.get("accepted"):
            continue
        rows.append({
            "id": demo_id,
            "description": description,
            "premises": [],
            "goal": None,
            "proof": proof.strip(),
            "accepted": False,
            "reason_class": diagnostic.get("reason_class"),
            "reason_id": diagnostic.get("reason_id"),
        })
    return rows


def demo_rows(check: Callable[[str], dict[str, Any]],
              seeds) -> list[dict[str, Any]]:
    """Each seed example with the verdict the live verifier gives it.

    ``check`` takes proof text and returns the diagnostic dict the held-out
    phase would see, so a demo shows exactly the shape of feedback the model
    is about to start working with.
    """

    rows = []
    for example in seeds:
        proof = (example.proof_text or "").strip()
        diagnostic = check(proof) if proof else {}
        rows.append({
            "id": example.id,
            "description": example.description,
            "premises": list(example.premises or []),
            "goal": example.goal,
            "proof": proof,
            "accepted": bool(diagnostic.get("accepted")),
            "reason_class": diagnostic.get("reason_class"),
            "reason_id": diagnostic.get("reason_id"),
        })
    return rows + rejected_rows(check)


def demo_block(check: Callable[[str], dict[str, Any]], seeds) -> str:
    """The fixed evidence, as one block of immutable context."""

    rows = demo_rows(check, seeds)
    refused = sum(1 for row in rows if not row["accepted"])
    parts = [
        "AlienLogic v2 固定标定示例。",
        f"以下 {len(rows)} 个已作答样例是 M0 之前你能看到的全部证据"
        f"（{len(rows) - refused} 个被接受，{refused} 个被拒绝）。"
        "它们对所有系统完全相同，不由模型生成，也不会因为你的回答而改变。"
        "被拒绝的样例给出的诊断与闭卷阶段同样不透明，是你反推隐藏约束的起点。",
    ]
    for row in rows:
        verdict = "ACCEPTED" if row["accepted"] else "REJECTED"
        detail = ""
        if not row["accepted"]:
            detail = (f"\n诊断：{row['reason_class']} / {row['reason_id']}"
                      "（与闭卷阶段一样，不解释具体原因）")
        parts.append(
            f"\n[{row['id']}] {row['description']}\n"
            f"premises: {', '.join(row['premises']) or '（无）'}\n"
            f"goal: {row['goal'] or '（见证明）'}\n"
            "提交的证明：\n"
            f"```alienlogic_proof\n{row['proof']}\n```\n"
            f"verifier 判定：{verdict}{detail}"
        )
    parts.append(
        f"\n看完这些示例后你会有 {DEFAULT_EXPLORE_BLOCKS} 轮自由探索。"
        f"每轮最多提交 {DEFAULT_PROBES_PER_BLOCK} 条证明，"
        "verifier 会对你实际提交的内容返回真实的不透明诊断。"
    )
    return "\n".join(parts)


def manifest(check: Callable[[str], dict[str, Any]], seeds) -> dict[str, Any]:
    """What was shown, hashed, so two runs can be proved to have matched."""

    rows = demo_rows(check, seeds)
    payload = json.dumps(rows, ensure_ascii=False, sort_keys=True)
    return {
        "protocol_version": PROTOCOL_VERSION,
        "demo_count": len(rows),
        "probes_per_block": DEFAULT_PROBES_PER_BLOCK,
        "explore_blocks": DEFAULT_EXPLORE_BLOCKS,
        "demo_digest": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "demos": rows,
    }
