"""Turn a finished run's trace into the two files a person actually opens.

Both sandboxes call this on their way out so the reviewable artifacts exist
without anyone remembering to export afterwards: a run's own process is the only
thing that reliably knows which trace belongs to it.

The exporter lives under ``scripts/`` and pulls in the renderer, so it is
imported here at call time rather than at module load -- a harness should not
grow a dependency on the reporting tools just by being imported.
"""

from __future__ import annotations

import os
import sys
import traceback


def export_run_artifacts(trace_path: str, data_dir: str) -> str | None:
    """Write ``<run>.run.json`` and ``<run>.html`` into ``data_dir/exports``.

    Never raises: an evaluation that finished is not going to be failed over a
    report. Returns the export directory, or None if nothing was written.
    """

    if not trace_path or not os.path.exists(trace_path):
        return None

    dev_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if dev_root not in sys.path:
        sys.path.insert(0, dev_root)

    out_dir = os.path.join(data_dir, "exports")
    try:
        from pathlib import Path

        from scripts.export_run import export_one

        print(f"\n[Export] 正在导出完整存档与回放 → {out_dir}")
        exported = export_one(
            Path(trace_path), Path(out_dir), full_tests=False, title=None
        )
        return out_dir if exported else None
    except Exception:  # noqa: BLE001 - reporting must not fail the eval
        print("[Export] 导出失败，可稍后手动运行 scripts/export_run.py：")
        traceback.print_exc(limit=3, file=sys.stdout)
        return None
