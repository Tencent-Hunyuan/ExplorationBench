"""
AlienLogic evaluation harness.

Thin wrapper around the internal `harness` (which contains the
generic seed / explore / milestone pipeline). It:
  - swaps the episode catalogue to `episodes` (the unified episode,
    the released world is the public demo episode `public_demo`);
  - sets the probe / explore-loop / token budgets of the protocol;
  - redirects logs and results into `data/logic/{logs,results}/`.

Usage:
  ALIENLOGIC_MODEL=gpt-5.4 ALIENLOGIC_MODEL_SHORT=gpt-5.4 \\
      python sandboxes/logic/run_eval.py --episode public_demo

All CLI args supported by the underlying harness are forwarded.
"""
from __future__ import annotations

import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Default episode override (must precede harness import so that the
# module-level default is captured).
os.environ.setdefault("ALIENLOGIC_EPISODE", "public_demo")

# Credentials and run knobs live in eval.local.toml; loading it exports the
# credentials and endpoints, leaving anything already in the environment alone,
# so a one-off `VAR=... python run_eval.py` still wins. This has to precede the
# harness import, which reads the environment at module level.
from common import local_config  # noqa: E402

_LOCAL_SETTINGS = local_config.load()

from sandboxes.logic import harness as harness
from sandboxes.logic import episodes as episode_catalog

harness.EPISODES = episode_catalog.EPISODES
harness.get_episode = episode_catalog.get_episode
harness.validate_episode = episode_catalog.validate_episode
harness.HeldoutTheorem = episode_catalog.HeldoutTheorem
harness.SeedExample = episode_catalog.SeedExample
harness.EpisodeSpec = episode_catalog.EpisodeSpec

# v2 matches the code sandbox: four exploration blocks, twelve probes each.
# The older shape ran eight shorter blocks, which is a different experiment --
# twice the milestones, two thirds the width -- and the two cannot be read in
# one table. Outside v2 the original numbers stand so earlier runs reproduce.
if harness.PROTOCOL_V2:
    harness.N_EXPLORE_LOOPS = 4
    harness.N_PROBES_PER_LOOP = 1
    harness.MAX_PROBES_PER_ROUND = 12
    harness.MAX_TOTAL_PROBES = 48
else:
    harness.N_EXPLORE_LOOPS = 8
    harness.N_PROBES_PER_LOOP = 1
    harness.MAX_PROBES_PER_ROUND = 8
    harness.MAX_TOTAL_PROBES = 60
# Budgets are env-overridable so ultra-verbose reasoning models (e.g. A20B-High
# burns ~50K reasoning tokens per logic proof) can be given a large enough cap
# to avoid thinking-truncated empty content. Defaults preserve prior behavior.
harness.DEFAULT_MAX_TOKENS = int(os.environ.get("EVAL_DEFAULT_MAX_TOKENS", "16000"))
harness.SUMMARY_MAX_TOKENS = int(os.environ.get("EVAL_SUMMARY_MAX_TOKENS", "24000"))
harness.TEST_MAX_TOKENS = int(os.environ.get("EVAL_TEST_MAX_TOKENS", "16000"))
harness.TEST_RETRY_MAX_TOKENS = int(os.environ.get("EVAL_TEST_RETRY_MAX_TOKENS", "24000"))
harness.LOG_RESPONSE_CHARS = 6000

harness._LOG_DIR = os.path.join(os.path.dirname(_PROJECT_ROOT), "logs", "logic", "run_logs", "live")
harness._OUT_DIR = os.path.join(os.path.dirname(_PROJECT_ROOT), "logs", "logic", "results")

if __name__ == "__main__":
    harness.main()
