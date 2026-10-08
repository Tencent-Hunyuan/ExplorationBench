"""The harness binds its episode types here; the released world is the demo."""

from sandboxes.logic.episodes import (  # noqa: F401
    EPISODES, EpisodeSpec, HeldoutTheorem, SeedExample, get_episode, validate_episode)
