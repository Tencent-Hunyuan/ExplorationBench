"""What each experimental condition is called, in one place.

The run ids stay as they were -- `ctrl_none_...`, `ctrl_passive_...` --
because thousands of artifacts, ledgers and checkpoints are named after
them and dozens of jobs are writing them right now. What changes is the
display name, which is all a reader ever sees.

The old names described the *mechanism* (`none`, `think`, `random`,
`passive`) and needed a legend before they meant anything. These describe
what the system is actually doing, so a table row reads on its own.

Import from either tree::

    from common.arm_names import ARM_EN, ARM_ZH        # dev/
    sys.path.insert(0, repo / 'dev'); from common...   # paper/figures/
"""

from __future__ import annotations

#: Internal key -> the name used in the paper and in English figures.
ARM_EN = {
    'self': 'Autonomous exploration',
    'none': 'Direct answer',
    'think': 'Without-tool answer',
    'random': 'Fixed-probe exploration',
    # The donor is the model's own best trace, picked after the fact --
    # the probes are replayed knowing which attempt turned out best. That
    # is what "hindsight" names; "passive" only said the model was not
    # choosing, which `random` is too.
    'passive': 'Hindsight exploration',
    'oracle': 'Open-book answering',
}

#: The same, for the Chinese dashboard and Chinese-language material.
ARM_ZH = {
    'self': '自主探索',
    'none': '直接作答',
    'think': '无工具作答',
    'random': '固定探针探索',
    'passive': '后验的探索',
    'oracle': '开卷答题',
}

#: Short forms, for axis ticks and heatmap rows where the full name does
#: not fit. Kept distinct at a glance rather than merely truncated.
ARM_SHORT = {
    'self': 'Autonomous',
    'none': 'Direct',
    'think': 'No-tool',
    'random': 'Fixed-probe',
    'passive': 'Hindsight',
    'oracle': 'Open-book',
}

#: The order the paper presents them: the treatment first, then the
#: controls from least to most evidence, then the diagnostic.
ARM_ORDER = ('self', 'none', 'think', 'random', 'passive', 'oracle')

#: The names these conditions used to carry, so an older artifact, figure
#: or note can still be matched to the current one.
LEGACY = {
    'self': 'Autonomous',
    'none': 'No exploration',
    'think': 'Reflection',
    'random': 'Undirected',
    'passive': 'Passive',
    'oracle': 'With oracle',
}


def en(key: str) -> str:
    return ARM_EN.get(key, key)


def zh(key: str) -> str:
    return ARM_ZH.get(key, key)


def short(key: str) -> str:
    return ARM_SHORT.get(key, key)
