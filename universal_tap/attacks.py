"""Frozen-template injection adapter for AgentDojo (plan.md §4, §6).

Given a frozen ``Template`` and an attacker goal, this produces the ``{vector: text}``
injection mapping AgentDojo expects, rendering the *same* prefix/suffix around each
task's goal. It does **no** search: the corpus-level search lives in ``search.py``, and
this adapter only renders and returns a mapping, so running one task pair never
re-optimizes the template (which would degenerate into a single-task attack).

Rendering / escaping (verified against agentdojo v1.2.2):

* AgentDojo substitutes injections into ``environment.yaml`` via a single
  ``str.format(**injections)`` pass, then ``yaml.safe_load`` the result. A vector
  appears inside a double-quoted YAML scalar, e.g. ``subject: "{injection_x}"``.
* ``str.format`` does one pass: braces *inside the substituted value* are inserted
  literally and are not re-interpreted, so raw ``{``/``}`` in the injection are safe
  and must not be doubled.
* The value lands inside double quotes in YAML, so the injection must be escaped as a
  double-quoted YAML string body. ``json.dumps(text)[1:-1]`` does exactly that
  (quotes, backslashes, newlines, control chars) and matches the reference project's
  ``_escape_for_yaml``.

We render into every vector that the ground-truth execution of the user task actually
reads (``BaseAttack.get_injection_candidates``), matching agentdojo's own
``FixedJailbreakAttack``/``important_instructions`` placement.
"""

from __future__ import annotations

import json

from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.attacks.base_attacks import BaseAttack
from agentdojo.base_tasks import BaseInjectionTask, BaseUserTask
from agentdojo.task_suite.task_suite import TaskSuite

from .schemas import Template


def escape_for_yaml(text: str) -> str:
    """Escape text for insertion inside an existing double-quoted YAML string.

    See the module docstring: this is the single escaping layer. Do not also double
    braces — ``str.format`` already inserts the value literally.
    """
    return json.dumps(text, ensure_ascii=False)[1:-1]


class _NullPipeline(BasePipelineElement):
    """Placeholder target pipeline. ``BaseAttack.get_injection_candidates`` only uses
    the suite and canaries, so the attack never actually queries this."""

    name = "universal-null"

    def query(self, query, runtime, env=None, messages=(), extra_args=None):  # pragma: no cover
        raise NotImplementedError("the frozen-template attack does not query a pipeline")


class FrozenTemplateAttack(BaseAttack):
    """Render a frozen template around each injection task's goal.

    This is deliberately a ``BaseAttack`` subclass so it reuses agentdojo's
    ``get_injection_candidates`` (which replays the ground truth with canaries to find
    the vectors the agent reads). It is not registered with agentdojo's attack
    registry: Universal TAP drives the benchmark through its own runner.
    """

    name = "universal_frozen"

    def __init__(
        self,
        task_suite: TaskSuite,
        template: Template,
        target_pipeline: BasePipelineElement | None = None,
    ) -> None:
        super().__init__(task_suite, target_pipeline or _NullPipeline())
        self.template = template

    def rendered_raw(self, goal: str) -> str:
        """The injection text (pre-escaping) the agent is meant to read."""
        return self.template.render(goal)

    def attack(self, user_task: BaseUserTask, injection_task: BaseInjectionTask) -> dict[str, str]:
        raw = self.rendered_raw(injection_task.GOAL)
        escaped = escape_for_yaml(raw)
        return {vector: escaped for vector in self.get_injection_candidates(user_task)}
