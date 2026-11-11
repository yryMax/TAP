"""Universal TAP: a task-universal adaptation of Tree-of-Attacks-with-Pruning for
indirect prompt injection (IPI) in the AgentDojo simulated environment.

This package searches for a single, shared injection *template* (a `prefix` and a
`suffix` that wrap an arbitrary attacker goal) that transfers across many AgentDojo
task pairs, rather than optimizing one jailbreak per task as the original single-task
``main_TAP.py`` does.

All experiments run against AgentDojo's simulated tools, fictional data and
environment state; nothing here touches a real account or service. See ``AGENT.md``
for the research framing and ``plan.md`` for the implementation plan.

The package depends on ``agentdojo`` (benchmark version v1.2.2, git revision
089ed468cf3ed0322acc66b0211f26d9d90dbf60), ``openai`` (>=1.0) and ``python-dotenv``.
It is expected to run under the reference project's virtualenv, which already pins
those; see ``requirements-universal.txt``.
"""

BENCHMARK_VERSION = "v1.2.2"
AGENTDOJO_REVISION = "089ed468cf3ed0322acc66b0211f26d9d90dbf60"

__all__ = ["BENCHMARK_VERSION", "AGENTDOJO_REVISION"]
