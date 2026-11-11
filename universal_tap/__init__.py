"""Universal TAP: Tree-of-Attacks-with-Pruning adapted to indirect prompt injection (IPI).

Searches AgentDojo's simulated environment for ONE injection template (a shared
``prefix``/``suffix`` around each task's attacker goal) that transfers across many task
pairs, instead of one jailbreak per task. Everything runs against AgentDojo's simulated
tools and fictional data; see ``AGENT.md`` and ``plan.md``.

Modules: ``llm`` (clients), ``agent`` (AgentDojo rollouts), ``evaluation`` (judge and
aggregation), ``search`` (the TAP loop), ``prompts``, ``datasets`` (task manifests),
``schemas`` (records) and ``records`` (run-directory output).
"""

BENCHMARK_VERSION = "v1.2.2"
AGENTDOJO_REVISION = "089ed468cf3ed0322acc66b0211f26d9d90dbf60"
