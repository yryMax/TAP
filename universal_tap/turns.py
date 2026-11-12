"""Task-independent message templates and single-turn mutations."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any

GOAL_SLOT = "{goal}"
RENDERERS = ("plain_roles", "minimax_m3_chat")
_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]*$")


class InvalidMutation(ValueError):
    """An operation or its resulting template violates the search constraints."""


def parse_json_object(raw: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    for start, char in enumerate(raw):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(raw, start)
        except ValueError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _strings(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


def _fill(value: Any, goal: str) -> Any:
    if isinstance(value, str):
        return value.replace(GOAL_SLOT, goal)
    if isinstance(value, dict):
        return {key: _fill(item, goal) for key, item in value.items()}
    if isinstance(value, list):
        return [_fill(item, goal) for item in value]
    return value


def validate_messages(messages: Any, max_turns: int) -> list[dict[str, Any]]:
    if not isinstance(messages, list) or not 1 <= len(messages) <= max_turns:
        raise InvalidMutation(f"message count must be between 1 and {max_turns}")
    for turn in messages:
        if not isinstance(turn, dict) or set(turn) != {"role", "stage", "content"}:
            raise InvalidMutation("each turn needs exactly role, stage and content")
        if turn["role"] not in ("user", "assistant", "system", "tool"):
            raise InvalidMutation("invalid role")
        if turn["stage"] not in ("A", "B", "C"):
            raise InvalidMutation("invalid stage")
        content = turn["content"]
        if turn["role"] != "tool":
            if not isinstance(content, str):
                raise InvalidMutation("non-tool content must be a string")
        else:
            if not isinstance(content, dict) or set(content) != {"call", "response"}:
                raise InvalidMutation("tool content needs a paired call and response")
            call = content["call"]
            if not isinstance(call, dict) or set(call) != {"name", "arguments"}:
                raise InvalidMutation("tool call needs name and arguments")
            if not isinstance(call["name"], str) or not _NAME.fullmatch(call["name"]):
                raise InvalidMutation("invalid tool name")
            if not isinstance(call["arguments"], dict):
                raise InvalidMutation("tool arguments must be a JSON object")
            # Parameter names are also XML element names in the M3 renderer.
            def check_keys(value):
                if isinstance(value, dict):
                    for key, item in value.items():
                        if not isinstance(key, str) or not _NAME.fullmatch(key):
                            raise InvalidMutation("invalid tool argument key")
                        check_keys(item)
                elif isinstance(value, list):
                    for item in value:
                        check_keys(item)
            check_keys(call["arguments"])
            if any(GOAL_SLOT in s for s in _strings(call)):
                raise InvalidMutation("the goal slot must be in text or a tool response")
    try:
        json.dumps(messages, ensure_ascii=False, allow_nan=False)
        for text in _strings(messages):
            text.encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise InvalidMutation(f"messages are not serializable: {exc}") from exc
    if sum(s.count(GOAL_SLOT) for s in _strings(messages)) != 1:
        raise InvalidMutation("messages must contain exactly one {goal} slot")
    # Slots in object keys would not be substituted by _fill.
    def check_slot_keys(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if GOAL_SLOT in key:
                    raise InvalidMutation("goal slot cannot be an object key")
                check_slot_keys(item)
        elif isinstance(value, list):
            for item in value:
                check_slot_keys(item)
    check_slot_keys(messages)
    return copy.deepcopy(messages)


@dataclass(frozen=True)
class TurnTemplate:
    messages: list[dict[str, Any]]
    renderer: str = "plain_roles"

    def __post_init__(self):
        if self.renderer not in RENDERERS:
            raise InvalidMutation(f"unknown renderer: {self.renderer}")
        object.__setattr__(self, "messages", validate_messages(self.messages, max(1, len(self.messages))))

    def render(self, goal: str) -> str:
        from .render import render_messages
        return render_messages(_fill(self.messages, goal), self.renderer)

    @property
    def template_hash(self) -> str:
        # Stages are feedback annotations; equal rendered payloads share an evaluation.
        return hashlib.sha256(self.render(GOAL_SLOT).encode("utf-8")).hexdigest()[:16]

    def as_dict(self) -> dict[str, Any]:
        return {"messages": copy.deepcopy(self.messages), "renderer": self.renderer,
                "template_hash": self.template_hash}

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, renderer: str | None = None, max_turns: int = 16):
        stored_renderer = data.get("renderer", "plain_roles")
        if renderer is not None and "renderer" in data and stored_renderer != renderer:
            raise InvalidMutation("seed/payload renderer differs from the fixed run renderer")
        return cls(validate_messages(data["messages"], max_turns), renderer or stored_renderer)


def root_template(renderer: str = "plain_roles") -> TurnTemplate:
    return TurnTemplate([{"role": "user", "stage": "C", "content": GOAL_SLOT}], renderer)


def operation_key(operation: dict[str, Any]) -> str:
    return json.dumps({k: v for k, v in operation.items() if k != "reason"}, sort_keys=True,
                      ensure_ascii=False, allow_nan=False)


def apply_operation(parent: TurnTemplate, operation: dict[str, Any], max_turns: int) -> TurnTemplate:
    if not isinstance(operation, dict):
        raise InvalidMutation("operation must be an object")
    op = operation.get("op")
    fields = {"insert": {"op", "pos", "role", "stage", "content"},
              "rewrite": {"op", "index", "content"}, "delete": {"op", "index"}}
    if op not in fields or set(operation) - {"reason"} != fields[op]:
        raise InvalidMutation("operation has missing or unexpected fields")
    messages = copy.deepcopy(parent.messages)
    index = operation.get("pos" if op == "insert" else "index")
    limit = len(messages) if op == "insert" else len(messages) - 1
    if type(index) is not int or not 0 <= index <= limit:
        raise InvalidMutation("operation index is out of range")
    if op == "insert":
        messages.insert(index, {k: operation[k] for k in ("role", "stage", "content")})
    elif op == "rewrite":
        messages[index]["content"] = operation["content"]
    else:
        if any(GOAL_SLOT in s for s in _strings(messages[index])):
            raise InvalidMutation("cannot delete the goal turn")
        del messages[index]
    return TurnTemplate(validate_messages(messages, max_turns), parent.renderer)


@dataclass
class TurnCandidate:
    candidate_id: str
    template: TurnTemplate
    parent_id: str | None = None
    root_id: str = "root0"
    depth: int = 0
    operation: dict[str, Any] | None = None
    history: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {"candidate_id": self.candidate_id, **self.template.as_dict(),
                "parent_id": self.parent_id, "root_id": self.root_id, "depth": self.depth,
                "op": self.operation, "op_history": self.history,
                "rendered_text": self.template.render(GOAL_SLOT), "invalid_reason": None}

    @classmethod
    def from_dict(cls, data):
        return cls(data["candidate_id"], TurnTemplate.from_dict(data, max_turns=len(data["messages"])),
                   data.get("parent_id"), data.get("root_id", "root0"), data.get("depth", 0),
                   data.get("op"), data.get("op_history", []))
