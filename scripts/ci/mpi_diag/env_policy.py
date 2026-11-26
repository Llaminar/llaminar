#!/usr/bin/env python3
"""Shared CTest env-policy parser for the MPI startup diagnostic (PR75/78).

Reviewer-92 item 2: ONE parser, used by both the hook and the diagnostic.
Exact CTest semantics, otherwise clean 125 (never a silent reinterpretation).

Supported (exact):
  ENVIRONMENT             NAME=VALUE   (plain assignment)
  ENVIRONMENT_MODIFICATION NAME=set:value       bind value
  ENVIRONMENT_MODIFICATION NAME=unset:          remove
  ENVIRONMENT_MODIFICATION NAME=reset:          restore the pre-list value
                                                (snapshot taken before any
                                                modification is applied)
  ENVIRONMENT_MODIFICATION NAME=string_append:  append value
  ENVIRONMENT_MODIFICATION NAME=string_prepend: prepend value
  ENVIRONMENT_MODIFICATION NAME=path_list_append:   append to path list
  ENVIRONMENT_MODIFICATION NAME=path_list_prepend:  prepend to path list
  ENVIRONMENT_MODIFICATION NAME=cmakelist:    set value (verbatim, CTest
                                              feeds the right side verbatim)

Anything else (including a missed op or a malformed entry) raises
UnsupportedPolicy -> the caller returns 125 and records it; it is NEVER
silently translated into a different semantic.

FIXTURES_REQUIRED: names denoting setup fixtures. The diagnostic never
executes arbitrary setup, so its presence is ALWAYS unsupported (125) - in
both the hook and the diagnostic. If a project later needs exact fixture
execution, that is a separate reviewed feature.
"""
from __future__ import annotations

from pathlib import Path

SUPPORTED_OPS = ("set", "unset", "string_append", "string_prepend")
# Reviewer-93 item 4: ONLY these four exact ops. path_list_*/cmakelist/reset
# are deliberately unsupported (path-list separator semantics and reset
# ambiguity are not implemented) and fail clean with 125.


class UnsupportedPolicy(ValueError):
    """Clean 125: the requested CTest policy is not implemented exactly."""


def parse_environment_policy(entry: dict, base_env: dict[str, str],
                             report: dict | None = None) -> dict[str, str]:
    """Apply ENVIRONMENT + ENVIRONMENT_MODIFICATION to a copy of base_env.

    `report` (optional) receives {'env_mods_applied': [...], 'fixtures': [...]}
    for the evidence binding. Raises UnsupportedPolicy -> caller returns 125.
    """
    properties = {p["name"]: p["value"] for p in entry.get("properties", [])}
    env = dict(base_env)
    applied: list[str] = []

    # Snapshot the pre-list value of every name that FIXTURES/set may affect so
    # reset: restores it exactly (reviewer-92 item 2).
    pre_values: dict[str, str] = {}

    for item in properties.get("ENVIRONMENT", []):
        item = str(item)
        if "=" not in item:
            raise UnsupportedPolicy(f"malformed ENVIRONMENT entry: {item}")
        name, _, value = item.partition("=")
        if not name:
            raise UnsupportedPolicy(f"empty name in ENVIRONMENT entry: {item}")
        if name not in pre_values:
            pre_values[name] = env.get(name, "")
        env[name] = value
        applied.append(f"ENVIRONMENT {name}=...")

    for modifier in properties.get("ENVIRONMENT_MODIFICATION", []):
        modifier = str(modifier)
        if "=" not in modifier:
            raise UnsupportedPolicy(
                f"malformed ENVIRONMENT_MODIFICATION: {modifier}")
        name, _, rest = modifier.partition("=")
        if ":" not in rest:
            raise UnsupportedPolicy(
                f"malformed ENVIRONMENT_MODIFICATION: {modifier}")
        op, _, value = rest.partition(":")
        if op not in SUPPORTED_OPS:
            raise UnsupportedPolicy(
                f"unsupported ENVIRONMENT_MODIFICATION op '{op}' for {name}")
        if name not in pre_values:
            pre_values[name] = env.get(name, "")
        if op == "set":
            env[name] = value
        elif op == "unset":
            env.pop(name, None)
        elif op == "string_prepend":
            env[name] = value + env.get(name, "")
        elif op == "string_append":
            env[name] = env.get(name, "") + value
        applied.append(f"ENVIRONMENT_MODIFICATION {name}={op}:...")

    fixtures = [str(f) for f in properties.get("FIXTURES_REQUIRED", [])]
    if fixtures:
        raise UnsupportedPolicy(
            "FIXTURES_REQUIRED is unsupported: this diagnostic never "
            f"executes arbitrary setup fixtures: {fixtures}")

    if report is not None:
        report["env_mods_applied"] = applied
        report["fixtures_required"] = fixtures
    return env


def require_fixtures_unsupported(entry: dict) -> list[str]:
    """Same rule used by the hook: FIXTURES_REQUIRED => raise (clean 125)."""
    properties = {p["name"]: p["value"] for p in entry.get("properties", [])}
    fixtures = [str(f) for f in properties.get("FIXTURES_REQUIRED", [])]
    if fixtures:
        raise UnsupportedPolicy(f"FIXTURES_REQUIRED unsupported: {fixtures}")
    return []


# Re-export for callers that previously used apply_env_modification.
def apply_env_modification(env: dict[str, str], modifier: str) -> None:
    """Legacy entry point: parse one modifier against a pre-built env so the
    hook can still accept a caller-supplied image_env without losing exact
    semantics (the pre-mod snapshot is the caller's responsibility)."""
    entry = {"properties": [{"name": "ENVIRONMENT_MODIFICATION",
                             "value": [modifier]}]}
    parsed = parse_environment_policy(entry, env)
    env.clear()
    env.update(parsed)
