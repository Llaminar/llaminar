"""Public exact/read-only CLI boundary for historical analyzer modules.

Internal historical readers and mathematical validators remain importable.
The executable workflow cannot initiate a new generic fit, CV or untouched-seal
transaction; installed generic programs continue to own unmeasured shapes.
"""
from __future__ import annotations

import argparse
import sys


def retired_option(option: str) -> bool:
    """Classify obsolete workflow controls without blocking exact retention."""
    return (option.startswith(("--generic", "--freeze-generic", "--certify-generic",
                               "--fit-", "--sealed-", "--burned-sealed-"))
            or option in {"--policy-json", "--frozen-policy-json", "--plan-sealed-pairs",
                          "--max-leaves", "--min-shape-groups-per-leaf",
                          "--policy-fitter", "--maximum-p95-regret-percent",
                          "--minimum-passing-domain-percent",
                          "--development-input", "--development-timing-sidecar"})


def restrict_analyzer_cli(parser: argparse.ArgumentParser,
                         *, allowed_modes: tuple[str, ...]) -> None:
    """Hide and reject retired controls before reading inputs or allocating work.

    Exact mode still uses the existing parser's complete argument validation.
    CPU exact generation has its dedicated transactional entrypoint, so the
    legacy CPU analyzers expose only explicit adaptation/read validation.
    """
    parser.description = (
        "Exact overlay emission and explicit historical observation readers. "
        "New generic fitting/CV/seal workflows are retired. "
        "Use scripts/refresh_native_quantised_dispatch_tables.sh for collection."
    )
    for action in parser._actions:
        if any(retired_option(option) for option in action.option_strings):
            action.help = argparse.SUPPRESS
    supplied = sys.argv[1:]
    if any(retired_option(token.split("=", 1)[0]) for token in supplied):
        parser.error("generic fitting/CV/seal controls are retired; existing generic runtime policies remain unchanged")
    if "--help" not in supplied and "-h" not in supplied and not any(
            token.split("=", 1)[0] in allowed_modes for token in supplied):
        parser.error("implicit generic generation is retired; select an explicit exact or historical reader mode")


def retired_training_cli(name: str) -> int:
    """Reject a former executable training transaction without loading evidence."""
    message = (f"{name}: generic fitting/CV/seal workflow is retired; use "
               "scripts/refresh_native_quantised_dispatch_tables.sh for exact overlays. "
               "Historical artifact readers remain available.")
    print(message, file=sys.stdout if any(arg in {"-h", "--help"} for arg in sys.argv[1:]) else sys.stderr)
    return 0 if any(arg in {"-h", "--help"} for arg in sys.argv[1:]) else 2
