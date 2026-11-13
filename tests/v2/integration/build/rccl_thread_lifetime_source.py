"""Extract unchanged native thread/caller functions for the builder-owned proof.

The repository lexer masks comments and literals without changing offsets.
Only complete real function bodies are copied; source text and native field
layout remain the dependency's authority. This module performs no compilation,
allocation-policy choice, runtime loading or accelerator discovery.
"""
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "tests/v2/unit/scripts"))
from check_cpu_prepared_allocation_policy import executable_source


def function(source: str, name: str) -> str:
    """Return one literal native definition, rejecting missing/duplicate functions."""
    code = executable_source(source)
    pattern = (r"(?m)^(?:(?:static|inline)\s+)*"
               r"(?:ncclResult_t\s+|void\s*\*\s*|void\s+)"
               + re.escape(name) + r"\s*\([^;{}]*\)\s*\{")
    matches = list(re.finditer(pattern, code))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one actual native definition: {name}")
    match = matches[0]
    cursor = code.index("{", match.start()) + 1
    depth = 1
    while depth:
        if cursor == len(code):
            raise RuntimeError(f"Unclosed actual native definition: {name}")
        depth += (code[cursor] == "{") - (code[cursor] == "}")
        cursor += 1
    return source[match.start():cursor] + "\n"


def write_inputs(group: Path, init: Path, destination: Path, native: bool) -> None:
    """Copy exact group, cleanup-edge and ignored-error caller bodies for one variant."""
    destination.mkdir(parents=True, exist_ok=True)
    source = group.read_text()
    functions = {
        "actual-worker.inc": "ncclAsyncJobMain",
        "actual-async-launch.inc": "asyncJobLaunch",
        "actual-complete.inc": "ncclAsyncJobComplete",
        "actual-group-abort.inc": "ncclGroupJobAbort",
    }
    if native:
        functions.update({
            "actual-created-drain.inc": "ncclLlaminarAbortAndJoinCreatedJobs",
            "actual-job-destructor.inc": "ncclLlaminarDestroyAsyncJob",
        })
    for filename, name in functions.items():
        (destination / filename).write_text(function(source, name))
    if not native:
        (destination / "actual-destructor.inc").write_text("")
    else:
        (destination / "actual-destructor.inc").write_text(
            function(source, "ncclLlaminarDestroyAsyncJob"))
    cleanup = function(source, "groupCleanup")
    marker = "  /* reset everything */"
    start = cleanup.index("  while (!ncclIntruQueueEmpty(asyncJobsPtr)) {",
                          cleanup.index(marker))
    end = cleanup.index("\n  return;", start)
    (destination / "actual-cleanup.inc").write_text(cleanup[start:end])
    start = cleanup.index("{") + 1
    end = cleanup.index("  struct ncclComm* comm;", start)
    (destination / "actual-cleanup-guard.inc").write_text(cleanup[start:end])
    caller = init.read_text()
    for filename, name in {
        "actual-ready.inc": "ncclCommEnsureReady",
        "actual-set-abort.inc": "setCommAbortFlags",
        "actual-comm-abort.inc": "ncclCommAbort_impl",
    }.items():
        (destination / filename).write_text(function(caller, name))
    start = caller.index("struct ncclCommFinalizeAsyncJob {")
    end = caller.index("\n};", start) + 3
    (destination / "actual-finalize-job.inc").write_text(caller[start:end] + "\n")
