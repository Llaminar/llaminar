#!/usr/bin/env python3
"""One driver for the native NEON tile proof: proof -> stack gate -> fused-instruction witness.

It is the one command of the proof's custom build step (phase `build`) and of the Integration CTest (phase
`verify`). Every stage runs and is receipted even after an earlier one failed; the driver exits non-zero when any
stage failed. No stage is optional or skipped.

One analysis per attempt (V1). The stages run in this process. Stage 1 (after `build` in phase build) re-hashes
every binding of this attempt's document - the dependency closure and its symlinks, the target command, the object,
the driver plan and its programs, the disassembler and the re-disassembly, the translation unit, the assembly, the
checker and the committed-count files - re-runs the complete analysis once and requires it equal to the document
(CPUNativeTileRegisterEvidence.verified_analysis() for the seal, unqualified_analysis() for an unqualified analysis
record). The result is frozen (VerifiedAnalysis: only str and bytes; the gate's loop annotations as canonical JSON
text) and handed in memory to the stack gate (CPUNativeTileStackGate.gate_from_analysis()) and the fused witness
(fused_from_analysis()), which neither verify nor analyse again. Nothing crosses an attempt: no file handoff, no
cache between processes, no proof skip. The standalone checker and gate command lines keep their full verification.

Run hygiene (DR-1..DR-3): before the attempt every output of the stages it runs is retired - in phase build the seal
(proof.json) and the unqualified analysis record (proof.analysis.json), in both phases stack-gate.json,
fused-stage.json, driver.json and the stage logs. Each stage's complete output (and the traceback of a failure)
goes to stage-<name>.log, line buffered while it runs; its sha256 is recorded.

Receipt (V3): driver.json is written atomically (temporary file, fsync, rename, directory fsync) first with status
"incomplete" and every stage "planned", again with a stage "running" before it starts and with its outcome after it
ends, and only at the end with "passed" or "failed". No signal is trapped: a killed attempt (CTest timeout, owner
retirement) leaves status "incomplete" with its running stage, which classify() reads as failed (stage unknown) -
never green.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import json
import os
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOOLS = ("CPUNativeTileProofDriver.py", "CPUNativeTileRegisterEvidence.py", "CPUNativeTileStackGate.py",
         "CPUNativeTileStackTable.py", "CPUNativeTileStackBaseline.py", "CPUCodegenEnvelope.py",
         "CPUNativeTileRegisterBudget.json", "CPUNativeTileOrderedAddSites.json", "CPUNativeTileChainCounts.json",
         "CPUNativeTileBoundedMemcpy.json")
STAGES = ("proof", "stack_gate", "fused")


def digest(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def load(name: str):
    spec = importlib.util.spec_from_file_location(name.replace(".py", "").lower(), HERE / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_atomically(path: Path, value: dict) -> None:
    """Replace a JSON file atomically and durably: a reader sees the old or the new receipt, never a torn one."""
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "w") as handle:
        handle.write(json.dumps(value, sort_keys=True, indent=1) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def classify(receipt: dict) -> dict:
    """{"status", "stages"} of a driver receipt as read later: green only for a completed passed attempt; an
    incomplete one (killed) is failed, its running stage unknown and its unstarted stages not run."""
    if receipt.get("status") == "passed" and all(stage.get("status") == "passed"
                                                 for stage in receipt.get("stages", {}).values()):
        return {"status": "passed", "stages": {name: "passed" for name in receipt["stages"]}}
    stages = {}
    for name, stage in receipt.get("stages", {}).items():
        status = stage.get("status")
        stages[name] = {"running": "unknown", "planned": "not-run"}.get(status, status)
    return {"status": "failed", "stages": stages}


def retired(args: argparse.Namespace) -> list[Path]:
    """The outputs this attempt retires before it runs (DR-1)."""
    proof = Path(args.proof)
    paths = [proof.with_name(name) for name in ("stack-gate.json", "fused-stage.json", "driver.json",
                                                "driver.json.tmp")]
    paths += [proof.with_name(f"stage-{name}.log") for name in STAGES]
    if args.phase == "build":
        paths += [proof, proof.with_suffix(".analysis.json")]
    return paths


class Context:
    """One attempt's in-memory state: the checker's arguments, the gate's, and (after stage 1) the frozen analysis."""

    def __init__(self, args: argparse.Namespace, checker_args: list[str]) -> None:
        self.args, self.checker_argv = args, checker_args
        self.verified = None
        self.annotations_text = None
        self.modules: dict = {}

    def module(self, name: str):
        if name not in self.modules:
            self.modules[name] = load(name)
        return self.modules[name]

    def checker_arguments(self, action: str) -> argparse.Namespace:
        plain = self.checker_argv if action == "build" else \
            [argument for argument in self.checker_argv if not argument.startswith("--depfile-target=")]
        return self.module("CPUNativeTileRegisterEvidence.py").parse_arguments([action] + plain)

    def gate_arguments(self) -> argparse.Namespace:
        proof = Path(self.args.proof)
        return argparse.Namespace(proof=self.args.proof, mode=self.args.mode, baseline=self.args.baseline,
                                  roles=self.args.roles, keymap=self.args.keymap,
                                  receipt=str(proof.with_name("stack-gate.json")),
                                  analysis=self.verified is not None and self.verified.kind == "unqualified-analysis")


def stage_proof(context: Context) -> dict:
    """Build (phase build), then this attempt's one verified analysis of the seal or the unqualified record."""
    ev = context.module("CPUNativeTileRegisterEvidence.py")
    workers = ev.physical_workers()  # V2: the per-program analysis workers this attempt admits.
    print(f"analysis workers: {json.dumps(workers, sort_keys=True)}")
    built = None
    if context.args.phase == "build":
        try:
            ev.build(context.checker_arguments("build"))
        except (ValueError, OSError) as error:
            built = str(error)
            print(f"build: {error}")
    arguments = context.checker_arguments("verify")
    proof = Path(context.args.proof)
    analyses: dict = {}
    if proof.exists():
        context.verified = ev.verified_analysis(arguments, analyses)
    elif ev.analysis_path(proof).exists():
        context.verified = ev.unqualified_analysis(arguments, analyses)
    else:
        raise ValueError(f"no sealed proof and no unqualified analysis record of this attempt ({built or 'none'})")
    gate = context.module("CPUNativeTileStackGate.py")
    annotations = gate.loop_annotations(ev, context.module("CPUNativeTileStackBaseline.py"), analyses,
                                        context.verified.data, context.verified.disassembly, context.verified.assembly)
    context.annotations_text = json.dumps(annotations, sort_keys=True, separators=(",", ":"))
    analyses.clear()  # Only the frozen values cross into the next stages.
    if context.verified.kind != ev.SEALED_KIND:
        print("admission failed: unqualified analysis record (diagnostic only)")
        return {"status": "failed", "input": context.verified.kind, "workers": workers,
                "reason": "admission failed: unqualified analysis record"}
    print("sealed proof verified; analysis re-run once and equal to the seal")
    return {"status": "passed", "input": context.verified.kind, "workers": workers}


def stage_gate(context: Context) -> dict:
    gate = context.module("CPUNativeTileStackGate.py")
    arguments = context.gate_arguments()
    if context.verified is None:
        gate.run_gate(arguments, lambda: (_ for _ in ()).throw(ValueError(
            "no verified analysis of this attempt (stage 1 failed structurally)")))
    receipt = gate.run_gate(arguments, lambda: gate.gate_from_analysis(arguments, context.verified,
                                                                      context.annotations_text))
    print(f"input {receipt['input']}, comparison {receipt['comparison']}, status {receipt['status']}, "
          f"{len(receipt['failing'])} programs with new stack traffic")
    for name in receipt["failing"]:
        print(f"{name}: " + "; ".join(receipt["results"][name]["failures"][:3]))
    return {"status": receipt["status"], "input": receipt["input"], "comparison": receipt["comparison"]}


def stage_fused(context: Context) -> dict:
    ev = context.module("CPUNativeTileRegisterEvidence.py")
    if context.verified is None:
        receipt = Path(context.args.proof).with_name("fused-stage.json")
        receipt.write_text(json.dumps({"stage": "fused", "status": "failed", "input": None, "verified": False,
                                       "reason": "no verified analysis of this attempt"}, indent=1) + "\n")
        raise ValueError("no verified analysis of this attempt (stage 1 failed structurally)")
    result = ev.fused_from_analysis(context.checker_arguments("fused"), context.verified)
    print(f"fused witness {result['status']} on {result['input']}")
    return {"status": result["status"], "input": result["input"]}


STAGE_FUNCTIONS = {"proof": stage_proof, "stack_gate": stage_gate, "fused": stage_fused}


def run_stage(name: str, function, context: Context, directory: Path) -> dict:
    """Run one in-process stage with its complete output in stage-<name>.log; any exception fails the stage."""
    log = directory / f"stage-{name}.log"
    started = time.time()
    with open(log, "w", buffering=1) as handle, contextlib.redirect_stdout(handle), contextlib.redirect_stderr(handle):
        print(f"$ stage {name}: {json.dumps(sys.argv)}")
        try:
            outcome = function(context)
        except (Exception, SystemExit) as error:  # A stage's failure (argparse exits too) is recorded, not propagated;
            # KeyboardInterrupt and signals are not caught: a cancelled attempt stays incomplete (V3).
            traceback.print_exc()
            outcome = {"status": "failed", "reason": f"{type(error).__name__}: {error}"[:3000]}
    return {**outcome, "seconds": round(time.time() - started, 3), "log": log.name, "log_sha256": digest(log)}


def drive(args: argparse.Namespace, checker_args: list[str], functions: dict | None = None) -> int:
    """Retire this attempt's outputs, write the incomplete receipt, run every stage (receipt updated before and
    after each), write the final receipt, return the aggregate exit status (1 when any stage failed)."""
    functions = functions or STAGE_FUNCTIONS
    directory = Path(args.proof).parent
    directory.mkdir(parents=True, exist_ok=True)
    for path in retired(args):
        path.unlink(missing_ok=True)
    tools = {name: digest(HERE / name) for name in TOOLS}
    tools.update({f"input:{key}": digest(Path(getattr(args, key))) for key in ("baseline", "roles", "keymap")})
    receipt = {"phase": args.phase, "mode": args.mode, "argv": sys.argv, "checker_argv": checker_args,
               "attempt": os.urandom(16).hex(), "tools_sha256": tools, "status": "incomplete",
               "stages": {name: {"status": "planned"} for name in STAGES}}
    path = directory / "driver.json"
    write_atomically(path, receipt)
    context = Context(args, checker_args)
    for name in STAGES:
        receipt["stages"][name] = {"status": "running", "started": time.time()}
        write_atomically(path, receipt)
        receipt["stages"][name] = run_stage(name, functions[name], context, directory)
        write_atomically(path, receipt)
    proof = Path(args.proof)
    receipt["stage1_document"] = "sealed-proof" if proof.exists() else \
        "unqualified-analysis" if proof.with_suffix(".analysis.json").exists() else None
    receipt["status"] = "passed" if all(stage["status"] == "passed" for stage in receipt["stages"].values()) \
        else "failed"
    write_atomically(path, receipt)
    for name, stage in receipt["stages"].items():
        print(f"tile proof stage {name}: {stage['status']} ({stage['seconds']} s)"
              + (f": {stage['reason']}" if stage.get("reason") else ""))
    return 0 if receipt["status"] == "passed" else 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("build", "verify"))
    for option in ("proof", "mode", "baseline", "roles", "keymap"):
        parser.add_argument(f"--{option}", required=True)
    argv = sys.argv[1:]
    if "--" not in argv:
        parser.error("the checker's arguments follow `--`")
    split = argv.index("--")
    args = parser.parse_args(argv[:split])
    raise SystemExit(drive(args, argv[split + 1:]))


if __name__ == "__main__":
    main()
