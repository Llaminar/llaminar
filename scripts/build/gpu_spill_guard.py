#!/usr/bin/env python3
"""Reject GPU memory spills, including cached kernels and out-of-line HIP callees.

CUDA uses ptxas's native fatal spill warning. HIP has no equivalent diagnostic,
so its compiler launcher checks the final code-object metadata for every bundled
architecture. This check never loads a GPU, changes code generation, or confuses
private storage alone with spilling. ROCm scalar-to-vector register-lane moves
are reported separately when final stack-slot evidence proves no memory spill.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, field
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile


class SpillGuardError(RuntimeError):
    """A spill, unreadable object, or incomplete compiler certificate is fatal."""


@dataclass(frozen=True)
class KernelSpills:
    """One compiled specialization's register-allocation evidence."""

    name: str
    sgpr: int
    vgpr: int
    private_bytes: int = 0
    dynamic_stack: bool = False


def parse_metadata(metadata: str) -> list[KernelSpills]:
    """Read the narrow, canonical YAML projection emitted by llvm-readobj.

    Only kernel-level fields at their exact indentation are interpreted; argument
    names and sizes must never become kernel evidence. Missing / duplicate spill
    fields fail closed. This is deliberately not a general YAML configuration
    parser and needs no build-time Python package beyond the standard library.
    """
    if "amdhsa.kernels: []" in metadata.splitlines():
        return []
    if metadata.splitlines().count("amdhsa.kernels:") != 1:
        raise SpillGuardError("missing or repeated amdhsa.kernels metadata")
    records: list[KernelSpills] = []
    current: dict[str, str] | None = None
    names: set[str] = set()

    def finish() -> None:
        """Seal a kernel only after both independent spill counts are present."""
        if current is None:
            return
        required = {"name", "sgpr_spill_count", "vgpr_spill_count",
                    "private_segment_fixed_size", "uses_dynamic_stack"}
        if set(current) != required:
            raise SpillGuardError(f"incomplete kernel spill metadata: {current}")
        name_parts = shlex.split(current["name"])
        if len(name_parts) != 1 or name_parts[0] in names:
            raise SpillGuardError(f"invalid or duplicate kernel name: {current['name']}")
        for key in required - {"name", "uses_dynamic_stack"}:
            if not re.fullmatch(r"[0-9]+", current[key]):
                raise SpillGuardError(f"invalid {key} for {name_parts[0]}")
        if current["uses_dynamic_stack"] not in ("true", "false"):
            raise SpillGuardError(f"invalid dynamic stack metadata for {name_parts[0]}")
        names.add(name_parts[0])
        records.append(KernelSpills(name_parts[0], int(current["sgpr_spill_count"]),
                                   int(current["vgpr_spill_count"]),
                                   int(current["private_segment_fixed_size"]),
                                   current["uses_dynamic_stack"] == "true"))

    inside = False
    for line in metadata.splitlines():
        if line == "amdhsa.kernels:":
            inside = True
            continue
        if not inside:
            continue
        if line and not line[0].isspace():
            break
        if line.startswith("  - "):
            finish()
            current = {}
        match = re.fullmatch(r"(?:  - |    )\.(name|sgpr_spill_count|vgpr_spill_count|private_segment_fixed_size|uses_dynamic_stack):\s*(.*?)\s*", line)
        if match:
            if current is None or match[1] in current:
                raise SpillGuardError(f"unowned or duplicate kernel field: {line.strip()}")
            current[match[1]] = match[2]
    finish()
    if not records:
        raise SpillGuardError("nonempty kernel metadata contained no complete records")
    return records


def run_tool(command: list[str]) -> str:
    """Capture an inspection tool's output without hiding its failure diagnostic."""
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    if result.returncode:
        raise SpillGuardError(f"{shlex.join(command)} failed: {result.stderr.strip()}")
    return result.stdout


def audit_rocm_object(path: Path, llvm_bin: Path) -> list[KernelSpills]:
    """Check every HIP machine-code target; host-only HIP objects need no evidence.

    Inspect after ccache, not just compiler output: an old cached spilling object
    must fail exactly like a newly compiled one. Temporary extraction is private
    to this invocation, so unrestricted parallel builds share no mutable files.
    """
    sections = run_tool([str(llvm_bin / "llvm-readelf"), "-SW", str(path)])
    if not re.search(r"\]\s+\.hip_fatbin\s", sections):
        # Relocatable device IR is not a final register-allocation certificate.
        if re.search(r"__CLANG_OFFLOAD_BUNDLE__|\.llvm\.offloading", sections):
            raise SpillGuardError("HIP relocatable/LTO object requires a final device-link spill audit")
        return []
    kernels: list[KernelSpills] = []
    with tempfile.TemporaryDirectory(prefix="llaminar-gpu-spills-") as directory:
        root = Path(directory)
        bundle = root / "device.bundle"
        run_tool([str(llvm_bin / "llvm-objcopy"), "--dump-section",
                  f".hip_fatbin={bundle}", str(path), str(root / "host.o")])
        bundler = str(llvm_bin / "clang-offload-bundler")
        targets = run_tool([bundler, "--list", "--type=o", f"--input={bundle}"]).splitlines()
        devices = [target for target in targets if target.startswith(("hip-", "hipv4-"))]
        if not devices or len(set(devices)) != len(devices):
            raise SpillGuardError("HIP bundle has no unique machine-code targets")
        for index, target in enumerate(devices):
            code = root / f"device-{index}.co"
            run_tool([bundler, "--unbundle", "--type=o", f"--targets={target}",
                      f"--input={bundle}", f"--output={code}"])
            notes = json.loads(run_tool([str(llvm_bin / "llvm-readobj"), "--notes",
                                        "--elf-output-style=JSON", str(code)]))
            metadata = [note["AMDGPU Metadata"]
                        for obj in notes for section in obj.get("NoteSections", [])
                        for note in section["NoteSection"]["Notes"]
                        if "AMDGPU Metadata" in note]
            if len(metadata) != 1:
                raise SpillGuardError(f"{target}: missing unique final AMDGPU metadata")
            for kernel in parse_metadata(metadata[0]):
                kernels.append(kernel)
    return kernels


def output_path(command: list[str]) -> Path:
    """Resolve the single CMake compile output, including Ninja response files."""
    arguments: list[str] = []
    for argument in command:
        arguments.extend(shlex.split(Path(argument[1:]).read_text())
                         if argument.startswith("@") else [argument])
    outputs = [arguments[index + 1] for index, argument in enumerate(arguments[:-1])
               if argument in ("-o", "--output-file")]
    outputs.extend(argument.partition("=")[2] for argument in arguments
                   if argument.startswith("--output-file="))
    if len(outputs) != 1 or not outputs[0] or outputs[0] == "-":
        raise SpillGuardError("compiler invocation needs exactly one named object output")
    return Path(outputs[0])


@dataclass(frozen=True)
class StackSlot:
    """One final physical frame slot, classified by LLVM, not guessed from size."""

    kind: str
    size: int


@dataclass
class FunctionAllocation:
    """Union of final frame slots across all compiled targets for one function."""

    slots: list[StackSlot] = field(default_factory=list)
    instances: int = 1


def allocation_evidence(diagnostics: str) -> tuple[dict[str, FunctionAllocation], str]:
    """Extract LLVM's narrow record schema; preserve unrelated diagnostics.

    Early regalloc remarks count SGPR-to-VGPR moves as spills. The final frame
    layout removes those non-memory slots and classifies surviving slots as
    spills, variables or ABI storage, including helpers absent from kernel
    metadata. LLVM emits these YAML records into cached stdout. Never discard
    partial evidence or interpret an unknown slot kind as harmless storage.
    """
    functions: dict[str, FunctionAllocation] = {}

    def consume(match: re.Match[str]) -> str:
        record = match[0]
        fields = dict(re.findall(r"^(Pass|Name|Function):[ \t]*(.*?)[ \t]*$", record, re.M))
        if fields.get("Pass") != "stack-frame-layout":
            return record
        if not fields.get("Function") or fields.get("Name") != "StackLayout":
            raise SpillGuardError("incomplete device allocation record")
        name = shlex.split(fields["Function"])
        if len(name) != 1:
            raise SpillGuardError("invalid device function name")
        entry = functions.setdefault(name[0], FunctionAllocation(instances=0))
        entry.instances += 1
        pending: str | None = None
        for key, value in re.findall(r"^  - (Type|Size):[ \t]*(.*?)[ \t]*$", record, re.M):
            parts = shlex.split(value)
            if len(parts) != 1:
                raise SpillGuardError("invalid device stack slot")
            if key == "Type":
                if pending is not None or parts[0] not in (
                        "Spill", "Fixed", "VariableSized", "Protector", "Variable"):
                    raise SpillGuardError("unknown or incomplete device stack slot type")
                pending = parts[0]
            else:
                if pending is None or not re.fullmatch(r"[0-9]+", parts[0]):
                    raise SpillGuardError("invalid device stack slot size")
                entry.slots.append(StackSlot(pending, int(parts[0])))
                pending = None
        if pending is not None:
            raise SpillGuardError("missing device stack slot size")
        return ""

    remaining = re.sub(r"^--- !\w+\n.*?^\.\.\.\n", consume, diagnostics,
                       flags=re.M | re.S)
    if re.search(r"^Pass:[ \t]*stack-frame-layout[ \t]*$", remaining, re.M):
        raise SpillGuardError("truncated device allocation evidence")
    return functions, remaining


def audit_device_allocations(functions: dict[str, FunctionAllocation],
                             kernels: list[KernelSpills]) -> list[str]:
    """Forbid memory spills; report only proven register-to-register moves.

    Final spill slots are forbidden even in helpers whose callers have zero
    spill counters. Intentional private arrays and ABI argument slots remain
    legal, including when the same function has register-only scalar moves.
    """
    violations = []
    for name, entry in functions.items():
        slots = [slot for slot in entry.slots if slot.kind == "Spill" and slot.size]
        if slots:
            violations.append(f"{name}: {len(slots)} memory spill slots, "
                              f"{sum(slot.size for slot in slots)} bytes")
    reports = []
    for name, required in Counter(kernel.name for kernel in kernels).items():
        if name not in functions or functions[name].instances < required:
            violations.append(f"{name}: missing per-function allocation evidence for {required} targets")
    for kernel in kernels:
        if kernel.sgpr:
            reports.append(f"{kernel.name}: {kernel.sgpr} scalar-to-vector register moves; "
                           "no memory spill")
        if kernel.vgpr:
            # Some AMD architectures can retain vector spills in accumulator
            # registers. The final physical stack, not the early bank counter,
            # is authoritative for whether the movement reaches memory.
            reports.append(f"{kernel.name}: {kernel.vgpr} vector register moves; no memory spill")
    if violations:
        raise SpillGuardError("memory register spills are forbidden; unable to certify:\n  "
                              + "\n  ".join(violations))
    return reports


def preserve_device_diagnostics(command: list[str]) -> list[str]:
    """Keep device-only code-generation flags intact through ccache.

    ccache classifies -Xarch_* as preprocessing-only and can silently remove the
    pair from the real compiler invocation. Its documented per-argument escape
    preserves both the selector and diagnostic flag in compilation and cache
    identity. Direct compiler invocations need no ccache-specific arguments.
    """
    if not any(Path(argument).name == "ccache" for argument in command):
        return command
    result: list[str] = []
    for argument in command:
        if argument == "-Xarch_device" or argument.startswith(
                ("-fsave-optimization-record=", "-foptimization-record-",
                 "-Rpass-analysis=stack-frame-layout", "-mllvm=")):
            result.append("--ccache-skip")
        result.append(argument)
    return result


def main(argv: list[str] | None = None) -> int:
    """Run the canonical compiler/cache chain, then certify its optimized output."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llvm-bin", type=Path, required=True)
    parser.add_argument("--config", required=True)
    # Included in the Ninja command identity so editing this guard rechecks cached
    # objects. It is not another certificate or a way to disable enforcement.
    parser.add_argument("--policy-id", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    output: Path | None = None
    try:
        if not command or args.config not in ("Debug", "Release", "Integration"):
            raise SpillGuardError("missing compiler command or unsupported build configuration")
        output = output_path(command)
        result = subprocess.run(preserve_device_diagnostics(command), text=True,
                                capture_output=True, check=False)
        if result.returncode:
            sys.stdout.write(result.stdout or "")
            sys.stderr.write(result.stderr or "")
            return result.returncode
        if args.config != "Debug":
            functions, diagnostic = allocation_evidence(result.stdout or "")
            sys.stdout.write(diagnostic)
            # The YAML above owns frame evidence. Suppress only its duplicate
            # human-readable remark; unrelated compiler warnings remain visible.
            sys.stderr.write(re.sub(
                r"[^\n]*: remark: \nFunction: .*?\[-Rpass-analysis=stack-frame-layout\]\n"
                r"(?:[ \t]*(?:[0-9]+[ \t]+)?\|[^\n]*\n)*",
                "", result.stderr or "", flags=re.S))
            kernels = audit_rocm_object(output, args.llvm_bin)
            for report in audit_device_allocations(functions, kernels):
                print(f"GPU spill guard: {report}", file=sys.stderr)
        else:
            sys.stdout.write(result.stdout or "")
            sys.stderr.write(result.stderr or "")
        return 0
    except (OSError, ValueError, KeyError, TypeError, SpillGuardError) as error:
        print(f"GPU spill guard: {output or 'compiler'}: {error}", file=sys.stderr)
        # Do not leave a linkable failed object behind. The cached copy is harmless:
        # every future cache hit must pass this same post-compile authority.
        if output is not None:
            output.unlink(missing_ok=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
