#!/usr/bin/env python3
"""Reject unaccounted CPU preparation and named workspace construction calls.

Source guards complement the runtime PMA admission and lifetime tests.

This deliberately bounded policy checks each retained preparation call. It is
not C++ semantic analysis: immutable intra-function aliases are supported, and
exact normalization, forwarding and workspace-fixture boundaries describe
existing infrastructure.
Unknown provenance fails closed. It neither bans generic C++ allocation nor
claims to certify allocator behavior, concurrency, or last-owner retirement.
"""
from __future__ import annotations
import argparse
import bisect
from dataclasses import dataclass
from pathlib import Path
import re
import sys
from check_gpu_blocking_sync_policy import (function_intervals, enclosing_caller,
                                            strip_comments_and_literals)

SUFFIXES = {".cpp", ".h", ".hpp", ".inc", ".cu", ".cuh", ".hip"}
PREPARATION = {"prepareGemmHandleLocal": 4, "prepareExpertGemmLocal": 3,
               "createGemmSlicedLocal": 3}
ROLES = ("src/v2/execution/", "src/v2/loaders/", "src/v2/planning/", "src/v2/models/")
TOKEN = re.compile(r"[A-Za-z_]\w*|::|->|&&|\|\||==|!=|<=|>=|[^\s]")
RAW = re.compile(r'(?:u8|u|U|L)?R"([^ ()\\\t\r\n]{0,16})\(')
NUMBER = re.compile(r"\b(?:0[xX][0-9a-fA-F]+|[0-9]+)(?:'[0-9a-fA-F]+)+")

# Cheap candidate filter only; executable tokens still decide every finding.
WORKSPACE_CONSTRUCTION = re.compile(
    r"\b(?:DeviceWorkspaceManager|WorkspaceAllocator)\s*(?:[({]|[A-Za-z_]\w*\s*(?:[(={;,]|\[))|"
    r"\b(?:make_unique|make_shared)\s*<[^;{}]*\b(?:DeviceWorkspaceManager|WorkspaceAllocator)\b|"
    r"\bnew\s+(?:::\s*)?(?:[A-Za-z_]\w*\s*::\s*)*(?:DeviceWorkspaceManager|WorkspaceAllocator)\b")

@dataclass(frozen=True)
class Token:
    text: str
    offset: int

@dataclass(frozen=True)
class Violation:
    path: str
    line: int
    caller: str
    message: str
    def __str__(self) -> str:
        return f"{self.path}:{self.line}: {self.caller}: {self.message}"

def executable_source(raw: str) -> str:
    """Reuse the source lexer after protecting raw strings and digit separators."""
    chars = list(raw)
    # Locate raw strings while respecting ordinary comments/literals: the
    # shared lexer masks a raw opener's first quote, but preserves its prefix R.
    index = 0
    while index < len(raw):
        if raw.startswith("//", index):
            end = raw.find("\n", index + 2)
            index = len(raw) if end < 0 else end
        elif raw.startswith("/*", index):
            end = raw.find("*/", index + 2)
            index = len(raw) if end < 0 else end + 2
        elif match := RAW.match(raw, index):
            end = raw.find(")" + match.group(1) + '"', match.end())
            end = len(raw) if end < 0 else end + len(match.group(1)) + 2
            for cursor in range(index, end):
                if chars[cursor] != "\n": chars[cursor] = " "
            index = end
        elif match := NUMBER.match(raw, index):
            for cursor in range(index, match.end()):
                if chars[cursor] == "'": chars[cursor] = "_"
            index = match.end()
        elif raw[index] in "\"'":
            quote = raw[index]
            index += 1
            while index < len(raw):
                if raw[index] == "\\": index += 2
                elif raw[index] == quote:
                    index += 1
                    break
                else: index += 1
        else: index += 1
    return strip_comments_and_literals("".join(chars))

def text(tokens: list[Token]) -> str:
    return "".join(token.text for token in tokens)

def closing(tokens: list[Token], start: int) -> int:
    """Match actual C++ grouping delimiters, excluding already-masked literals."""
    pairs = {"(": ")", "[": "]", "{": "}"}
    stack: list[str] = []
    for index in range(start, len(tokens)):
        token = tokens[index].text
        if token in pairs: stack.append(pairs[token])
        elif stack and token == stack[-1]:
            stack.pop()
            if not stack: return index
    raise ValueError("unbalanced source grouping at guarded call")

def arguments(tokens: list[Token], opening: int) -> tuple[list[list[Token]], int]:
    end = closing(tokens, opening)
    result: list[list[Token]] = []
    start = opening + 1
    index = start
    while index < end:
        value = tokens[index].text
        if value in ("(", "[", "{"):
            index = closing(tokens, index)
        elif value == ",":
            result.append(tokens[start:index])
            start = index + 1
        index += 1
    if start < end: result.append(tokens[start:end])
    return result, end

def accounted(expression: list[Token]) -> bool:
    """Accept a complete accounted constructor, never a conditional substring."""
    while expression and expression[0].text == "(" and closing(expression, 0) == len(expression)-1:
        expression = expression[1:-1]
    values = [token.text for token in expression]
    try: index = values.index("accounted")
    except ValueError: return False
    if index < 2 or values[index-2:index] != ["CPUWeightStoragePlacement", "::"]:
        return False
    if any(value != "::" and not re.fullmatch(r"[A-Za-z_]\w*", value)
           for value in values[:index-2]): return False
    if index+1 >= len(values) or values[index+1] != "(": return False
    args, end = arguments(expression, index+1)
    return end == len(expression)-1 and len(args) == 4

def assignments(body: list[Token], variable: str) -> list[tuple[list[Token], list[Token]]]:
    """Collect every assignment, including assignments hidden in fallback arms."""
    result = []
    for index, token in enumerate(body[:-1]):
        if token.text != variable or body[index+1].text != "=": continue
        left = index-1
        while left >= 0 and body[left].text not in (";", "{", "}"): left -= 1
        end = index+2
        while end < len(body) and body[end].text != ";":
            if body[end].text in ("(", "[", "{"): end = closing(body, end)
            end += 1
        result.append((body[left+1:index], body[index+2:end]))
    return result

def placement_owned(expression: list[Token], body: list[Token], path: str,
                    caller: str, seen: frozenset[str] = frozenset()) -> bool:
    """Follow immutable aliases or one reviewed mutable normalization boundary."""
    if accounted(expression): return True
    if len(expression) != 1: return False
    variable = expression[0].text
    if variable in seen: return False
    writes = assignments(body, variable)
    if len(writes) == 1 and "const" in [token.text for token in writes[0][0]]:
        return placement_owned(writes[0][1], body, path, caller, seen | {variable})
    if variable != "placement" or len(writes) != 2: return False
    initial, final = writes[0][1], writes[1][1]
    if not accounted(final): return False
    compact = text(body)
    # Only this CPU/GPU binding boundary can conditionally upgrade local policy;
    # the CPU arm rejects a missing model issuer before entering preparation.
    if (path, caller) == ("src/v2/loaders/PreparedWeightStore.cpp", "PreparedWeightStore::prepareGemm"):
        expected = ("autoplacement=CPUWeightStoragePlacement::local();"
            "if(device.is_cpu()){if(constautoauthority=physicalMemoryAuthority())placement="+
            text(final)+";elsethrowstd::logic_error();}")
        return text(initial) == "CPUWeightStoragePlacement::local()" and expected in compact
    # The optional initial placement is assigned only by the accounted-field
    # rule below. Stage entrypoints require their admitted prepared store before
    # reaching the second normalization arm. Calls still check this exact value.
    if (path, caller) == ("src/v2/execution/moe/MoEExpertWeightService.cpp", "MoEExpertWeightService::prepareGemmEngines"):
        expected = ("autoplacement="+text(initial)+
            ";if(!ctx.initial_cpu_storage_placement&&ctx.prepared_store)placement="+text(final)+";")
        return text(initial).startswith("ctx.initial_cpu_storage_placement.value_or(") and expected in compact
    return False

def weight_policy(expression: list[Token], body: list[Token], seen: frozenset[str] = frozenset()) -> bool:
    """Recognize only explicit local weight-policy declarations and their aliases."""
    if "CPUWeightStoragePlacement::" in text(expression): return True
    if len(expression) != 1 or expression[0].text in seen: return False
    variable = expression[0].text
    return any(weight_policy(value,body,seen | {variable}) for _,value in assignments(body,variable))

def construction(tokens: list[Token], index: int) -> tuple[list[list[Token]] | None, int] | None:
    """Recognize named construction; a None argument list marks an array.

    Named workspace arrays are unsupported: require individual accounted owners
    instead of attempting to interpret element initialization or array extent.
    """
    next_index=index+1
    if index and tokens[index-1].text in ("~","class","struct"): return None
    if next_index>=len(tokens): return None
    left=index-1
    while left>=0 and tokens[left].text=="::":
        left-=1
        if left>=0 and tokens[left].text!="new": left-=1
    if left>=0 and tokens[left].text=="new" and tokens[next_index].text not in ("(","{"):
        return (None,index) if tokens[next_index].text=="[" else ([],index)
    if tokens[next_index].text in (">","["):
        left=index-1
        while left>=0 and tokens[left].text not in ("<",";","{","}"): left-=1
        if left<1 or tokens[left-1].text not in ("make_unique","make_shared"): return None
        if tokens[next_index].text=="[": return None,index
        if next_index+1<len(tokens) and tokens[next_index+1].text=="(":
            return arguments(tokens,next_index+1)
    named=bool(re.fullmatch(r"[A-Za-z_]\w*",tokens[next_index].text))
    if named: next_index+=1
    if named and next_index<len(tokens) and tokens[next_index].text=="[": return None,index
    if next_index<len(tokens) and tokens[next_index].text=="=": next_index+=1
    if next_index<len(tokens) and tokens[next_index].text in ("(","{"):
        parsed=arguments(tokens,next_index)
        if named and tokens[next_index].text=="(" and not parsed[0]:
            return None  # C++ parses Type name() as a function declaration.
        return parsed
    if named and next_index<len(tokens) and tokens[next_index].text in (";",","): return [],next_index
    return None

def workspace_fixture_boundary(path: str, caller: str, name: str,
        complete: str, statement: str) -> bool:
    """Permit only the existing exact low-level fixture selectors, not callers."""
    if (path,caller,name)==("src/v2/execution/local_execution/device/WorkspaceAllocator.cpp",
            "WorkspaceAllocator::createDeviceWorkspaceManager","DeviceWorkspaceManager"):
        return complete==("if(physical_memory_authority_){returnstd::make_unique<DeviceWorkspaceManager>"
            "(device,budget_bytes,physical_memory_authority_);}"
            "returnstd::make_unique<DeviceWorkspaceManager>(device,budget_bytes);")
    selectors={
        ("src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp",
         "DeviceGraphOrchestrator::ensureDeviceWorkspaceAllocated"):
            "workspace_allocator_=physical_memory_authority_?std::make_shared<WorkspaceAllocator>"
            "(physical_memory_authority_):std::make_shared<WorkspaceAllocator>();",
        ("src/v2/execution/local_execution/device/ReusableExecutionWorkspace.h","acquireImpl"):
            "slot.allocator=physical_memory_authority?std::make_shared<WorkspaceAllocator>"
            "(std::move(physical_memory_authority)):std::make_shared<WorkspaceAllocator>();",
    }
    return name=="WorkspaceAllocator" and statement==selectors.get((path,caller))

def scan_source(path: str, raw: str) -> list[Violation]:
    """Check executable tokens in one production file, with bounded provenance."""
    source = executable_source(raw)
    tokens = [Token(match.group(), match.start()) for match in TOKEN.finditer(source)]
    intervals = function_intervals(source)
    offsets = [token.offset for token in tokens]
    violations: list[Violation] = []
    forwarded = 0
    low_level_declarations = 0
    facade_calls = 0
    workspace_fixtures: dict[tuple[str,str],int] = {}
    for index, token in enumerate(tokens):
        name = token.text
        if name not in {*PREPARATION, "allocateUnadmitted", "allocate", "local", "onNode", "createGemm", "DeviceWorkspaceManager", "WorkspaceAllocator", "initial_cpu_storage_placement"}: continue
        if name in ("local","onNode") and (index < 2 or text(tokens[index-2:index]) != "CPUWeightStoragePlacement::"):
            continue
        if name in ("allocate","createGemm") and not path.startswith(ROLES): continue
        created=None
        if name in ("DeviceWorkspaceManager","WorkspaceAllocator"):
            if not path.startswith(ROLES): continue
            created=construction(tokens,index)
            if created is None: continue
        caller = enclosing_caller(intervals, token.offset)
        interval = min((entry for entry in intervals if entry[0] < token.offset < entry[1]),
                       key=lambda entry: entry[1]-entry[0], default=None)
        body = tokens[bisect.bisect_right(offsets,interval[0]):index] if interval else []
        def reject(message: str) -> None:
            violations.append(Violation(path, raw.count("\n", 0, token.offset)+1, caller, message))
        if name in ("DeviceWorkspaceManager","WorkspaceAllocator"):
            if not path.startswith(ROLES): continue
            args,end=created
            if args is None:
                reject(f"{name} arrays are unsupported; declare individual explicitly accounted owners")
                continue
            # Exact public constructor declarations/definitions are not calls.
            if (interval is None and path=="src/v2/execution/local_execution/device/DeviceWorkspaceManager.h" and
                len(args)==2 and text(args[0])=="DeviceIddevice" and text(args[1])=="size_tbudget_bytes"): continue
            if (interval is None and path=="src/v2/execution/local_execution/device/DeviceWorkspaceManager.cpp" and
                len(args)>=2 and text(args[0])=="DeviceIddevice" and text(args[1])=="size_tbudget_bytes"): continue
            if (interval is None and path=="src/v2/execution/local_execution/device/WorkspaceAllocator.h" and
                not args and end+2<len(tokens) and text(tokens[end+1:end+3])=="=default"): continue
            unbound=(name=="DeviceWorkspaceManager" and len(args)==2) or (name=="WorkspaceAllocator" and not args)
            # Template commas are outside this deliberately small constructor
            # grammar. Reject an ambiguous split rather than mistake it for PMA.
            ambiguous=any(text(arg).count("<")!=text(arg).count(">") for arg in args)
            multiple=(index+1<len(tokens) and re.fullmatch(r"[A-Za-z_]\w*",tokens[index+1].text) and
                end+1<len(tokens) and tokens[end+1].text==",")
            if not unbound and not ambiguous and not multiple: continue
            left=index-1
            while left>=0 and tokens[left].text not in (";","{","}"): left-=1
            complete=text(tokens[bisect.bisect_right(offsets,interval[0]):bisect.bisect_left(offsets,interval[1])]) if interval else ""
            statement=text(tokens[left+1:end+2])
            key=(caller,name)
            workspace_fixtures[key]=workspace_fixtures.get(key,0)+1
            if (not ambiguous and not multiple and workspace_fixtures[key]==1 and
                workspace_fixture_boundary(path,caller,name,complete,statement)): continue
            reject(f"{name} construction omits an explicit allocation authority or is ambiguous")
            continue
        if name == "allocate":
            if (path.startswith(ROLES) and index >= 2 and tokens[index-1].text in (".","->")):
                receiver = [tokens[index-2]]
                if weight_policy(receiver,body) and not placement_owned(receiver,body,path,caller):
                    reject("direct weight allocation uses an unaccounted CPU placement")
            continue
        if name in ("local","onNode"):
            if (path.startswith(ROLES) and index >= 2 and
                text(tokens[index-2:index]) == "CPUWeightStoragePlacement::" and
                index+1 < len(tokens) and tokens[index+1].text == "("):
                end = closing(tokens,index+1)
                if text(tokens[end+1:end+3]) in (".allocate","->allocate"):
                    reject("direct weight allocation uses an unaccounted CPU placement")
            continue
        if name == "allocateUnadmitted":
            # The exact low-level method declaration is infrastructure, not an
            # executable caller. Any reference elsewhere can expose the bypass.
            if (path == "src/v2/memory/CPUStorageRoot.h" and interval is None and
                text(tokens[max(0,index-4):index]) == "AlignedVector<T>"):
                low_level_declarations += 1
                if low_level_declarations == 1: continue
            if (path == "src/v2/memory/CPUWeightStoragePlacement.h" and caller == "allocate" and interval):
                complete = text(tokens[bisect.bisect_right(offsets,interval[0]):bisect.bisect_left(offsets,interval[1])])
                expected = ("if(authority_)returnCPUStorageRoot<T>(count,owner_,allocator_,placement_,"
                    "CPUStorageInitialization::Uninitialized).allocate(authority_);"
                    "returnplacement_.allocateUnadmitted<T>(count,CPUStorageInitialization::Uninitialized);")
                facade_calls += 1
                if complete == expected and facade_calls == 1: continue
            reject("allocateUnadmitted is forbidden outside its exact low-level declaration/facade")
            continue
        if name == "initial_cpu_storage_placement":
            declaration = "std::optional<CPUWeightStoragePlacement>initial_cpu_storage_placement;"
            if (path == "src/v2/execution/moe/MoEExpertWeightService.h" and interval is None and
                text(tokens[max(0,index-6):index+2]) == declaration):
                continue
            if index+1 < len(tokens) and tokens[index+1].text == "=":
                end = index+2
                while end < len(tokens) and tokens[end].text != ";":
                    if tokens[end].text in ("(", "[", "{"): end = closing(tokens,end)
                    end += 1
                if not placement_owned(tokens[index+2:end],body,path,caller):
                    reject("initial retained CPU placement must be explicitly accounted")
                continue
            # Only the two exact reads in the normalization above are supported.
            # emplace/reset, member pointers, references and other aliases fail
            # closed instead of bypassing the explicit assignment rule.
            complete = tokens[bisect.bisect_right(offsets,interval[0]):bisect.bisect_left(offsets,interval[1])] if interval else []
            normalization_read = (
                text(tokens[max(0,index-2):index+4]) == "ctx.initial_cpu_storage_placement.value_or(" or
                (text(tokens[max(0,index-5):index]) == "if(!ctx." and
                 text(tokens[index+1:index+6]) == "&&ctx.prepared_store)"))
            if not ((path,caller) == ("src/v2/execution/moe/MoEExpertWeightService.cpp",
                    "MoEExpertWeightService::prepareGemmEngines") and normalization_read and
                sum(entry.text == name for entry in complete) == 2 and
                placement_owned([Token("placement",token.offset)],complete,path,caller)):
                reject("unsupported setter or alias of initial retained CPU placement")
            continue
        if index+1 >= len(tokens) or tokens[index+1].text != "(":
            if name in PREPARATION:
                reject(f"indirect {name} reference has no checkable placement")
            elif path.startswith(ROLES) and name == "createGemm":
                reject("legacy createGemm reference bypasses prepared storage")
            continue
        args, end = arguments(tokens,index+1)
        # Function definitions and the exact public API declarations are not calls.
        if end+1 < len(tokens) and tokens[end+1].text == "{": continue
        if (path == "src/v2/kernels/KernelFactory.h" and interval is None and
            any("TensorBase" in text(arg) for arg in args)): continue
        if name == "createGemm":
            if path.startswith(ROLES): reject("legacy createGemm bypasses prepared storage")
            continue
        if len(args) <= PREPARATION[name]:
            reject(f"{name} omits explicit accounted CPU placement")
            continue
        placement = args[PREPARATION[name]]
        if placement_owned(placement,body,path,caller): continue
        if (path,caller,name,text(placement)) == ("src/v2/kernels/KernelFactory.cpp",
                "KernelFactory::prepareExpertGemmLocal","prepareExpertGemmLocal","placement"):
            forwarded += 1
            if forwarded == 1 and not assignments(body,"placement"): continue
        reject(f"{name} has unaccounted or unproven CPU placement")
    return violations

def scan(root: Path) -> list[Violation]:
    """Scan production C++ only, rejecting a missing or vacuous source inventory."""
    source_root=root/"src/v2"
    if not source_root.is_dir():
        return [Violation("src/v2",0,"<scanner>","missing production source tree")]
    result=[]
    executable=False
    for path in sorted(source_root.rglob("*")):
        if path.suffix not in SUFFIXES or not path.is_file(): continue
        raw=path.read_text(encoding="utf-8")
        if not executable: executable=bool(TOKEN.search(executable_source(raw)))
        preparation=any(name in raw for name in (*PREPARATION,"allocateUnadmitted","CPUWeightStoragePlacement",
            "createGemm","initial_cpu_storage_placement"))
        workspace=(not preparation and any(name in raw for name in ("DeviceWorkspaceManager","WorkspaceAllocator")) and
            WORKSPACE_CONSTRUCTION.search(executable_source(raw)))
        if not preparation and not workspace: continue
        result.extend(scan_source(path.relative_to(root).as_posix(),raw))
    if not executable:
        result.append(Violation("src/v2",0,"<scanner>","empty production source tree"))
    return result

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root",type=Path,required=True)
    args = parser.parse_args()
    violations = scan(args.repo_root)
    for violation in violations: print(violation,file=sys.stderr)
    if not violations: print("CPU retained preparation source policy: clean")
    return bool(violations)

if __name__ == "__main__": raise SystemExit(main())
