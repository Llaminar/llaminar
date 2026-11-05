#!/usr/bin/env python3
## @file protocol_replay.py
# @brief Replay the actual host-only completed semantic marker without native readiness.
"""Replay the actual host-only completed semantic marker without loading a runtime."""
from pathlib import Path
import hashlib
import json
import os
import subprocess

root = Path(os.environ["LLAMINAR_NATIVE_REPLAY_ROOT"])
backend = os.environ["LLAMINAR_NATIVE_REPLAY_BACKEND"]
compiler = os.environ["LLAMINAR_NATIVE_REPLAY_COMPILER"]
source = root / "tree" / backend / "src/collectives.cc"
text = source.read_text()
start = text.index("llaminarNcclCompletedProtocolVersion(")
opening = text.index("{", start)
depth, end = 1, opening + 1
while depth:
    depth += (text[end] == "{") - (text[end] == "}")
    end += 1
output = root / "replay"
output.mkdir(exist_ok=True)
cpp = output / "completed-protocol.cpp"
cpp.write_text('#include <cstdint>\n#include <iostream>\nextern "C" std::uint32_t ' + text[start:end] +
    '\nint main(){const auto v=llaminarNcclCompletedProtocolVersion();std::cout<<v<<"\\n";return v==3?0:1;}\n')
command = [compiler, "-std=c++17", "-O2", str(cpp), "-o", str(cpp.with_suffix(""))]
build = subprocess.run(command, capture_output=True, text=True)
if build.returncode:
    raise RuntimeError(build.stdout + build.stderr)
for _ in range(20):
    run = subprocess.run([str(cpp.with_suffix(""))], capture_output=True, text=True, timeout=1)
    if run.returncode or run.stdout != "3\n" or run.stderr:
        raise RuntimeError("Actual completed semantic marker rejected protocol3: " + run.stdout + run.stderr)
result = dict(backend=backend, compile_exit=0, returncode=0, protocol_version=3, runs=20,
    source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
(output / "protocol-result.json").write_text(json.dumps(result, indent=2) + "\n")
print("PASS actual host completed protocol3:20 observations; no native runtime")
