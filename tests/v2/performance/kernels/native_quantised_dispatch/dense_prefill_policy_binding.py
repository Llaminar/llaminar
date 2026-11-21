"""Freeze installed dense policy against the linked production selector before timing.

The native cold query reads the core's actual immutable table, not the trainer's
generated include. Independent canonical serialization joins that compiled table
to the exact retained include bytes. The executable/core closure binds the real
architecture-aware selector and generic dispatch; source filenames are not code
identity. This witness is admission metadata, never new performance evidence.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
import subprocess
from typing import Mapping

from .corpus_provenance import mapping_digest, require_release_trainer, trainer_provenance

SCHEMA = "rocm-dense-prefill-installed-policy-v1"
QUERY_TEST = "ROCmDensePrefillPolicy.CompiledInventory"
ROW_SCHEMA = "rocm-dense-prefill-installed-rows-v1"
_PRODUCERS = {"NativeCooperative": 0, "NativeStreaming": 1,
              "Int8BlockwiseV3": 2, "Int8BlockwiseV7": 3, "NativeWmma": 4}


def _digest(text: str) -> str:
    """Hash exact UTF-8 bytes, preserving the generated policy's literal envelope."""
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def installed_inventory_identity(base: str) -> tuple[int, str]:
    """Independently serialize every field of the admitted generated ROCm table."""
    from .dense_production_overlay import _split_generated_overlay

    _, entries, _ = _split_generated_overlay(base, "rocm")
    encoded = ROW_SCHEMA + "\n"
    pattern = re.compile(r", \{(\d+), (\d+), (\d+), (\d+), (true|false), VNNIPrefillProducer::(\w+)\}")
    for key, line in entries.items():
        match = pattern.search(line)
        if match is None:
            raise ValueError("installed dense policy lacks a complete physical launch")
        architecture = ((1, key.architecture.major, key.architecture.minor, key.architecture.stepping)
                        if key.architecture is not None else (2, 0, 0, 0))
        fields = (key.execution_codebook, key.m, key.n, key.k, int(key.prepared_family),
                  *(int(value) for value in match.groups()[:4]),
                  int(match.group(5) == "true"), _PRODUCERS[match.group(6)], *architecture)
        encoded += ",".join(map(str, fields)) + "\n"
    return len(entries), _digest(encoded)


def validate_installed_policy(binding: Mapping[str, object], trainer: Mapping[str, object]) -> None:
    """Reject changed base bytes, forged compiled rows or a different selector closure."""
    required = {"schema", "base_text", "base_sha256", "compiled_entry_count",
                "compiled_inventory_sha256", "trainer_provenance_digest", "query_test",
                "query_log", "query_log_sha256", "binding_digest"}
    if not isinstance(binding, dict) or set(binding) != required or binding.get("schema") != SCHEMA:
        raise ValueError("installed dense policy requires its complete original binding")
    require_release_trainer(trainer)
    if trainer.get("backend") != "rocm":
        raise ValueError("installed ROCm policy requires its own backend closure")
    if binding["trainer_provenance_digest"] != mapping_digest(trainer):
        raise ValueError("installed dense policy belongs to a different executable/core selector")
    base, log = binding["base_text"], binding["query_log"]
    if not isinstance(base, str) or not isinstance(log, str):
        raise ValueError("installed dense policy requires original include and native query bytes")
    if binding["base_sha256"] != _digest(base) or binding["query_log_sha256"] != _digest(log):
        raise ValueError("installed dense policy artifact digest mismatch")
    unsigned = dict(binding); unsigned.pop("binding_digest")
    if binding["binding_digest"] != mapping_digest(unsigned):
        raise ValueError("installed dense policy binding digest mismatch")
    reports = re.findall(r"^ROCM_DENSE_PREFILL_POLICY_V1,([1-9][0-9]*),(sha256:[0-9a-f]{64})$", log, re.MULTILINE)
    if (binding["query_test"] != QUERY_TEST or len(reports) != 1 or
            "[  PASSED  ] 1 test." not in log or "[  FAILED  ]" in log or "[  SKIPPED ]" in log):
        raise ValueError("installed dense policy requires exactly one successful native query")
    count, digest = installed_inventory_identity(base)
    if (type(binding["compiled_entry_count"]) is not int or
            (binding["compiled_entry_count"], binding["compiled_inventory_sha256"]) != (count, digest) or
            reports[0] != (str(count), digest)):
        raise ValueError("frozen dense include does not match the actual compiled selector table")


def freeze_installed_policy(base_include: Path, binary: Path) -> dict[str, object]:
    """Read the immutable base and authenticate its actual compiled owner before collection."""
    before = trainer_provenance(binary, "rocm")
    require_release_trainer(before)
    base = Path(base_include).read_bytes().decode("utf-8")
    count, digest = installed_inventory_identity(base)
    env = os.environ.copy()
    env.update(GTEST_COLOR="no", LLAMINAR_FORCE_CPU_ONLY_STARTUP="1",
               LLAMINAR_SKIP_CUDA_STARTUP="1", LLAMINAR_SKIP_ROCM_STARTUP="1")
    result = subprocess.run((str(Path(binary).resolve()), "--gtest_filter=" + QUERY_TEST),
                            env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    if result.returncode != 0:
        raise ValueError("compiled dense policy query failed: " + result.stdout[-4000:])
    if before != trainer_provenance(binary, "rocm"):
        raise ValueError("dense policy executable/core changed during admission")
    if base != Path(base_include).read_bytes().decode("utf-8"):
        raise ValueError("installed dense include changed during admission")
    binding = dict(schema=SCHEMA, base_text=base, base_sha256=_digest(base),
                   compiled_entry_count=count, compiled_inventory_sha256=digest,
                   trainer_provenance_digest=mapping_digest(before), query_test=QUERY_TEST,
                   query_log=result.stdout, query_log_sha256=_digest(result.stdout))
    binding["binding_digest"] = mapping_digest(binding)
    validate_installed_policy(binding, before)
    return binding
