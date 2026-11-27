#!/usr/bin/env python3
"""Exercise the inert TLS proposal in disposable files; never qualify production.

Copy the complete fixture/support closure before applying the patch. The source
checkout and historical tools remain unchanged. This is a developer regression
check, not a production traffic, arithmetic, build or economy acceptance gate.
"""
import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]


def load(name, path):
    """Load separate original/proposed modules without changing their globals."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    """Run focused controls against only a temporary proposed implementation."""
    with tempfile.TemporaryDirectory(prefix="tlsle-review-") as directory:
        root = Path(directory)
        for relative in ("tests/v2/cmake", "tests/v2/unit/cmake"):
            shutil.copytree(ROOT / relative, root / relative,
                            ignore=shutil.ignore_patterns("__pycache__"))
        # The pre-existing bounded-memcpy regression reads kernel source only.
        (root / "src").symlink_to(ROOT / "src", target_is_directory=True)
        subprocess.run(["git", "apply", "--check", str(HERE / "tlsle-review-only.patch")],
                       cwd=root, check=True)
        subprocess.run(["git", "apply", str(HERE / "tlsle-review-only.patch")], cwd=root, check=True)
        test = root / "tests/v2/unit/cmake/test_cpu_native_tile_register_evidence.py"
        subprocess.run([sys.executable, "-B", str(test), "TLSLocalExecPairTest", "TLSDescriptorTest",
                        "CallModelTest", "CallProvenanceTest", "RelocationTest", "-v"], check=True)
        fixture = load("proposal_fixtures", test)
        original = load("original_checker", ROOT / "tests/v2/cmake/CPUNativeTileRegisterEvidence.py")
        data = fixture.witness_object([("f", fixture.tlsle_sequence() + "\tret\n")])
        try:
            original.read_object(data)
        except ValueError as error:
            assert str(error) == "relocation type 549 is not admitted in code (.text.f)", str(error)
            print("Original TLS negative control:", error)
        else:
            raise AssertionError("original unexpectedly admitted TLSLE")
        unit = fixture.other_unit("selected", {"helper": "\tret\n"})
        reason = original.CallModel.select(unit, unit, "helper", {
            "binding": "direct", "nonpreemptible": "static-executable"})
        assert reason == "direct binding without a non-preemptibility proof", reason
        print("Original static vocabulary control (NOT static proof):", reason)


if __name__ == "__main__":
    main()
