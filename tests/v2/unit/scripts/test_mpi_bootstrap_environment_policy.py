## @file test_mpi_bootstrap_environment_policy.py
#  @brief Adversarial policy checks for the single typed live MPI environment bridge.
"""Exercise the actual canonical source guard without SDKs or native runtimes."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[4]
GUARD = REPO / "scripts/check_no_direct_getenv_outside_debugenv.sh"
SCHEMA = "src/v2/utils/MPISessionEnvironment.h"
OWNER = "src/v2/utils/MPISessionOwner.cpp"
MODULE = "src/v2/utils/NativeMPISessionSchizo.cpp"


class MPIBootstrapEnvironmentPolicy(unittest.TestCase):
    """A function-scoped exception must not become another generic env authority."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="mpi-env-policy-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        for name in (SCHEMA, OWNER, MODULE):
            target = self.root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(REPO / name, target)
        # Preserve the canonical legacy exceptions so this fixture also checks
        # the guard's real stale-exception validation, instead of bypassing it.
        for name in ("src/v2/app/RuntimeInitPhase.cpp", "src/v2/utils/CPUFeatures.h",
                     "src/v2/utils/DebugEnv.h"):
            self.write(name, 'const char *legacy = std::getenv("LEGACY");\n')

    def write(self, name, body):
        target = self.root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body)

    def run_guard(self, success, diagnostic=None):
        result = subprocess.run(["bash", str(GUARD), str(self.root)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0 if success else 1, result.stdout + result.stderr)
        if diagnostic:
            self.assertIn(diagnostic, result.stderr)

    def append(self, name, body):
        with (self.root / name).open("a") as source:
            source.write(body)

    def test_actual_mpi_infrastructure_has_one_live_bridge(self):
        self.run_guard(True)

    def test_original_owner_raw_reads_reject(self):
        self.append(OWNER, 'auto regression = ::getenv("TMPDIR");\n')
        self.run_guard(False, OWNER)

    def test_original_module_raw_read_rejects(self):
        self.append(MODULE, 'auto regression = std::getenv("OMPI_MCA_llaminar_session_protocol");\n')
        self.run_guard(False, MODULE)

    def test_header_is_not_blanket_exempt(self):
        self.append(SCHEMA, 'auto regression = std::getenv("UNRELATED");\n')
        self.run_guard(False, SCHEMA)

    def test_arbitrary_string_overload_rejects(self):
        self.append(SCHEMA, 'const char *readAny(const char *name) { return std::getenv(name); }\n')
        self.run_guard(False, SCHEMA)

    def test_changed_bridge_body_rejects(self):
        path = self.root / SCHEMA
        path.write_text(path.read_text().replace(
            "return std::getenv(environmentName(key));", "return std::getenv(unchecked_name);"))
        self.run_guard(False, SCHEMA)

    def test_second_bridge_definition_rejects(self):
        source = (self.root / SCHEMA).read_text()
        start = source.index("    inline const char *processEnvironmentValue(EnvironmentKey key)")
        end = source.index("\n    }", start) + len("\n    }")
        self.append(SCHEMA, source[start:end])
        self.run_guard(False, SCHEMA)

    def test_unrelated_engine_raw_read_rejects(self):
        self.write("src/v2/execution/Engine.cpp", 'auto value = std::getenv("UNRELATED");\n')
        self.run_guard(False, "src/v2/execution/Engine.cpp")

    def test_unrelated_engine_cannot_use_bootstrap_accessor(self):
        self.write("src/v2/execution/Engine.cpp", 'auto value = mpi_session::processEnvironmentValue(key);\n')
        self.run_guard(False, "MPI bootstrap accessor is infrastructure-only")

    def test_legacy_stale_exception_still_rejects(self):
        self.write("src/v2/utils/CPUFeatures.h", "// The old exception was migrated.\n")
        self.run_guard(False, "exception allowlist contains files with no remaining getenv calls")


if __name__ == "__main__":
    unittest.main()
