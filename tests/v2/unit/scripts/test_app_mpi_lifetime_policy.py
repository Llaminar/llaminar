"""Keep process MPI ownership out of frontend request and command handlers.

The runtime regression proves destruction ordering; this source-policy guard
prevents another frontend from reintroducing an independent MPI lifetime.
"""
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[4]
APP = ROOT / "src/v2/app"


class AppMPILifetimePolicy(unittest.TestCase):
    def test_process_operations_have_one_infrastructure_owner(self):
        allowed = {
            "MPI_Init": "MPIProcessSession.cpp",
            "MPI_Init_thread": "MPIProcessSession.cpp",
            "MPI_Finalize": "MPIShutdown.cpp",
            "mpiShutdown": ("MPIProcessSession.cpp", "MPIShutdown.cpp"),
        }
        for source in APP.rglob("*.cpp"):
            # Comments document the lifecycle but are not executable API calls.
            body = re.sub(r"/\*.*?\*/|//[^\n]*", "", source.read_text(), flags=re.S)
            for operation, owners in allowed.items():
                with self.subTest(source=source.relative_to(ROOT), operation=operation):
                    if source.name not in ((owners,) if isinstance(owners, str) else owners):
                        self.assertIsNone(re.search(rf"\b{operation}\s*\(", body))


if __name__ == "__main__":
    unittest.main()
