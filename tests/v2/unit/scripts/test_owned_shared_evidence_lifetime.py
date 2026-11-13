## @file test_owned_shared_evidence_lifetime.py
#  @brief Device-free regression of harness-owned diagnostic tree cleanup.
"""Exercise argument dispatch and real temporary-directory ownership, without MPI."""
from __future__ import annotations
import contextlib
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

SCRIPTS = Path(__file__).resolve().parents[2] / 'integration/scripts'
sys.path.insert(0, str(SCRIPTS))


def load(name):
    """Load the real script's entrypoint; only native transaction work is substituted."""
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class OwnedSharedEvidenceLifetime(unittest.TestCase):
    """Diagnostic cleanup must never be confused with runtime backing retirement."""
    def test_default_cleanup_and_explicit_preservation_on_success_and_failure(self):
        factories = {
            'test_mpi_owned_shared_backing': ['--runner', '/runner', '--pause-module', '/pause'],
            'test_moe_owned_shared_startup': ['--probe', '/probe'],
            'test_mpi_owned_shared_borrow': ['--probe', '/probe', '--repeat-probe', '/repeat', '--race-module', '/race'],
        }
        original_temporary = tempfile.TemporaryDirectory
        for name, extra in factories.items():
            module = load(name)
            for explicit in (False, True):
                for failure in (False, True):
                    with self.subTest(script=name, explicit=explicit, failure=failure):
                        with original_temporary() as parent:
                            observed = []
                            def transaction(args):
                                args.evidence.mkdir(parents=True)
                                (args.evidence / 'output.log').write_text('original diagnostic failure receipt')
                                observed.append(args.evidence)
                                if failure:
                                    raise RuntimeError('expected test transaction failure')
                            argv = [name, '--helper', '/helper', '--mpirun', '/mpirun', *extra]
                            if explicit:
                                argv += ['--evidence', str(Path(parent) / 'retained')]
                            error = io.StringIO()
                            with mock.patch.object(module, 'transaction', side_effect=transaction), \
                                 mock.patch.object(sys, 'argv', argv), \
                                 mock.patch.object(module.tempfile, 'TemporaryDirectory', side_effect=lambda **kw: original_temporary(dir=parent, **kw)), \
                                 contextlib.redirect_stderr(error):
                                if failure:
                                    with self.assertRaisesRegex(RuntimeError, 'expected test transaction failure'):
                                        module.main()
                                else:
                                    module.main()
                            self.assertEqual(len(observed), 1)
                            self.assertEqual(observed[0].exists(), explicit)
                            if not explicit:
                                self.assertEqual(list(Path(parent).iterdir()), [])
                                if failure:
                                    self.assertIn('original diagnostic failure receipt', error.getvalue())
                            else:
                                self.assertEqual((observed[0] / 'output.log').read_text(), 'original diagnostic failure receipt')

    def test_failed_retirement_keeps_unknown_runtime_backing(self):
        """A real script entrypoint may not rmtree a scope with unknown retirement."""
        for name, extra in {
            'test_mpi_owned_shared_backing': ['--runner', '/runner', '--pause-module', '/pause'],
            'test_moe_owned_shared_startup': ['--probe', '/probe'],
            'test_mpi_owned_shared_borrow': ['--probe', '/probe', '--repeat-probe', '/repeat', '--race-module', '/race'],
        }.items():
            with self.subTest(script=name):
                module = load(name)
                original_temporary = tempfile.TemporaryDirectory
                with original_temporary() as parent:
                    scopes = []
                    def transaction(args):
                        backing = args.evidence / 'case/sessions/llaminar-mpi-original'
                        backing.mkdir(parents=True)
                        (backing / 'live-backing').write_text('original unknown-retirement inode')
                        args.session_bases = [backing.parent]
                        scopes.append(backing)
                        raise RuntimeError('injected retirement failure')
                    argv = [name, '--helper', '/helper', '--mpirun', '/mpirun', *extra]
                    with mock.patch.object(module, 'transaction', side_effect=transaction), \
                         mock.patch.object(sys, 'argv', argv), \
                         mock.patch.object(module.tempfile, 'TemporaryDirectory', side_effect=lambda **kw: original_temporary(dir=parent, **kw)), \
                         contextlib.redirect_stderr(io.StringIO()):
                        with self.assertRaisesRegex(RuntimeError, 'injected retirement failure'):
                            module.main()
                    self.assertEqual((scopes[0] / 'live-backing').read_text(), 'original unknown-retirement inode')
                    # This test owns the synthetic fixture; no live process or
                    # guardian was launched. Native suites never perform this cleanup.


if __name__ == '__main__':
    unittest.main()
