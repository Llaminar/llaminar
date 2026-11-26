#!/usr/bin/env python3
"""Tool-free controls for the native partition plan link-authority receipt verifier.

A receipt shaped like a correct build-time observation must authenticate; every
stale, foreign, incomplete or contradicting observation must be rejected. These
controls execute no compiler, CMake, CTest or Ninja.
"""
import copy
import json
import pathlib
import sys
import tempfile
import types
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import native_partition_plan_link_authority as authority  # noqa: E402

IDENTITY = dict(path='/usr/bin/g++-14', realpath='/usr/bin/aarch64-linux-gnu-g++-14',
                sha256='0' * 64, version='g++-14 (fixture) 14.2.0\n')
UNDEFINED = ('consumer.cpp:(.text.startup+0x24): undefined reference to '
             '`llaminar2::cpu::native_quantised::failSuperblockWalk(int, int, unsigned long, unsigned long)\'\n'
             'consumer.cpp:(.text.startup+0x48): undefined reference to '
             '`llaminar2::cpu::native_quantised::failOrderedPartitionPlan(int, int, int, long)\'\n')


def outcome(returncode, output=''):
    """One recorded tool outcome."""
    return dict(returncode=returncode, output=output)


def registered(name, labels):
    """One MPI-free CTest registration as the canonical helper writes it."""
    return dict(name=name, command=['/build/llaminar_mpi_session', '--session-base', '/build/s', '--',
                                    '/usr/bin/cmake'], labels=labels)


def valid_report():
    """A receipt equal to what a correct build observes."""
    return dict(
        schema=authority.SCHEMA, sources=authority.sources(), compiler=IDENTITY,
        tools=dict(cmake='cmake', ctest='ctest', ninja='ninja'),
        observations={
            'link': dict(configure=outcome(0), linked_consumer=outcome(0),
                         unlinked_consumer=outcome(1, 'FAILED: unlinked_consumer\n' + UNDEFINED)),
            'registration': dict(
                configure=outcome(0),
                tests=[registered(authority.UNIT_TEST, ['V2', 'Unit']),
                       registered(authority.PREFLIGHT_TEST, ['V2', 'Integration', 'ProductionTestPreflight'])],
                preflight=[authority.PREFLIGHT_TEST]),
            'registration-without-mpi-procs': dict(
                configure=outcome(1, 'CMake Error at CMakeLists.txt:265 (message):\n  '
                                  + authority.MULTIPLE_RANKS + '\n'),
                tests=None, preflight=None),
        })


class NativePartitionPlanLinkReceipt(unittest.TestCase):
    """Authenticate the correct receipt; reject each contradicting one."""

    def rejects(self, mutate, message):
        """Apply one mutation to a valid receipt and require the named rejection."""
        report = valid_report()
        mutate(report)
        with self.assertRaisesRegex(RuntimeError, message):
            authority.verify_report(report, IDENTITY)

    def test_correct_receipt_authenticates(self):
        authority.verify_report(valid_report(), IDENTITY)

    def test_missing_receipt_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            (root / 'identity.json').write_text(json.dumps(IDENTITY))
            args = types.SimpleNamespace(report=root / 'receipt.json', identity=root / 'identity.json')
            with self.assertRaisesRegex(RuntimeError, 'Missing native partition plan link receipt'):
                authority.verify(args)

    def test_stale_source_is_rejected(self):
        def stale(report):
            report['sources']['cmake/NativePartitionPlan.cmake'] = 'f' * 64
        self.rejects(stale, 'stale')

    def test_changed_registration_block_is_rejected(self):
        def stale(report):
            report['sources']['walk_contract_registration'] = 'f' * 64
        self.rejects(stale, 'stale')

    def test_foreign_compiler_is_rejected(self):
        report = valid_report()
        other = dict(IDENTITY, sha256='1' * 64)
        with self.assertRaisesRegex(RuntimeError, 'compiler differs'):
            authority.verify_report(report, other)

    def test_incomplete_cases_are_rejected(self):
        self.rejects(lambda report: report['observations'].pop('registration-without-mpi-procs'), 'incomplete')

    def test_missing_reporter_in_authority_is_rejected(self):
        def broken(report):
            report['observations']['link']['linked_consumer'] = outcome(1, UNDEFINED)
        self.rejects(broken, 'consumer linking the authority failed')

    def test_consumer_without_authority_must_not_link(self):
        def linked(report):
            report['observations']['link']['unlinked_consumer'] = outcome(0)
        self.rejects(linked, 'linked its reporters')

    def test_consumer_without_authority_must_fail_on_each_reporter(self):
        for reporter in authority.REPORTERS:
            with self.subTest(reporter=reporter):
                def partial(report, reporter=reporter):
                    report['observations']['link']['unlinked_consumer'] = outcome(
                        1, ''.join(line + '\n' for line in UNDEFINED.splitlines() if reporter not in line))
                self.rejects(partial, 'did not fail on ' + reporter)

    def test_consumer_failure_for_another_reason_is_rejected(self):
        def compile_error(report):
            report['observations']['link']['unlinked_consumer'] = outcome(
                1, 'error: failSuperblockWalk failOrderedPartitionPlan: no such file\n')
        self.rejects(compile_error, 'another reason')

    def test_mpi_launching_registration_is_rejected(self):
        def mpi(report):
            tests = report['observations']['registration']['tests']
            tests[1]['command'][4:4] = ['mpirun', '-np', '2', '--']
        self.rejects(mpi, 'launches MPI')

    def test_unregistered_walk_contract_is_rejected(self):
        def missing(report):
            report['observations']['registration']['tests'].pop()
        self.rejects(missing, 'is not registered')

    def test_walk_contract_outside_preflight_is_rejected(self):
        def outside(report):
            report['observations']['registration']['preflight'] = []
        self.rejects(outside, 'preflight inventory')

    def test_registration_without_mpi_procs_must_fail_configuration(self):
        def accepted(report):
            report['observations']['registration-without-mpi-procs'] = copy.deepcopy(
                report['observations']['registration'])
        self.rejects(accepted, 'without MPI_PROCS configured')

    def test_registration_failure_must_name_multiple_ranks(self):
        def other(report):
            report['observations']['registration-without-mpi-procs']['configure'] = outcome(1, 'CMake Error')
        self.rejects(other, 'multiple-ranks diagnostic')

    def test_negative_removes_only_the_integration_mpi_procs(self):
        _, registration = authority.registration_blocks()
        negative = authority.without_preflight_mpi_procs(registration)
        unit_end = registration.index(f'add_v2_test({authority.PREFLIGHT_TEST}')
        self.assertEqual(negative[:unit_end], registration[:unit_end])
        self.assertNotIn(' MPI_PROCS 1', negative[unit_end:])
        self.assertEqual(len(registration) - len(negative), len(' MPI_PROCS 1'))
        with self.assertRaisesRegex(RuntimeError, 'exactly once'):
            authority.without_preflight_mpi_procs(negative)


if __name__ == '__main__':
    unittest.main()
