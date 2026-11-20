#!/usr/bin/env python3
"""@file test_ctest_timeout_metadata.py
@brief Prove actual CTest JSON retains the timeout-owner retirement contract.

CTest3.28 supports runtime signals but omits these fields from JSON, so version
or successful configuration alone cannot certify the MPI inventory. A real
LANGUAGES NONE fixture exercises the exact selected CMake/CTest/Ninja tools.
This checks tool evidence, not actual MPI timeout cleanup or model inference;
the existing process-lifecycle suites retain those separate responsibilities.
"""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

TOOLS: argparse.Namespace
ROOT = Path(__file__).resolve().parents[4]


def validate_development_producers(text):
    """Reject the retired independent CMake producer and fixed build-concurrency policy."""
    statement = ''
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        continued = line.endswith('\\')
        statement += ' ' + (line[:-1] if continued else line)
        if continued:
            continue
        command, _, arguments = statement.strip().partition(' ')
        if command.upper() == 'RUN' and 'apt.kitware.com' in arguments:
            raise ValueError('development image retains an independent unpinned CMake producer')
        if command.upper() == 'ENV' and 'CMAKE_BUILD_PARALLEL_LEVEL=' in arguments:
            raise ValueError('development image retains a fixed ordinary build worker limit')
        statement = ''


def validate(inventory):
    """Require the exact actual signal/grace fields, without a default or property-name mirror."""
    rows = [row for row in inventory.get('tests', []) if row['name'] == 'TimeoutOwnerMetadata']
    if len(rows) != 1:
        raise ValueError('missing or duplicate actual timeout-owner fixture')
    values = {row['name']: row['value'] for row in rows[0].get('properties', [])}
    if values.get('TIMEOUT_SIGNAL_NAME') != 'SIGTERM' or values.get('TIMEOUT_SIGNAL_GRACE_PERIOD') != 36:
        raise ValueError('actual CTest JSON does not expose SIGTERM/36 timeout-owner retirement metadata')


class CTestTimeoutMetadata(unittest.TestCase):
    """Exercise real tool output and independent omission/corruption controls."""

    @classmethod
    def setUpClass(cls):
        """Configure a real compiler-free project and retain its actual JSON inventory."""
        with tempfile.TemporaryDirectory(prefix='llaminar-ctest-owner-metadata-') as directory:
            root = Path(directory)
            source, build = root / 'source', root / 'build'
            source.mkdir()
            (source / 'CMakeLists.txt').write_text('''cmake_minimum_required(VERSION 3.25)
project(CTestTimeoutOwnerMetadata NONE)
enable_testing()
add_test(NAME TimeoutOwnerMetadata COMMAND ${CMAKE_COMMAND} -E true)
set_tests_properties(TimeoutOwnerMetadata PROPERTIES
    TIMEOUT_SIGNAL_NAME SIGTERM TIMEOUT_SIGNAL_GRACE_PERIOD 36)
''', encoding='utf-8')
            result = subprocess.run([str(TOOLS.cmake), '-S', str(source), '-B', str(build), '-G', 'Ninja',
                '-DCMAKE_MAKE_PROGRAM:FILEPATH=' + str(TOOLS.ninja)], capture_output=True, text=True, timeout=20)
            if result.returncode:
                raise AssertionError(result.stdout + result.stderr)
            cls.inventory = json.loads(subprocess.check_output(
                [str(TOOLS.ctest), '--test-dir', str(build), '--show-only=json-v1'], text=True, timeout=20))

    def test_actual_timeout_owner_fields_are_observable(self):
        """Only actual selected-tool output, not the CMake version claim, proves the interface."""
        validate(self.inventory)

    def test_development_image_uses_only_the_shared_tool_selection_policy(self):
        """Structural policy complements, never replaces, actual selected-tool JSON proof."""
        source = (ROOT / '.devcontainer/Dockerfile').read_text(encoding='utf-8')
        validate_development_producers(source)
        for retired in ['RUN wget https://apt.kitware.com/keys/kitware.asc\n',
                        'ENV CMAKE_BUILD_PARALLEL_LEVEL=56\n']:
            with self.subTest(retired=retired), self.assertRaises(ValueError):
                validate_development_producers(source + '\n' + retired)

    def test_missing_or_changed_retirement_fields_are_rejected(self):
        """An omitted field, altered signal or insufficient grace cannot be silently repaired."""
        baseline = {'tests': [{'name': 'TimeoutOwnerMetadata', 'properties': [
            {'name': 'TIMEOUT_SIGNAL_NAME', 'value': 'SIGTERM'},
            {'name': 'TIMEOUT_SIGNAL_GRACE_PERIOD', 'value': 36}]}]}
        for name, value in [('TIMEOUT_SIGNAL_NAME', None), ('TIMEOUT_SIGNAL_NAME', 'SIGKILL'),
                            ('TIMEOUT_SIGNAL_GRACE_PERIOD', None), ('TIMEOUT_SIGNAL_GRACE_PERIOD', 35)]:
            changed = copy.deepcopy(baseline)
            properties = changed['tests'][0]['properties']
            properties[:] = [row for row in properties if row['name'] != name]
            if value is not None:
                properties.append({'name': name, 'value': value})
            with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                validate(changed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cmake', type=Path, required=True)
    parser.add_argument('--ctest', type=Path, required=True)
    parser.add_argument('--ninja', type=Path, required=True)
    TOOLS, rest = parser.parse_known_args()
    for name in ('cmake', 'ctest', 'ninja'):
        setattr(TOOLS, name, getattr(TOOLS, name).resolve(strict=True))
    unittest.main(argv=[sys.argv[0], *rest])
