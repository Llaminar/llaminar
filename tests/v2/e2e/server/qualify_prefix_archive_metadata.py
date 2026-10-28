#!/usr/bin/env python3
"""Qualify the metadata-only auditor against one genuine current native writer.

This model-free test runs one existing DiskPrefixStorageBackend fixture, waits
for its normal process exit, then reads only journal/fixture metadata and stats
payload inodes. It never opens or hashes payload bytes. Explicit output retains
new qualification evidence; it cannot restore or certify a reclaimed app archive.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile
from unittest.mock import patch
import xml.etree.ElementTree as ET

from prefix_archive_metadata_audit import inventory


def identity(path):
    """Observe native producer file custody without reading its executable payload."""
    state = path.stat()
    return dict(device=state.st_dev, inode=state.st_ino, bytes=state.st_size,
                mtime_ns=state.st_mtime_ns, ctime_ns=state.st_ctime_ns)


def qualify(binary, directory):
    """Require one real successful native fixture before independent metadata audit."""
    binary = binary.resolve(strict=True)
    before = identity(binary)
    archive = directory/'archive'
    archive.mkdir()
    xml = directory/'native-writer.xml'
    command = [str(binary), '--gtest_filter=Test__DiskPrefixStorageBackend.MetadataAuditCurrentWriterProducesRetiredArchive',
               '--gtest_output=xml:'+str(xml)]
    environment = os.environ.copy()
    environment.update(LLAMINAR_FORCE_CPU_ONLY_STARTUP='1', LLAMINAR_SKIP_CUDA_STARTUP='1',
        LLAMINAR_SKIP_ROCM_STARTUP='1', OMP_NUM_THREADS='1', OMP_DYNAMIC='false',
        LLAMINAR_TEST_PREFIX_METADATA_AUDIT_DIRECTORY=str(archive))
    with (directory/'native-writer.log').open('x') as log:
        completed = subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT)
    if completed.returncode != 0:
        raise ValueError('Current native prefix-writer fixture failed; preserve its log/XML')
    observed = ET.parse(xml).getroot()
    cases = list(observed.iter('testcase'))
    if (len(cases) != 1 or cases[0].get('name') != 'MetadataAuditCurrentWriterProducesRetiredArchive'
        or cases[0].get('status') != 'run' or cases[0].get('result') != 'completed'
        or any(list(observed.iter(kind)) for kind in ('failure', 'error', 'skipped'))):
        raise ValueError('Native writer fixture missing, skipped or failed')
    metadata = json.loads((archive/'native-writer-fixture.json').read_text())
    if (metadata.get('schema') != 'llaminar.test.prefix_archive_native_fixture.v1'
        or type(metadata.get('budget_bytes')) is not int or not 0 < metadata['budget_bytes'] <= 4096
        or metadata.get('expected_entries') != 1
        or metadata.get('expected_payload_bytes') != metadata['budget_bytes']):
        raise ValueError('Unexpected native writer qualification geometry')
    original_open = Path.open

    def no_payload_open(path, *args, **kwargs):
        if path.parent.name.endswith('.kvcache.blocks'):
            raise AssertionError('Metadata audit attempted to open a payload')
        return original_open(path, *args, **kwargs)

    with patch.object(Path, 'open', no_payload_open):
        report = inventory(archive, metadata['budget_bytes'])
    if (report['entries'] != metadata['expected_entries']
        or report['active_payload_bytes'] != metadata['expected_payload_bytes']
        or report['payload_read_bytes'] != 0
        or report['journal'] != metadata['model_artifact_identity']+'.kvcache'):
        raise ValueError('Native writer and independent physical archive audit differ')
    if identity(binary) != before:
        raise ValueError('Native writer executable changed during qualification')
    result = dict(schema='llaminar.test.prefix_archive_native_audit.v1', passed=True,
        scope='new_model_free_fixture_not_reclaimed_app_archive', native_command=command,
        native_writer=str(binary), native_writer_identity=before, native_cases=1,
        native_failures=0, native_skips=0, archive=report)
    with (directory/'result.json').open('x') as output:
        json.dump(result, output, indent=2)
        output.write('\n')
    return result


def main():
    """Use one new output root or a scoped preflight temporary; never discover models/devices."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native-writer', required=True, type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.output is not None:
        directory = args.output.resolve()
        directory.mkdir(exist_ok=False)
        report = qualify(args.native_writer, directory)
    else:
        with tempfile.TemporaryDirectory(prefix='llaminar-prefix-native-audit-') as temporary:
            report = qualify(args.native_writer, Path(temporary))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
