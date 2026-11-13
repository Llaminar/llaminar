#!/usr/bin/env python3
"""Freeze explicit local QFN paired metadata into one test-target build input.

This validates small document/source metadata only. It never loads weights or
logits, resolves placement, expands a topology, starts a runner or approves a
numerical result. Actual source cases reject unused IDs during registration.
The unset CMake option never calls this tool or creates an artifact dependency.
"""
from pathlib import Path
import argparse
import hashlib
import json

from model_kld_corpus import identity, metadata_digest, require, unchanged
import model_kld_corpus_v2 as corpus_v2
from model_kld_corpus_execution import original_source_identity, validate_absolute_pilot
from run_model_parity_benchmarks import validate_workload

MAXIMUM_BYTES = 16 * 1024 * 1024


def read_document(path, pins, digest=None):
    """Read one bounded metadata object and pin it before any generated output."""
    path = Path(path).resolve(strict=True)
    require(not any(c in str(path) for c in ';\n\r'), 'Local CMake metadata path is not representable')
    pin = identity(path)
    require(path.is_file() and 0 < pin[3] <= MAXIMUM_BYTES, 'Local paired input exceeds bounded metadata extent')
    raw = path.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    require(digest is None or sha == digest, 'Local paired document digest changed: '+str(path))
    require(identity(path) == pin, 'Local paired document changed while reading')
    value = json.loads(raw)
    corpus_v2._finite_json(value)
    require(type(value) is dict, 'Local paired metadata must be an object')
    pins[str(path)] = pin
    return value, raw, sha


def freeze(bindings_path, source_root):
    """Validate the explicit bindings without inventing canonical case eligibility.

    Catalog/manifest, all policy groups and six candidate selectors remain
    canonical validator inputs. Source-issued stats are checked without reading
    payloads. Unknown source case IDs are the later typed registration guard.
    """
    pins = {}
    path = Path(bindings_path).resolve(strict=True)
    document, raw, sha = read_document(path, pins)
    require(set(document) == {'schema','source','bindings'} and
            document['schema'] == 'llaminar.qfn-local-paired-profiles.v1', 'Wrong local QFN binding schema')
    bindings = document['bindings']
    require(type(bindings) is list and 0 < len(bindings) <= 128, 'Missing bounded local QFN bindings')
    seen = set()
    for binding in bindings:
        require(type(binding) is dict and set(binding) == {'exact_cell','role','catalog','workload','pilot','benchmark','arithmetic'},
                'Local QFN binding has missing or foreign fields')
        exact = binding['exact_cell']
        require(type(exact) is str and '.ProductionParity/' in exact and exact not in seen and
                '\0' not in exact and binding['role'] in ('required','expanded'), 'Invalid/duplicate local exact case')
        seen.add(exact)
        values = {}
        for name in ('catalog','workload','pilot','benchmark'):
            ref = binding[name]
            require(type(ref) is dict and set(ref) == {'path','sha256'} and
                    type(ref['path']) is str and ref['path'] and '\0' not in ref['path'] and
                    type(ref['sha256']) is str and len(ref['sha256']) == 64 and
                    all(c in '0123456789abcdef' for c in ref['sha256']), 'Invalid local document reference')
            values[name], _, _ = read_document(Path(source_root)/ref['path'], pins, ref['sha256'])
        manifest = corpus_v2.validate_manifest(values['workload'])
        require(values['catalog'] == {'schema':2,'manifest_digest':metadata_digest(manifest)} and
                manifest['source'] == document['source'], 'Local QFN workload/catalog/source identity disagrees')
        matches = [t for t in manifest['topologies'] if t['canonical_cell'] == exact]
        require(len(matches) == 1, 'Local QFN corpus must own exactly one original topology per binding')
        require(binding['arithmetic'] == matches[0]['arithmetic'] and
                set(binding['arithmetic']) == set(corpus_v2.ARITHMETIC_FIELDS),
                'Local QFN candidate arithmetic differs from its source-owned corpus')
        validate_absolute_pilot(values['pilot'], manifest, matches[0])
        validate_workload(values['benchmark'])  # Existing runnable production workload schema, not a perf result.
    source = document['source']
    require('shared_predictor' in source, 'Local QFN paired source must explicitly declare the inactive predictor')
    for descriptor in source['shards'] + [source['shared_predictor']]:
        native_path = Path(descriptor['path']).resolve(strict=True)
        pins[str(native_path)] = original_source_identity(native_path, descriptor)
    unchanged(pins)
    delimiter = 'QFN'+sha[:12]
    require((')'+delimiter+'"').encode() not in raw, 'Local binding contains reserved generated-header delimiter')
    # The complete original source and document bytes are retained in the test
    # binary. Never reopen an ambient document during GTest registration.
    frozen = json.dumps(dict(binding_path=str(path),binding_sha256=sha,
        source_sha256=metadata_digest(source),document=document),ensure_ascii=False,separators=(',',':'))
    require((')'+delimiter+'\"') not in frozen, 'Local binding contains reserved generated-header delimiter')
    header = ('/** @file QFNLocalPairedBindings.generated.h\n'
              ' * @brief Build-local immutable metadata only; no model/runtime qualification. */\n'
              '#pragma once\nnamespace llaminar2::test::parity::qwen4exp::local_paired_build {\n'
              'inline constexpr char input[]=R\"'+delimiter+'('+frozen+')'+delimiter+'\";\n}\n')
    return header, dict(schema='llaminar.qfn-local-paired-build-input.v1', binding_path=str(path),
        binding_sha256=sha,source_sha256=metadata_digest(source),source=source,
        exact_cells=sorted(seen),predictor=source['shared_predictor']['path'],
        dependencies=sorted(pins),scope='local_build_metadata_not_runtime_or_numeric_evidence')


def write_if_changed(path, text):
    """Keep unchanged configure inputs from needlessly relinking the matrix."""
    path = Path(path)
    if path.is_file() and path.read_text() == text:return
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(text)


def main(argv=None):
    """Freeze only after every metadata/source check passes; no partial input is published."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bindings',type=Path,required=True)
    parser.add_argument('--source-root',type=Path,required=True)
    parser.add_argument('--header',type=Path,required=True)
    parser.add_argument('--metadata',type=Path,required=True)
    args = parser.parse_args(argv)
    header, metadata = freeze(args.bindings,args.source_root)
    write_if_changed(args.header,header)
    write_if_changed(args.metadata,json.dumps(metadata,sort_keys=True,indent=2)+'\n')
    print(json.dumps(metadata))


if __name__ == '__main__':main()
