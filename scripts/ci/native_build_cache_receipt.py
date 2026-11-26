#!/usr/bin/env python3
# @file native_build_cache_receipt.py
# @brief Interpret BuildKit operation provenance without claiming cached work is fresh.
import json
from pathlib import Path
import sys


def receipt(progress, status):
    """Join streamed vertex updates by identity and report the original operation."""
    vertices = {}
    for line in Path(progress).read_text().splitlines():
        event = json.loads(line)
        key = event.get('id')
        if key:
            vertices.setdefault(key, {}).update(event)
    matches = [v for v in vertices.values() if 'arm_native_build_evidence.py capture' in v.get('name', '')]
    if len(matches) != 1 or not matches[0].get('completed'):
        raise ValueError('missing or ambiguous completed operation vertex')
    origin = json.loads(Path(status).read_text())['nonce']
    return dict(cached=matches[0].get('cached', False), origin=origin)


if __name__ == '__main__':
    print(json.dumps(receipt(*sys.argv[1:3]), indent=2))
