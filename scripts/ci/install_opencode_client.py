#!/usr/bin/env python3
"""Install the exact native OpenCode client used by the app stress workload.

The published-suite controller owns this dependency, independently of the tested
inference image. Authenticate the pinned npm code artifact before extracting
its single executable, then verify the actual CLI version before atomic
publication. This never downloads models or initializes prefix-cache storage.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
from urllib.request import urlopen

VERSION = '1.18.34'
PACKAGE_URL = 'https://registry.npmjs.org/opencode-linux-x64/-/opencode-linux-x64-1.18.34.tgz'
PACKAGE_INTEGRITY = 'sha512-RTAMjCve4euxP2QKLuvRmdoW5J5DQK1DiZqt+7slfixyjAEi79QC2Df2oYKogibaAI4IEU8uzenoJeEl3k+UEw=='
EXECUTABLE_MEMBER = 'package/bin/opencode'


def install_client(archive: Path, destination: Path) -> dict:
    """Validate the complete code artifact and publish only a working pinned CLI."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with archive.open('rb') as source:
            digest = hashlib.file_digest(source, 'sha512').digest()
            actual = 'sha512-' + base64.b64encode(digest).decode('ascii')
            if actual != PACKAGE_INTEGRITY:
                raise ValueError('OpenCode package integrity differs from the tested client pin')
            source.seek(0)
            with tarfile.open(fileobj=source, mode='r:gz') as package:
                members = [member for member in package.getmembers() if member.name == EXECUTABLE_MEMBER]
                if len(members) != 1 or not members[0].isfile() or members[0].size <= 0:
                    raise ValueError('OpenCode archive must contain one regular native executable')
                with package.extractfile(members[0]) as binary, tempfile.NamedTemporaryFile(
                        prefix='.opencode-', dir=destination.parent, delete=False) as target:
                    temporary = Path(target.name)
                    shutil.copyfileobj(binary, target)
        temporary.chmod(0o755)
        version = subprocess.check_output([str(temporary), '--version'], text=True, timeout=30,
            env={**os.environ, 'OPENCODE_DISABLE_AUTOUPDATE': 'true', 'OPENCODE_DISABLE_MODELS_FETCH': 'true'}).strip()
        if version != VERSION:
            raise ValueError(f'OpenCode CLI version {version!r} differs from tested version {VERSION}')
        temporary.replace(destination)
        temporary = None
        return {'version': version, 'package_url': PACKAGE_URL, 'package_integrity': actual,
                'executable': str(destination), 'version_verified': True}
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main() -> int:
    """Download once or consume a preseeded archive without changing the pin."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path, default=Path('/usr/local/bin/opencode'))
    parser.add_argument('--archive', type=Path, help='Preseeded copy of the exact pinned npm code artifact')
    args = parser.parse_args()
    if args.archive is not None:
        result = install_client(args.archive, args.destination)
    else:
        with tempfile.TemporaryDirectory(prefix='llaminar-opencode-install-') as directory:
            archive = Path(directory) / 'client.tgz'
            with urlopen(PACKAGE_URL, timeout=30) as response, archive.open('wb') as target:
                shutil.copyfileobj(response, target)
            result = install_client(archive, args.destination)
    print(json.dumps(result), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
