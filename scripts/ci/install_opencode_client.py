#!/usr/bin/env python3
"""Build and qualify the pinned OpenCode dependency before atomic publication.

The suite-driver builder owns compiler/Bun dependencies. Its final image receives
only the qualified native executable and receipt. The local provider is explicit;
the source-owned models fixture supplies immutable build metadata, not a claim
that this dedicated test client qualifies the general OpenCode model catalog.
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
import sys
import tarfile
import tempfile
from urllib.request import urlopen

VERSION = '1.18.34-llaminar.1'
UPSTREAM_COMMIT = 'aec0b9a6d8898f68f923aaf08b7306d931fd9d76'
SOURCE_URL = f'https://codeload.github.com/anomalyco/opencode/tar.gz/{UPSTREAM_COMMIT}'
SOURCE_SHA256 = '277b0f716a7b346b271b84ed0d7809b89110333634de847e49adcab1a1137285'
BUN_VERSION = '1.3.14'
BUN_URL = f'https://registry.npmjs.org/@oven/bun-linux-x64/-/bun-linux-x64-{BUN_VERSION}.tgz'
BUN_INTEGRITY = 'sha512-7OVTAKvwfPmSbIV1HpdOoVVx5VRc427GuPPne93N6vk4eQBPId9nXmZDh9/zGaKPdbVjVtQSZafWQoUjx38Utw=='
MODELS_FIXTURE = 'packages/opencode/test/tool/fixtures/models-api.json'
MODELS_SHA256 = '876afae217cdff37c7267ba757d4d375a64eb52cdb2caa24c7f027b5142e3265'
SUPPORT = Path(__file__).with_name('opencode-client')


def download(url: str, target: Path) -> None:
    """Fetch a pinned input; callers authenticate it before extraction/execution."""
    with urlopen(url, timeout=30) as source, target.open('wb') as output:
        shutil.copyfileobj(source, output)


def authenticate(path: Path, expected: str, algorithm: str = 'sha256') -> None:
    """Reject changed input bytes before any archive member can execute."""
    with path.open('rb') as source:
        digest = hashlib.file_digest(source, algorithm).digest()
    actual = 'sha512-' + base64.b64encode(digest).decode() if algorithm == 'sha512' else digest.hex()
    if actual != expected:
        raise ValueError(f'OpenCode dependency integrity mismatch: {path.name}')


def extract_source(archive: Path, directory: Path) -> Path:
    """Authenticate the exact upstream source and reject unsafe archive paths."""
    authenticate(archive, SOURCE_SHA256)
    with tarfile.open(archive, 'r:gz') as source:
        source.extractall(directory, filter='data')
    root = directory / f'opencode-{UPSTREAM_COMMIT}'
    if not root.is_dir():
        raise ValueError('OpenCode source archive is missing its pinned root')
    authenticate(root / MODELS_FIXTURE, MODELS_SHA256)
    return root


def qualify_client(binary: Path) -> dict:
    """Run actual native CLI lifecycle controls against a local scripted HTTP peer."""
    output = subprocess.check_output([sys.executable, str(SUPPORT / 'qualify_client.py'), str(binary)], text=True)
    controls = json.loads(output)
    expected = {'stop', 'tool_calls', 'tool_stop', 'provider_overflow'}
    if (not isinstance(controls, list) or len(controls) != len(expected)
            or {row.get('mode') for row in controls} != expected
            or not all(row.get('passed') is True for row in controls)):
        raise ValueError('OpenCode native lifecycle qualification is incomplete')
    return {'passed': True, 'controls': controls}


def publish_client(binary: Path, destination: Path, provenance: dict) -> dict:
    """Keep the existing installation until version and real lifecycle both pass."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix='.opencode-', dir=destination.parent, delete=False) as target:
            temporary = Path(target.name)
            with binary.open('rb') as source:
                shutil.copyfileobj(source, target)
        temporary.chmod(0o755)
        version = subprocess.check_output([str(temporary), '--version'], text=True, timeout=30,
            env={**os.environ, 'OPENCODE_DISABLE_AUTOUPDATE': 'true', 'OPENCODE_DISABLE_MODELS_FETCH': 'true'}).strip()
        if version != VERSION:
            raise ValueError(f'OpenCode CLI version {version!r} differs from tested version {VERSION}')
        qualification = qualify_client(temporary)
        with temporary.open('rb') as executable:
            digest = hashlib.file_digest(executable, 'sha256').hexdigest()
        temporary.replace(destination)
        temporary = None
        return {**provenance, 'version': version, 'version_verified': True,
                'executable': str(destination), 'binary_sha256': digest, 'native_qualification': qualification}
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def install_client(archive: Path, destination: Path, *, bun_archive: Path | None = None) -> dict:
    """Build the authenticated source with its lockfile, then qualify and publish."""
    with tempfile.TemporaryDirectory(prefix='llaminar-opencode-build-') as temporary:
        directory = Path(temporary)
        source = extract_source(archive, directory)
        if bun_archive is None:
            bun_archive = directory / 'bun.tgz'
            download(BUN_URL, bun_archive)
        authenticate(bun_archive, BUN_INTEGRITY, 'sha512')
        tools = directory / 'tools'
        tools.mkdir()
        with tarfile.open(bun_archive, 'r:gz') as package:
            members = [member for member in package.getmembers() if member.name == 'package/bin/bun']
            if len(members) != 1 or not members[0].isfile():
                raise ValueError('Bun archive must contain one regular native executable')
            with package.extractfile(members[0]) as binary, (tools / 'bun').open('wb') as target:
                shutil.copyfileobj(binary, target)
        (tools / 'bun').chmod(0o755)
        patch = SUPPORT / 'terminal-stop.patch'
        subprocess.run(['patch', '--batch', '--fuzz=0', '-p1', '-i', str(patch.resolve())],
                       cwd=source, check=True, stdout=sys.stderr)
        env = {**os.environ, 'PATH': str(tools) + os.pathsep + os.environ.get('PATH', ''),
               'OPENCODE_VERSION': VERSION, 'OPENCODE_CHANNEL': 'latest',
               'MODELS_DEV_API_JSON': str(source / MODELS_FIXTURE), 'HUSKY': '0'}
        subprocess.run([str(tools / 'bun'), 'install', '--frozen-lockfile'], cwd=source,
                       env=env, check=True, stdout=sys.stderr)
        subprocess.run([str(tools / 'bun'), 'run', 'script/build.ts', '--single', '--skip-install',
                        '--skip-embed-web-ui'], cwd=source / 'packages/opencode', env=env,
                       check=True, stdout=sys.stderr)
        provenance = {'upstream_commit': UPSTREAM_COMMIT, 'source_url': SOURCE_URL,
            'source_sha256': SOURCE_SHA256, 'bun_version': BUN_VERSION, 'bun_integrity': BUN_INTEGRITY,
            'patch_sha256': hashlib.sha256(patch.read_bytes()).hexdigest(),
            'models_fixture': MODELS_FIXTURE, 'models_sha256': MODELS_SHA256,
            'models_catalog_qualified': False,
            'lockfile_sha256': hashlib.sha256((source / 'bun.lock').read_bytes()).hexdigest()}
        return publish_client(source / 'packages/opencode/dist/opencode-linux-x64/bin/opencode', destination, provenance)


def main() -> int:
    """Build from pinned source; --archive accepts a preseeded exact source tarball."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path, default=Path('/usr/local/bin/opencode'))
    parser.add_argument('--archive', type=Path, help='Preseeded exact upstream source archive')
    parser.add_argument('--bun-archive', type=Path, help='Preseeded exact pinned Bun archive')
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='llaminar-opencode-input-') as temporary:
        archive = args.archive or Path(temporary) / 'source.tgz'
        if args.archive is None:
            download(SOURCE_URL, archive)
        result = install_client(archive, args.destination, bun_archive=args.bun_archive)
    print(json.dumps(result), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
