"""Replay immutable upstream observations through the actual native CPU kernel.

This transport adapter does not calculate expected values. It uses the same
strict fixture decoder as the independent reference tests, sends recorded source
and input bytes to the C++ candidate, and checks every transformed, A4 and output
byte. Arithmetic candidates exercise ordinary and existing-team entry. Tensor
candidates exercise single/fused production engines, retained full/partial/empty
work, reset, a complete-output-row weight slice and singleton verifier output
with ordinary projection workspace. Explicit transport entry IDs
prevent a lower-level candidate from certifying the tensor interface. No compiler,
source download, model, accelerator or upstream DSO is needed at test time.
"""

import argparse
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile

from .test_w4a4 import decode_trace


MAGIC = 0x52344157


def observations(case: dict, interface: str) -> list[tuple[str, bytes]]:
    """Select recorded bytes, preserving the driver's explicit inactive initial state."""
    fields = ('rotated', 'a4', 'output')
    if interface == 'arithmetic':
        return [(name, case[name].tobytes()) for name in fields]
    tokens = case['input'].shape[0]
    result = []

    def selected(array, live: int, guard: bytes) -> bytes:
        """Keep observed row prefixes; never calculate a numerical expectation."""
        original = array.tobytes()
        if len(original) % tokens:
            raise AssertionError('recorded array does not preserve original row coordinates')
        row_bytes = len(original) // tokens
        tail = (tokens - live) * row_bytes
        if tail % len(guard):
            raise AssertionError('inactive initializer does not fit the recorded element extent')
        return original[:live * row_bytes] + guard * (tail // len(guard))

    for phase, live in enumerate((tokens, tokens - 1, 0, tokens)):
        result.append((f'{phase}/live', struct.pack('<i', live)))
        for name in fields:
            guard = b'\x45\x23\xc1\x7f' if name == 'output' else b'\xa5'
            wanted = (case[name].tobytes() if name == 'rotated' and case['type_id'] == 44
                      else selected(case[name], live, guard))
            result.append((f'{phase}/{name}', wanted))
        if interface == 'fused-tensor':
            first = 1 if case['weight_shape'][0] > 1 else 0
            result.append((f'{phase}/sliced_output', selected(case['output'][:, first:], live, b'\x45\x23\xc1\x7f')))
        result.append((f'{phase}/singleton_output', selected(case['output'], live, b'\x45\x23\xc1\x7f')))
        result.append((f'{phase}/validity', bytes([1]) * live + bytes(tokens - live)))
    return result


def main() -> None:
    """Require complete recorded observations and report the exact failing field."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executable', type=Path, required=True)
    parser.add_argument('--trace', type=Path,
                        default=Path(__file__).parent / 'fixtures/w4a4/native_trace_v1.json')
    parser.add_argument('--threads', type=int, nargs='+', default=list(range(1, 9)))
    parser.add_argument('--interface', choices=('arithmetic', 'gemm'), default='arithmetic')
    args = parser.parse_args()
    if not args.threads or any(threads <= 0 for threads in args.threads):
        parser.error('thread counts must be positive')
    executable = args.executable.resolve(strict=True)
    if not executable.is_file() or not os.access(executable, os.X_OK):
        parser.error('the native replay candidate must be an existing executable')
    cases = decode_trace(json.loads(args.trace.read_text()))
    runs = 0
    with tempfile.TemporaryDirectory(prefix='llaminar-w4a4-replay-') as temporary:
        input_path, output_path = (Path(temporary) / name for name in ('input.bin', 'actual.bin'))
        for case in cases:
            outputs, width = case['weight_shape']
            tokens = case['input'].shape[0]
            entries = ((0, 'ordinary'), (1, 'existing-team')) if args.interface == 'arithmetic' else (
                (2, 'tensor'), (3, 'fused-tensor'))
            for threads in args.threads:
                for entry, name in entries:
                    identity = f"{case['name']}/threads={threads}/entry={name}"
                    expected = observations(case, 'arithmetic' if args.interface == 'arithmetic' else name)
                    header = struct.pack('<7I', MAGIC, 1, case['type_id'], outputs, width, tokens, entry)
                    input_path.write_bytes(header + case['weights'].tobytes() + case['input'].tobytes())
                    output_path.unlink(missing_ok=True)
                    environment = {**os.environ, 'OMP_NUM_THREADS': str(threads), 'OMP_DYNAMIC': 'FALSE'}
                    candidate = subprocess.run([str(executable), str(input_path), str(output_path)],
                                               env=environment, capture_output=True, timeout=10)
                    if candidate.returncode:
                        raise AssertionError(f'{identity}: candidate rejected recorded input: '
                                             f'{candidate.stderr.decode(errors="replace")}')
                    actual = output_path.read_bytes()
                    if actual[:len(header)] != header:
                        raise AssertionError(f'{identity}: incomplete or incompatible result header')
                    offset = len(header)
                    for field, wanted in expected:
                        observed = actual[offset:offset + len(wanted)]
                        if observed != wanted:
                            first = next((index for index, (a, b) in enumerate(zip(observed, wanted))
                                          if a != b), min(len(observed), len(wanted)))
                            raise AssertionError(f'{identity}: {field} differs at byte {first}; '
                                                 f'actual={observed[first:first + 8].hex()} '
                                                 f'expected={wanted[first:first + 8].hex()}')
                        offset += len(wanted)
                    if len(actual) != offset:
                        raise AssertionError(f'{identity}: trailing unrecorded result bytes')
                    runs += 1
        # Authenticate the candidate boundary too. These negative controls
        # change transport/configuration only, never recorded expected bytes.
        wrong_entry = 2 if args.interface == 'arithmetic' else 0
        wrong_header = struct.pack('<7I', MAGIC, 1, case['type_id'], outputs, width, tokens, wrong_entry)
        payload = header + case['weights'].tobytes() + case['input'].tobytes()
        for name, invalid in (
            ('wrong-interface', wrong_header + payload[len(header):]),
            ('incomplete', payload[:-1]),
            ('trailing', payload + bytes([0])),
            ('wrong-policy', payload[:4] + struct.pack('<I', 2) + payload[8:]),
        ):
            input_path.write_bytes(invalid)
            output_path.unlink(missing_ok=True)
            candidate = subprocess.run([str(executable), str(input_path), str(output_path)],
                                       env=environment, capture_output=True, timeout=10)
            if candidate.returncode == 0 or output_path.exists():
                raise AssertionError(f'{name}: candidate published success for an incompatible recording')
    # Tensor phases record both the selected entry and singleton verifier entry.
    phases = 1 if args.interface == 'arithmetic' else 8
    print(f'Native W4A4 CPU {args.interface} replay: {len(cases)} recorded cases, '
          f'{runs} candidate invocations, {runs * phases} exact retained executions')


if __name__ == '__main__':
    main()
