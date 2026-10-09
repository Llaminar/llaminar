#!/usr/bin/env python3
"""Authenticate every app response against its native generated token sequence.

The audit owns no generation or retry. It joins retained proxy exchanges to a
complete server log by prompt/completion counts and the observed request window,
then asks the image's metadata-only decoder for native bytes and parser replay.
Independent inverse rendering checks tool arguments; exact comparisons also
cover ordinary text and reasoning, including emoji and tool-free compaction.
No control response may be skipped implicitly and Python optimization cannot
disable any evidence check. Only log/code metadata is fingerprinted; model and
prefix-cache payloads are never hashed.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import uuid

from opencode_tool_stress import authenticate_native_qwen_calls, requested_streaming_usage

TOKEN = re.compile(r'^\[([^]]+)\].*\[ChatCompletion/token\] path=(\w+) index=(\d+) token=(-?\d+) forced=(\w+) disposition=(\w+)')
PREFIX = re.compile(r'Prefix cache summary .*requested_tokens=(\d+) matched_tokens=(\d+)')


def native_groups(lines, started: float) -> list[dict]:
    """Validate one complete trace, including midnight rollover and terminal rows."""
    if type(started) not in (int, float) or not math.isfinite(started) or started <= 0:
        raise ValueError('Native audit requires the recorded server start time')
    day = datetime.datetime.fromtimestamp(started, datetime.timezone.utc).date()
    previous_clock, current, groups = None, None, []
    for line in lines:
        token = TOKEN.search(line)
        if token:
            clock, path, index, identity, forced, disposition = token.groups()
            parsed = datetime.time.fromisoformat(clock)
            if previous_clock is not None and parsed < previous_clock:
                if previous_clock.hour < 23 or parsed.hour > 0:
                    raise ValueError('Native token timestamps moved backwards within a day')
                day += datetime.timedelta(days=1)
            previous_clock = parsed
            observed = datetime.datetime.combine(day, parsed, datetime.timezone.utc).timestamp()
            index, identity = int(index), int(identity)
            if (forced != 'false' or path not in ('stream', 'nonstream')
                    or disposition not in ('text', 'stop_token', 'runner_complete')
                    or not -(2**31) <= identity < 2**31
                    or (disposition == 'text' and identity < 0)):
                raise ValueError('Unknown or forced native token disposition')
            if index == 0:
                if current is not None:
                    raise ValueError('Native generation is missing its terminal summary')
                current = {'tokens': [], 'all_ids': [], 'path': path, 'first': observed, 'terminal_seen': False}
            if current is None or index != len(current['all_ids']) or current['path'] != path:
                raise ValueError('Native token index/path is missing, duplicated or reordered')
            if current['terminal_seen']:
                raise ValueError('Native generation emitted a token after its terminal disposition')
            current['all_ids'].append(identity)
            current['last'] = observed
            if disposition == 'text':
                current['tokens'].append(identity)
            else:
                current['terminal_seen'] = True
        elif prefix := PREFIX.search(line):
            if current is None:
                raise ValueError('Prefix summary has no native generation')
            prompt, matched = int(prefix[1]), int(prefix[2])
            if not 0 <= matched <= prompt or prompt == 0:
                raise ValueError('Invalid native prompt/prefix counts')
            current.update(prompt_tokens=prompt, matched_tokens=matched)
            groups.append(current)
            current = None
        elif '[ChatCompletion/token]' in line:
            raise ValueError('Malformed native token trace line')
    if current is not None or not groups:
        raise ValueError('Native trace is empty or has an unfinished generation')
    return groups


def thinking_override(request: dict) -> bool | None:
    """Keep the public top-level/nested reasoning contract in decoder evidence."""
    nested = request.get('chat_template_kwargs', {})
    if not isinstance(nested, dict):
        raise ValueError('Invalid chat template kwargs')
    values = [owner['enable_thinking'] for owner in (request, nested) if 'enable_thinking' in owner]
    if any(type(value) is not bool for value in values) or len(set(values)) > 1:
        raise ValueError('Conflicting or non-boolean reasoning policy')
    return values[0] if values else None


def wire_exchange(identity: str, exchange: dict) -> dict:
    """Reconstruct authenticated streaming channels without dropping no-tool turns."""
    if exchange.get('response', {}).get('status') != 200:
        raise ValueError(f'{identity}: native audit found a failed HTTP response')
    usage = requested_streaming_usage(exchange)
    if usage is None or usage['completion_tokens'] <= 0:
        raise ValueError(f'{identity}: requested terminal streaming usage is missing')
    started, elapsed = exchange['started'], exchange['elapsed_seconds']
    if any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0 for value in (started, elapsed)):
        raise ValueError('Invalid proxy request time window')
    calls, reasoning, content = {}, '', ''
    for line in exchange['response']['body'].splitlines():
        if not line.startswith('data:') or line[5:].strip() == '[DONE]':
            continue
        event = json.loads(line[5:])
        for choice in event.get('choices', []):
            delta = choice.get('delta', {})
            reasoning += delta.get('reasoning_content', '')
            content += delta.get('content', '')
            for call in delta.get('tool_calls', []):
                index = call['index']
                if type(index) is not int or index < 0:
                    raise ValueError('Invalid tool delta index')
                row = calls.setdefault(index, {'name': '', 'arguments': ''})
                function = call.get('function', {})
                row['name'] += function.get('name', '')
                row['arguments'] += function.get('arguments', '')
    if set(calls) != set(range(len(calls))):
        raise ValueError('Streaming tool indices are not contiguous')
    wire = [{'name': calls[index]['name'], 'arguments': json.loads(calls[index]['arguments'])}
            for index in range(len(calls))]
    if any(not row['name'] or not isinstance(row['arguments'], dict) for row in wire):
        raise ValueError('Incomplete tool name or non-object arguments')
    return {'id': identity, 'start': started, 'end': started + elapsed, 'usage': usage,
            'tools': exchange['request'].get('tools', []), 'wire_calls': wire,
            'reasoning': reasoning, 'content': content,
            'enable_thinking': thinking_override(exchange['request'])}


def join_responses(groups: list[dict], exchanges: list[dict]) -> tuple[list[dict], list[dict]]:
    """Require a one-to-one join; counts alone cannot authenticate a neighboring turn."""
    if not exchanges or len({row['id'] for row in exchanges}) != len(exchanges):
        raise ValueError('Missing or duplicated proxy response identity')
    used, joined, decoder_input = set(), [], []
    for exchange in sorted(exchanges, key=lambda row: row['end']):
        candidates = [(index, group) for index, group in enumerate(groups)
            if index not in used and group['path'] == 'stream'
            and group['prompt_tokens'] == exchange['usage']['prompt_tokens']
            and len(group['all_ids']) == exchange['usage']['completion_tokens']
            # Native INFO logs have millisecond precision. The fixed 10-ms
            # join tolerance is clock quantization, not a request timeout.
            and group['first'] >= exchange['start'] - .01
            and group['last'] <= exchange['end'] + .01]
        if len(candidates) != 1:
            raise ValueError(f"{exchange['id']}: expected one native response, got {len(candidates)}")
        index, group = candidates[0]
        used.add(index)
        joined.append({**exchange, 'native_group': index, 'native': group})
        decoder_input.append({'id': exchange['id'], 'tokens': group['tokens'], 'tools': exchange['tools'],
                              'enable_thinking': exchange['enable_thinking']})
    if len(used) != len(groups):
        raise ValueError('Unaccounted native responses; controls require their own recorded exchanges')
    return joined, decoder_input


def compare_decoded(joined: list[dict], decoded: list[dict]) -> dict:
    """Compare complete channels and independently authenticate native XML values."""
    if not joined or len(joined) != len(decoded):
        raise ValueError('Missing native decoder response evidence')
    changes, verified = [], 0
    for original, replay in zip(joined, decoded):
        if original['id'] != replay['id']:
            raise ValueError('Decoder reordered or changed response identities')
        if replay['tool_format'] != 'qwen_3_xml':
            raise ValueError('Independent native tool audit requires the Qwen XML grammar')
        independent = authenticate_native_qwen_calls(replay['native_content'], original['wire_calls'], original['tools'])
        verified += independent['verified_calls']
        mismatches = [field for field in ('content', 'reasoning') if original[field] != replay[field]]
        if original['wire_calls'] != replay['calls']:
            mismatches.append('tool_calls')
        if not independent['passed']:
            mismatches.append('native_arguments')
        if mismatches:
            changes.append({'id': original['id'], 'channels': mismatches, 'independent_arguments': independent})
    return {'complete': True, 'passed': not changes, 'joined_responses': len(joined),
            'verified_native_calls': verified, 'changed_responses': len(changes), 'changes': changes}


def decoder_command(model: Path, *, decoder: Path | None = None,
                    image: str | None = None) -> list[str]:
    """Select native argv or owned-container arguments for one explicit decoder.

    Local diagnostics name their built executable. Published-image checks run
    the helper shipped in the same runtime ID, with its model shard directory
    read-only and stdin attached. No binary overlay or alternate image may
    substitute for an unavailable decoder. This metadata-only process does
    not receive GPU devices or open prefix-cache storage.
    """
    if (decoder is None) == (image is None):
        raise ValueError('Choose exactly one native decoder executable or image')
    if decoder is not None:
        return [str(decoder), str(model)]
    if not isinstance(image, str) or not re.fullmatch(r'sha256:[0-9a-f]{64}', image):
        raise ValueError('Native decoder requires an immutable Docker image ID')
    root = Path(__file__).resolve().parents[4]
    sys.path.insert(0, str(root / 'scripts/ci'))
    from docker_paths import mounts
    model = model.resolve(strict=True)
    if not model.is_file():
        raise ValueError('Native decoder model must be an existing GGUF entry file')
    return ['--interactive', '--network', 'none',
        '--cap-add', 'SYS_NICE', '--security-opt', 'seccomp=unconfined',
        '--env', 'LLAMINAR_LOG_LEVEL=ERROR',
        *mounts([(model.parent, '/model', True)]),
        '--entrypoint', '/usr/local/bin/llaminar_native_tool_evidence_decoder',
        image, '/model/' + model.name]


def decode_records(model: Path, records: list[dict], output: Path, *,
                   decoder: Path | None = None, image: str | None = None) -> None:
    """Keep native JSON output only after its exact producer has retired.

    Image mode retains the daemon's completed stdout/stderr before removal;
    short decoder exits cannot lose their final bytes in an attach stream.
    Local diagnostics use the explicitly selected executable directly.
    """
    arguments = decoder_command(model, decoder=decoder, image=image)
    if image is None:
        with (output / 'decoded.json').open('w') as stdout, (output / 'decoder.stderr').open('w') as stderr:
            subprocess.run(arguments, input=json.dumps(records), text=True, stdout=stdout,
                stderr=stderr, check=True, env={**os.environ, 'LLAMINAR_LOG_LEVEL': 'ERROR'})
        return
    from opencode_cell_runtime import ContainerOwners, run_retired
    owners = ContainerOwners()
    try:
        result, _ = run_retired(owners, 'llaminar-native-audit-' + uuid.uuid4().hex,
            arguments, log=output / 'decoder.log', input_text=json.dumps(records))
        (output / 'decoded.json').write_text(result.stdout)
        (output / 'decoder.stderr').write_text(result.stderr)
        result.check_returncode()
    finally:
        (output / 'decoder-retirement.json').write_text(json.dumps(owners.close(), indent=2) + '\n')


def main() -> int:
    """Retain input bindings and decoded evidence without changing original results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exchanges', type=Path, required=True)
    parser.add_argument('--server-log', type=Path, required=True)
    parser.add_argument('--server-start-unix', type=float, required=True)
    native = parser.add_mutually_exclusive_group(required=True)
    native.add_argument('--decoder', type=Path)
    native.add_argument('--decoder-image', help='Exact immutable serving-image ID for its installed decoder')
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    result = {'complete': False, 'passed': False}
    try:
        fingerprint, size = hashlib.sha256(), 0

        def log_lines(stream):
            """Fingerprint only retained log bytes while avoiding a full-log copy."""
            nonlocal size
            for raw in stream:
                fingerprint.update(raw)
                size += len(raw)
                yield raw.decode('utf-8')

        with args.server_log.open('rb') as stream:
            groups = native_groups(log_lines(stream), args.server_start_unix)
        binding = {'bytes': size, 'sha256': fingerprint.hexdigest(), 'observed': time.time()}
        (args.output / 'source-log-binding.json').write_text(json.dumps(binding, indent=2) + '\n')
        exchanges = [wire_exchange(str(path.relative_to(args.exchanges)), json.loads(path.read_text()))
                     for path in sorted(args.exchanges.glob('session-*/exchange-*.json'))
                     if not path.name.endswith('.progress.json')]
        joined, records = join_responses(groups, exchanges)
        (args.output / 'joined.json').write_text(json.dumps(joined, ensure_ascii=False) + '\n')
        decode_records(args.model, records, args.output, decoder=args.decoder, image=args.decoder_image)
        result = compare_decoded(joined, json.loads((args.output / 'decoded.json').read_text()))
    except (ValueError, OSError, KeyError, TypeError, subprocess.SubprocessError) as error:
        result['error'] = str(error)
    (args.output / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({key: value for key, value in result.items() if key != 'changes'}), flush=True)
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
