#!/usr/bin/env python3
"""Replay real native CLI turn lifecycles through a scripted local HTTP provider.

No model or accelerator runs here. Positive streaming usage reproduces the actual
TP4 compaction boundary. Every request retains the production completion budget;
the provider-overflow control must preserve its error and nonzero CLI exit.
"""
import json
import os
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def qualify(binary: str, mode: str) -> dict:
    """Check one independent session, including cross-process next-turn resume."""
    requests = []

    class Handler(BaseHTTPRequestHandler):
        """Serve deterministic SDK SSE bytes while retaining every real request."""

        def log_message(self, *_args):
            """Keep stdout exclusively for the machine-readable proof."""

        def send(self, status: int, content_type: str, data: bytes):
            """Send one bounded response with exact byte framing."""
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            """Replay terminal, tool, summary and genuine overflow observations."""
            requests.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
            ordinal = len(requests)
            if ordinal == 1 and mode == 'provider_overflow':
                self.send(400, 'application/json', json.dumps({'error': {
                    'code': 'context_length_exceeded', 'message': 'maximum context length exceeded',
                }}).encode())
                return
            delta = {'content': 'increment complete' if ordinal == 1 else
                     'summary' if ordinal == 2 else 'next phase complete'}
            finish = 'stop'
            if ordinal == 1 and mode in ('tool_calls', 'tool_stop'):
                delta = {'tool_calls': [{'index': 0, 'id': 'call_regression', 'type': 'function',
                    'function': {'name': 'bash', 'arguments': json.dumps({
                        'command': 'echo regression', 'description': 'Verify tool progress'})}}]}
                finish = 'tool_calls' if mode == 'tool_calls' else 'stop'
            usage = {'prompt_tokens': 15150 if ordinal == 1 else 1000,
                     'completion_tokens': 89 if ordinal == 1 else 10,
                     'total_tokens': 15239 if ordinal == 1 else 1010}
            chunks = [
                {'choices': [{'index': 0, 'delta': {'role': 'assistant'}}]},
                {'choices': [{'index': 0, 'delta': delta}]},
                {'choices': [{'index': 0, 'delta': {}, 'finish_reason': finish}]},
                {'choices': [], 'usage': usage},
            ]
            data = ''.join('data: ' + json.dumps({'id': 'chatcmpl-regression',
                'object': 'chat.completion.chunk', **chunk}) + '\n\n' for chunk in chunks)
            self.send(200, 'text/event-stream', (data + 'data: [DONE]\n\n').encode())

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        with tempfile.TemporaryDirectory(prefix='opencode-client-regression-') as root:
            config = {'autoupdate': False, 'share': 'disabled', 'enabled_providers': ['test'],
                'lsp': False, 'formatter': False, 'permission': {'bash': 'allow'},
                'provider': {'test': {'npm': '@ai-sdk/openai-compatible', 'name': 'Regression',
                    'options': {'baseURL': f'http://127.0.0.1:{server.server_port}/v1',
                                'apiKey': 'test', 'timeout': False},
                    'models': {'regression': {'name': 'Regression', 'tool_call': True,
                                             'limit': {'context': 47168, 'output': 32768}}}}}}
            env = {**os.environ, 'OPENCODE_CONFIG_CONTENT': json.dumps(config),
                'OPENCODE_CONFIG_DIR': root + '/config', 'XDG_CONFIG_HOME': root + '/config',
                'XDG_DATA_HOME': root + '/data', 'XDG_CACHE_HOME': root + '/cache',
                'XDG_STATE_HOME': root + '/state', 'OPENCODE_DISABLE_AUTOUPDATE': 'true',
                'OPENCODE_DISABLE_MODELS_FETCH': 'true'}
            env.pop('OPENCODE_CONFIG', None)
            command = [binary, 'run', '--pure', '--dir', root, '--format', 'json',
                       '--model', 'test/regression', '--title', 'compaction-regression']
            first = subprocess.run(command + ['first increment; finish this turn' if mode == 'stop'
                else 'use bash once then finish'], env=env, text=True, capture_output=True, timeout=30)
            if first.returncode and mode != 'provider_overflow':
                raise ValueError(f'{mode}: native client failed: {first.stderr[-3000:]} {first.stdout[-4000:]}')
            events = [json.loads(line) for line in first.stdout.splitlines() if line.startswith('{')]
            session = next(event['sessionID'] for event in events if 'sessionID' in event)
            first_calls = len(requests)
            if mode == 'stop':
                second = subprocess.run(command + ['--session', session, 'next phase'],
                    env=env, text=True, capture_output=True, timeout=30)
                if second.returncode:
                    raise ValueError(f'Next user phase failed: {second.stderr[-3000:]}')
            passed = (first_calls == (1 if mode == 'stop' else 3) and len(requests) == 3
                      and all(request.get('max_tokens') == 32000 for request in requests))
            # The middle request must be the real tool-free compaction exchange.
            passed &= len(requests) == 3 and not requests[1].get('tools')
            if mode == 'provider_overflow':
                passed &= first.returncode == 1 and any(event['type'] == 'error'
                    and event['error']['name'] == 'ContextOverflowError' for event in events)
            else:
                passed &= not any(event['type'] == 'error' for event in events)
            if mode in ('tool_calls', 'tool_stop'):
                passed &= any(event['type'] == 'tool_use' and event['part']['tool'] == 'bash'
                    and event['part']['state']['status'] == 'completed' for event in events)
            return {'mode': mode, 'first_calls': first_calls, 'total_calls': len(requests),
                    'max_tokens': [request.get('max_tokens') for request in requests], 'passed': passed}
    finally:
        server.shutdown()
        server.server_close()
        worker.join()


def main() -> int:
    """Reject the original completion-preemption bug and preserve three controls."""
    results = [qualify(sys.argv[1], mode) for mode in
               ('stop', 'tool_calls', 'tool_stop', 'provider_overflow')]
    print(json.dumps(results, indent=2))
    return 0 if all(result['passed'] for result in results) else 1


if __name__ == '__main__':
    raise SystemExit(main())
