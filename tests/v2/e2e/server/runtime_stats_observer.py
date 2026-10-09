#!/usr/bin/env python3
"""Retain public runtime-statistics observations throughout a live workload.

A failed diagnostic poll must remain a failed qualification observation, but
must not discard all later evidence or cancel the model's current generation.
Transport failures are journaled immediately and the next scheduled poll reads
the same endpoint. Invalid statistics terminate observation with an explicit
failure. The owner decides when the workload ends; this observer has no model
request handle, shutdown authority, or generation deadline.
"""
from __future__ import annotations

from contextlib import ExitStack
import json
import math
from pathlib import Path
import threading
import time
from typing import Callable
from urllib.error import URLError


class RuntimeStatsObserver:
    """Journal validated snapshots and every failed poll without hiding either."""

    def __init__(self, read: Callable[[], dict], output: Path, *, interval_seconds: float = 1.0):
        """Bind one endpoint reader, an exclusive journal, and the sampling cadence.

        ``read`` owns the finite HTTP observation timeout and validates each
        response's schema. It must only perform a read of the running server.
        ``output`` receives successful snapshots in the existing audit format;
        failures use a separate journal so snapshot consumers cannot silently
        mistake a missing response for a valid empty statistics object.
        """
        if not math.isfinite(interval_seconds) or interval_seconds < 0:
            raise ValueError("Statistics polling interval must be finite and nonnegative")
        self.read, self.output = read, Path(output)
        self.interval_seconds = interval_seconds
        self.stopped = threading.Event()
        self.thread = threading.Thread(target=self._observe, daemon=True)
        self.count = self.active_count = self.failed_polls = 0
        self.maximum_seconds = self.maximum_attempt_seconds = 0.0
        self.error: str | None = None
        self.previous: dict | None = None
        self._files = ExitStack()

    def _record_failure(self, error: BaseException, kind: str, elapsed: float) -> None:
        """Flush the failure before another poll; the first failure stays authoritative."""
        self.failed_polls += 1
        if self.error is None:
            self.error = repr(error)
        self.maximum_attempt_seconds = max(self.maximum_attempt_seconds, elapsed)
        self._errors.write(json.dumps({
            "observed": time.time(), "http_seconds": elapsed,
            "attempt": self.count + self.failed_polls, "kind": kind,
            "error": repr(error),
        }) + "\n")
        self._errors.flush()

    def _accept(self, value: dict, elapsed: float) -> None:
        """Check epoch-local churn against the last successful observation and journal it."""
        previous = self.previous
        if previous is not None and value['epoch'] == previous['epoch']:
            current = value['prefix_cache']['storage']['churn']
            earlier = previous['prefix_cache']['storage']['churn']
            for kind, counts in current.items():
                if kind == 'scope':
                    continue
                for field in ('operations', 'bytes'):
                    if counts[field] < earlier[kind][field]:
                        raise ValueError(f"Statistics churn regressed: {kind}.{field}")
        self._snapshots.write(json.dumps({
            'observed': time.time(), 'http_seconds': elapsed, 'stats': value,
        }) + '\n')
        self._snapshots.flush()
        self.previous = value
        self.count += 1
        self.active_count += value['requests']['active'] > 0
        self.maximum_seconds = max(self.maximum_seconds, elapsed)
        self.maximum_attempt_seconds = max(self.maximum_attempt_seconds, elapsed)

    def _observe(self) -> None:
        """Continue scheduled reads after transport errors while retaining a failed result."""
        while not self.stopped.is_set():
            started = time.monotonic()
            try:
                value = self.read()
            except (OSError, URLError) as error:
                self._record_failure(error, 'transport', time.monotonic() - started)
            except BaseException as error:
                self._record_failure(error, 'validation', time.monotonic() - started)
                return
            else:
                try:
                    self._accept(value, time.monotonic() - started)
                except BaseException as error:
                    self._record_failure(error, 'validation', time.monotonic() - started)
                    return
            self.stopped.wait(self.interval_seconds)

    def __enter__(self) -> RuntimeStatsObserver:
        """Admit journal ownership before launching a workload or observation thread."""
        try:
            self._snapshots = self._files.enter_context(self.output.open('x'))
            self._errors = self._files.enter_context(self.output.with_suffix('.errors.jsonl').open('x'))
            self.thread.start()
        except BaseException:
            self._files.close()
            raise
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        """Join the bounded reader and report failure without replacing a workload exception."""
        self.stopped.set()
        self.thread.join()
        self._files.close()
        report = {
            'snapshots': self.count, 'while_active': self.active_count,
            'failed_polls': self.failed_polls, 'maximum_http_seconds': self.maximum_seconds,
            'maximum_attempt_seconds': self.maximum_attempt_seconds, 'error': self.error,
            'passed': self.error is None and self.active_count > 0,
        }
        self.output.with_suffix('.result.json').write_text(json.dumps(report, indent=2) + '\n')
        if exc is None and not report['passed']:
            raise AssertionError(report)
