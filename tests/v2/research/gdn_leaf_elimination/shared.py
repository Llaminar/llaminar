#!/usr/bin/env python3
"""Qualify projection-eliding CPU leaves against the new production M=1 readout.

Retain raw accepted-leaf operands, preserve dense internal state arithmetic,
exercise every acceptance/continuation boundary, and authenticate each serial
oracle call against an independent FP32 interpreter. This is a supplied-core
experiment; canonical real-model qualification is a separate production gate.
"""
import argparse
import ctypes
from dataclasses import dataclass
import json
import os
from pathlib import Path

import numpy as np
from exact import check_guard, digest, guarded, inputs
from harness import Arithmetic, Comparison, LCG, Precision, topology


@dataclass(frozen=True)
class RawUpdate:
    """Owned operands for the single deferred accepted leaf, with no correction."""
    key: np.ndarray
    value: np.ndarray
    decay: np.float32
    beta: np.float32


@dataclass(frozen=True)
class State:
    """One immutable dense parent and at most one unresolved accepted update."""
    parent: np.ndarray
    pending: RawUpdate | None = None


class Native:
    """Bind only the shared-readout experiment's explicit native contracts."""
    def __init__(self, library):
        """Load the selected core/bridge, with no substitute on ABI/control failure."""
        self.library = ctypes.CDLL(str(Path(library).resolve(strict=True)))
        self.pointer = ctypes.POINTER(ctypes.c_float)
        definitions = {
            'gdn_shared_serial': [ctypes.c_int]*2 + [self.pointer]*7,
            'gdn_shared_prepare': [ctypes.c_int] + [self.pointer]*4,
            'gdn_shared_group': [ctypes.c_int]*4 + [self.pointer]*10,
            'gdn_shared_resolve': [ctypes.c_int]*2 + [self.pointer]*3 + [ctypes.c_float]*2 + [self.pointer],
        }
        for name, types in definitions.items():
            function = getattr(self.library, name)
            function.argtypes, function.restype = types, ctypes.c_int

    def ptr(self, array):
        """Require exact caller-owned contiguous FP32 storage."""
        if array.dtype != np.float32 or not array.flags.c_contiguous:
            raise ValueError('native arrays must be contiguous FP32')
        return array.ctypes.data_as(self.pointer)

    def call(self, name, *args):
        """Invalidate the experiment on any rejected contract or oracle mismatch."""
        code = getattr(self.library, name)(*args)
        if code != 1:
            raise AssertionError(f'{name} failed: {code}')

    def prepare(self, q, gates):
        """Prepare inputs using the production-matched scalar gate expressions."""
        scaled, decay_beta = np.empty_like(q), np.empty(2, np.float32)
        self.call('gdn_shared_prepare', q.size, *(self.ptr(x) for x in (q, gates, scaled, decay_beta)))
        return scaled, decay_beta

    def serial(self, parent, q, k, v, gates):
        """Return actual production M=1 state, new output, and interpreted legacy output."""
        state = parent.copy()
        out, old = np.empty_like(v), np.empty_like(v)
        self.call('gdn_shared_serial', q.size, v.size, *(self.ptr(x) for x in (q, k, v, gates, state, out, old)))
        return state, out, old

    def resolve(self, state):
        """Materialize a pending raw update at a real dense-consumer boundary."""
        if state.pending is None:
            return state.parent.copy()
        update = state.pending
        result, guard = guarded(state.parent.shape)
        self.call('gdn_shared_resolve', *state.parent.shape,
                  *(self.ptr(x) for x in (state.parent, update.key, update.value)),
                  update.decay, update.beta, self.ptr(result))
        check_guard(guard)
        return result

    def group(self, parent, q, k, v, decay, beta, internal):
        """Evaluate real siblings with guards and poisoned leaf-state storage."""
        rows, dk = q.shape
        dv = v.shape[1]
        adjusted, ag = guarded((rows, dk))
        eta, eg = guarded((rows,))
        output, og = guarded((rows, dv))
        snapshots, sg = guarded((rows, dk, dv))
        before = parent.tobytes()
        self.call('gdn_shared_group', int(internal), rows, dk, dv, *(self.ptr(x) for x in
            (parent, q, k, v, decay, beta, adjusted, eta, output, snapshots)))
        for guard in (ag, eg, og, sg):
            check_guard(guard)
        if not internal and np.any(snapshots.view(np.uint32) != 0x7FC12345):
            raise AssertionError('a leaf published dense state')
        if before != parent.tobytes():
            raise AssertionError('verification mutated its parent')
        return output, snapshots


def evaluate(native, root, parents, q, k, v, gates):
    """Resolve ancestors once, then group each parent's internal and leaf children."""
    if not parents:
        return np.empty((0, v.shape[1]), np.float32), {}, {'query_actions': 0, 'key_actions': 0}
    dense = {-1: native.resolve(root)}
    internal = set(parents) - {-1}
    output = np.empty_like(v)
    prepared = [native.prepare(q[i], gates[i]) for i in range(len(parents))]
    scaled = np.stack([p[0] for p in prepared])
    decay = np.array([p[1][0] for p in prepared], np.float32)
    beta = np.array([p[1][1] for p in prepared], np.float32)
    committed = {}
    for parent in (-1, *range(len(parents))):
        for publish in (True, False):
            ids = [i for i, p in enumerate(parents) if p == parent and (i in internal) == publish]
            if not ids:
                continue
            values, states = native.group(dense[parent], scaled[ids], k[ids], v[ids], decay[ids], beta[ids], publish)
            output[ids] = values
            for local, i in enumerate(ids):
                if publish:
                    dense[i] = states[local].copy()
                    committed[i] = State(dense[i].copy())
                else:
                    committed[i] = State(dense[parent].copy(), RawUpdate(k[i].copy(), v[i].copy(), decay[i], beta[i]))
    return output, committed, {'query_actions': len(parents), 'key_actions': len(internal) + int(root.pending is not None)}


def strict_case(native, shape, regime, dk, dv, seed):
    """Compare all endpoints and continued outputs over four accepted rounds."""
    rng, arithmetic = LCG(seed), Arithmetic(Precision.FP32)
    parent = rng.vector(arithmetic, dk*dv, 128).reshape(dk, dv)
    root = State(parent.copy())
    metrics = {name: Comparison() for name in ('outputs', 'states', 'continuation', 'terminal', 'legacy_drift')}
    receipts = []
    for round_index in range(4):
        parents = topology(shape, rng)
        q, k, v, gates = inputs(rng, parents, dk, dv, regime)
        serial_states, expected = {-1: parent}, np.empty_like(v)
        for i, ancestor in enumerate(parents):
            state, out, old = native.serial(serial_states[ancestor], q[i], k[i], v[i], gates[i])
            serial_states[i], expected[i] = state, out
            metrics['legacy_drift'].observe(old, out, {'round': round_index, 'node': i})
        actual, states, receipt = evaluate(native, root, parents, q, k, v, gates)
        receipts.append(receipt)
        metrics['outputs'].observe(expected, actual, {'round': round_index})
        for endpoint in (None, *range(len(parents))):
            selected = root if endpoint is None else states[endpoint]
            expected_state = parent if endpoint is None else serial_states[endpoint]
            metrics['states'].observe(expected_state, native.resolve(selected), {'endpoint': endpoint})
            # Independent raw operands ensure acceptance cannot reuse an old readout.
            cq, ck, cv, cg = inputs(rng, (-1, -1, -1), dk, dv, 'ordinary')
            co, continued, _ = evaluate(native, selected, (-1, -1, -1), cq, ck, cv, cg)
            for i in range(3):
                cs, expected_out, _ = native.serial(expected_state, cq[i], ck[i], cv[i], cg[i])
                metrics['continuation'].observe(expected_out, co[i], {'endpoint': endpoint, 'child': i})
                metrics['terminal'].observe(cs, native.resolve(continued[i]), {'endpoint': endpoint, 'child': i})
        if parents:
            endpoint = len(parents) - 1
            root, parent = states[endpoint], serial_states[endpoint]
            # Acceptance owns raw operands independently of reusable candidate banks.
            q.fill(77); k.fill(88); v.fill(99); gates.fill(11)
            metrics['terminal'].observe(parent, native.resolve(root), 'dirty candidate reuse')
    for name, metric in metrics.items():
        if name != 'legacy_drift' and metric.mismatches:
            raise AssertionError(f'{shape}/{regime}/{dk}/{dv}: {name}: {metric.first}')
    return {'shape': shape, 'regime': regime, 'dk': dk, 'dv': dv,
            'comparisons': {k: vars(v) for k, v in metrics.items()}, 'work': receipts}


def live_rows(native, seed):
    """Poison retained banks across every shallow M and large/small/empty replay."""
    rng, arithmetic = LCG(seed), Arithmetic(Precision.FP32)
    dk, dv, capacity = 7, 16, 63
    parent = rng.vector(arithmetic, dk*dv, 128).reshape(dk, dv)
    q, k, v, gates = inputs(rng, (-1,)*capacity, dk, dv, 'ordinary')
    prepared = [native.prepare(q[i], gates[i]) for i in range(capacity)]
    scaled = np.stack([x[0] for x in prepared])
    decay = np.array([x[1][0] for x in prepared], np.float32)
    beta = np.array([x[1][1] for x in prepared], np.float32)
    buffers = [guarded(shape) for shape in ((capacity, dk), (capacity,), (capacity, dv), (capacity, dk, dv))]
    adjusted, eta, output, snapshots = [x[0] for x in buffers]
    expected = [native.serial(parent, q[i], k[i], v[i], gates[i])[:2] for i in range(capacity)]
    metric = Comparison()
    inventory = [*range(1, 17), 31, 63, 2, 0, 16, 1, 63]
    for rows in inventory:
        for internal in (0, 1):
            for array, _ in buffers:
                array.view(np.uint32).fill(0x7FC12345)
            native.call('gdn_shared_group', internal, rows, dk, dv, *(native.ptr(x) for x in
                (parent, scaled, k, v, decay, beta, adjusted, eta, output, snapshots)))
            for i in range(rows):
                metric.observe(expected[i][1], output[i], {'rows': rows, 'node': i, 'internal': internal})
                if internal:
                    metric.observe(expected[i][0], snapshots[i], {'rows': rows, 'node': i})
            for array, guard in buffers:
                check_guard(guard)
                first_unused = 0 if array is snapshots and not internal else rows
                if np.any(array[first_unused:].view(np.uint32) != 0x7FC12345):
                    raise AssertionError('publication exceeded live extent')
    if metric.mismatches:
        raise AssertionError(f'live-row replay: {metric.first}')
    return {'row_counts': inventory, 'comparison': vars(metric)}


def run(library, seed):
    """Retain identities and a strict native core sweep; never claim model qualification."""
    native = Native(library)
    cases = [strict_case(native, shape, regime, dk, dv, seed)
             for dk, dv in ((7, 16), (64, 64), (128, 128))
             for shape in ('empty', 'singleton', 'chain', 'star', 'ternary', 'irregular', 'early_leaves')
             for regime in ('ordinary', 'zero', 'endpoint', 'tiny')]
    manifest = json.loads(library.with_name('native-build.json').read_text())
    return {'schema': 'gdn-shared-readout-v1', 'byte_equivalence_observed': True,
            'reference': 'new production CPU M=1 readout; legacy output bytes intentionally differ',
            'seed': seed, 'build': manifest, 'requested_isa': os.environ.get('LLAMINAR_ISA_LEVEL', 'native'),
            'core_sha256': digest(manifest['core_library']), 'bridge_sha256': digest(library),
            'driver_sha256': digest(__file__), 'cases': cases, 'live_rows': live_rows(native, seed),
            'production_qualified': False}


def main():
    """Run mandatory byte checks and optionally retain the complete report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', required=True, type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--seed', type=int, default=20261024)
    args = parser.parse_args()
    result = run(args.library.resolve(strict=True), args.seed)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'cases': len(result['cases']), 'byte_equivalence_observed': True}))


if __name__ == '__main__':
    main()
