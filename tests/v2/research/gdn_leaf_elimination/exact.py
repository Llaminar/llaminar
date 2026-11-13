#!/usr/bin/env python3
"""Test compact, exactly rounded leaf updates against real CPU serial decode.

This variant saves dense leaf-state publication, not leaf key projections.
Sibling rows execute in one native tile; internal states remain dense, so no
reassociated checkpoint/history expansion can change their FP32 rounding.
Accepted leaves retain one exact correction and their immutable dense parent.
The next verification fuses pending materialization into its first key pass.
"""

import argparse
import ctypes
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from harness import Arithmetic, Comparison, LCG, Precision, topology


def digest(path):
    """Hash an explicitly named artifact with a bounded read buffer."""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def guarded(shape):
    """Return poisoned live storage surrounded by immutable bit-pattern guards."""
    count = int(np.prod(shape))
    backing = np.full(count + 16, 0x7FC12345, dtype=np.uint32)
    return backing[8:8 + count].view(np.float32).reshape(shape), backing


def check_guard(backing):
    """Require that neither side of an actual live publication was touched."""
    if np.any(backing[:8] != 0x7FC12345) or np.any(backing[-8:] != 0x7FC12345):
        raise AssertionError("native publication crossed its live extent")


@dataclass(frozen=True)
class Update:
    """An accepted byte-exact update; no value vector or query remains live."""

    decay: np.float32
    key: np.ndarray
    correction: np.ndarray


@dataclass(frozen=True)
class State:
    """One owned dense parent and at most one accepted pending update."""

    parent: np.ndarray
    pending: Update | None = None


class Native:
    """Bind the production oracle and fixed-order native candidate explicitly."""

    def __init__(self, library):
        """Require every native entrypoint; no substitution on missing support."""
        self.library = ctypes.CDLL(str(Path(library).resolve(strict=True)))
        self.pointer = ctypes.POINTER(ctypes.c_float)
        definitions = {
            "gdn_leaf_cpu_compare": [ctypes.c_int] * 2 + [self.pointer] * 8,
            "gdn_leaf_cpu_prepare": [ctypes.c_int] + [self.pointer] * 4,
            "gdn_leaf_exact_outputs": [ctypes.c_int] * 3 + [self.pointer] * 8,
            "gdn_leaf_exact_snapshots": [ctypes.c_int] * 3 + [self.pointer] * 9,
            "gdn_leaf_exact_after_pending": [ctypes.c_int] * 3 + [self.pointer] * 8 +
                [ctypes.c_float] + [self.pointer] * 3,
            "gdn_leaf_exact_pending_snapshots": [ctypes.c_int] * 3 + [self.pointer] * 8 +
                [ctypes.c_float] + [self.pointer] * 4,
            "gdn_leaf_exact_materialize": [ctypes.c_int] * 2 + [self.pointer, ctypes.c_float] +
                [self.pointer] * 3,
        }
        for name, arguments in definitions.items():
            function = getattr(self.library, name)
            function.argtypes, function.restype = arguments, ctypes.c_int

    def ptr(self, array):
        """Reject implicit copies and strides at the native storage boundary."""
        if array.dtype != np.float32 or not array.flags.c_contiguous:
            raise ValueError("native operand must be contiguous FP32")
        return array.ctypes.data_as(self.pointer)

    def call(self, name, *arguments):
        """A native contract or oracle-control failure invalidates the experiment."""
        result = getattr(self.library, name)(*arguments)
        if result != 1:
            raise AssertionError(f"{name} rejected comparison: {result}")

    def prepare(self, q, gates):
        """Use the same authenticated query scaling and gate conversion as the oracle."""
        scaled, decay_beta = np.empty_like(q), np.empty(2, dtype=np.float32)
        self.call("gdn_leaf_cpu_prepare", len(q), *(self.ptr(x) for x in (q, gates, scaled, decay_beta)))
        return scaled, decay_beta

    def serial(self, parent, q, k, v, gates):
        """Execute the actual production M=1 step and authenticate its FMA control."""
        state = parent.copy()
        output, fused, wide = (np.empty_like(v) for _ in range(3))
        self.call("gdn_leaf_cpu_compare", len(q), len(v),
                  *(self.ptr(x) for x in (q, k, v, gates, state, output, fused, wide)))
        return state, output

    def materialize(self, state):
        """Resolve a compact accepted update only at an explicit dense-consumer boundary."""
        if state.pending is None:
            return state.parent.copy()
        pending = state.pending
        output, guard = guarded(state.parent.shape)
        self.call("gdn_leaf_exact_materialize", *state.parent.shape, self.ptr(state.parent),
                  pending.decay, self.ptr(pending.key), self.ptr(pending.correction), self.ptr(output))
        check_guard(guard)
        return output

    def group(self, parent, q, k, v, decay, beta, *, snapshots):
        """Run one real sibling group with poisoned outputs and exact guard checks."""
        rows, dk = q.shape
        dv = v.shape[1]
        if (parent.parent.shape != (dk, dv) or k.shape != q.shape or
                v.shape != (rows, dv) or decay.shape != (rows,) or beta.shape != (rows,)):
            raise ValueError("incompatible native group geometry")
        if parent.pending is not None and (parent.pending.key.shape != (dk,) or
                                           parent.pending.correction.shape != (dv,)):
            raise ValueError("pending update geometry differs from its parent")
        output, output_guard = guarded((rows, dv))
        corrections, correction_guard = guarded((rows, dv))
        dense, dense_guard = guarded((rows, dk, dv)) if snapshots else (None, None)
        before = parent.parent.copy()
        arrays = (parent.parent, q, k, v, decay, beta, output, corrections)
        args = [rows, dk, dv, *(self.ptr(x) for x in arrays)]
        resolved = parent.parent
        if parent.pending is not None:
            resolved, resolved_guard = guarded(parent.parent.shape)
            args += [parent.pending.decay, self.ptr(parent.pending.key),
                     self.ptr(parent.pending.correction), self.ptr(resolved)]
            name = "gdn_leaf_exact_pending_snapshots" if snapshots else "gdn_leaf_exact_after_pending"
        else:
            name = "gdn_leaf_exact_snapshots" if snapshots else "gdn_leaf_exact_outputs"
        if snapshots:
            args.append(self.ptr(dense))
        self.call(name, *args)
        for guard in (output_guard, correction_guard, dense_guard):
            if guard is not None:
                check_guard(guard)
        if parent.pending is not None:
            check_guard(resolved_guard)
            if rows == 0:
                if np.any(resolved.view(np.uint32) != 0x7FC12345):
                    raise AssertionError("empty verification materialized a pending state")
                resolved = parent.parent
        if parent.parent.tobytes() != before.tobytes():
            raise AssertionError("verification mutated its immutable parent")
        return output, corrections, dense, resolved


@dataclass(frozen=True)
class Evaluation:
    """Own candidate results until one accepted endpoint is published."""

    parents: tuple[int, ...]
    states: dict[int, np.ndarray]
    updates: tuple[Update, ...]
    outputs: tuple[np.ndarray, ...]
    original: State

    def commit(self, endpoint):
        """Retain only an accepted dense internal node or one compact leaf update."""
        if endpoint is None:
            return self.original
        if endpoint < 0 or endpoint >= len(self.parents):
            raise ValueError("accepted endpoint outside this transaction")
        if endpoint in self.states:
            return State(self.states[endpoint].copy())
        update = self.updates[endpoint]
        # Copy the single selected parent so subsequent candidate-buffer reuse
        # cannot overwrite the pending update's sole state authority.
        return State(self.states[self.parents[endpoint]].copy(),
                     Update(update.decay, update.key.copy(), update.correction.copy()))


def evaluate(native, history, parents, q, k, v, gates):
    """Batch siblings and materialize exactly the nodes with current descendants."""
    if any(parent < -1 or parent >= i for i, parent in enumerate(parents)):
        raise ValueError("tree must be topologically ordered")
    prepared = [native.prepare(q[i], gates[i]) for i in range(len(parents))]
    scaled = np.stack([item[0] for item in prepared]) if prepared else q.copy()
    decay = np.array([item[1][0] for item in prepared], dtype=np.float32)
    beta = np.array([item[1][1] for item in prepared], dtype=np.float32)
    internal = set(parents) - {-1}
    states = {-1: history.parent}
    outputs, updates = [None] * len(parents), [None] * len(parents)
    for parent_index in (-1, *range(len(parents))):
        children = [i for i, parent in enumerate(parents) if parent == parent_index]
        if not children:
            continue
        parent = history if parent_index == -1 else State(states[parent_index])
        for snapshots in (True, False):
            indices = [i for i in children if (i in internal) == snapshots]
            if not indices:
                continue
            output, correction, dense, resolved = native.group(parent, scaled[indices], k[indices],
                v[indices], decay[indices], beta[indices], snapshots=snapshots)
            states[parent_index] = resolved
            parent = State(resolved)
            for row, i in enumerate(indices):
                outputs[i] = output[row].copy()
                updates[i] = Update(decay[i], k[i].copy(), correction[row].copy())
                if snapshots:
                    states[i] = dense[row].copy()
    return Evaluation(parents, states, tuple(updates), tuple(outputs), history)


def inputs(rng, parents, dk, dv, regime):
    """Generate raw production operands, including rounded endpoint and small gates."""
    arithmetic = Arithmetic(Precision.FP32)
    count = len(parents)
    q = rng.vector(arithmetic, count * dk).reshape(count, dk)
    k = rng.vector(arithmetic, count * dk).reshape(count, dk)
    for row in k:
        row /= np.float32(np.sqrt(np.sum(row * row, dtype=np.float32) + np.float32(1e-6)))
    v = rng.vector(arithmetic, count * dv).reshape(count, dv)
    gates = np.array([[rng.integer(257) / 64 - 2, rng.integer(257) / 64 - 2,
                       -(rng.integer(127) + 1) / 64, 0] for _ in parents], dtype=np.float32).reshape(count, 4)
    if regime == "endpoint":
        gates[:, 2] = 0
        gates[:, 1] = [100 if i % 2 else -100 for i in range(count)]
    elif regime == "zero":
        gates[::2, 2] = -1000
    elif regime == "tiny":
        gates[:, 0] = 0
        gates[:, 2] = -130
    return q, k, v, gates


def run_exact(library, seed=20261024):
    """Require byte equivalence at every output, endpoint, flush and continued round."""
    native = Native(library)
    arithmetic = Arithmetic(Precision.FP32)
    results = []
    for dk, dv in ((7, 16), (64, 64), (128, 128)):
        for shape in ("empty", "singleton", "chain", "star", "ternary", "irregular", "early_leaves"):
            for regime in ("ordinary", "zero", "endpoint", "tiny"):
                rng = LCG(seed)
                serial_state = rng.vector(arithmetic, dk * dv, 128).reshape(dk, dv)
                history = State(serial_state.copy())
                metrics = {name: Comparison() for name in ("outputs", "accepted_states", "continuation", "flush")}
                nodes, internal_states, fused_pending = 0, 0, 0
                for round_index in range(4):
                    parents = topology(shape, rng)
                    q, k, v, gates = inputs(rng, parents, dk, dv, regime)
                    expected_states, expected_outputs = [], []
                    for i, parent in enumerate(parents):
                        state = serial_state if parent < 0 else expected_states[parent]
                        updated, output = native.serial(state, q[i], k[i], v[i], gates[i])
                        expected_states.append(updated)
                        expected_outputs.append(output)
                    evaluated = evaluate(native, history, parents, q, k, v, gates)
                    nodes += len(parents)
                    internal_states += len(set(parents) - {-1})
                    fused_pending += int(history.pending is not None and bool(parents))
                    for i, output in enumerate(expected_outputs):
                        metrics["outputs"].observe(output, evaluated.outputs[i], {"round": round_index, "node": i})
                    next_inputs = inputs(rng, (-1, -1, -1), dk, dv, "ordinary")
                    for endpoint in (None, *range(len(parents))):
                        where = {"round": round_index, "endpoint": endpoint}
                        accepted = evaluated.commit(endpoint)
                        expected = serial_state if endpoint is None else expected_states[endpoint]
                        metrics["accepted_states"].observe(expected, native.materialize(accepted), where)
                        continued = evaluate(native, accepted, (-1, -1, -1), *next_inputs)
                        for i in range(3):
                            next_state, next_output = native.serial(expected, *(x[i] for x in next_inputs))
                            metrics["continuation"].observe(next_output, continued.outputs[i], where)
                            metrics["flush"].observe(next_state, native.materialize(continued.commit(i)), where)
                    internal = set(parents) - {-1}
                    leaves = sorted(set(range(len(parents))) - internal)
                    endpoint = (None if not leaves or round_index == 2 else
                                min(internal) if round_index == 3 and internal else leaves[-1])
                    history = evaluated.commit(endpoint)
                    if endpoint is not None:
                        serial_state = expected_states[endpoint]
                    metrics["flush"].observe(serial_state, native.materialize(history), {"round": round_index})
                if any(metric.mismatches for metric in metrics.values()):
                    raise AssertionError(f"exact candidate failed: {dk,dv,shape,regime}: {metrics}")
                results.append({"dk": dk, "dv": dv, "shape": shape, "regime": regime,
                                "nodes": nodes, "internal_state_writes": internal_states,
                                "fused_pending_writes": fused_pending,
                                "comparisons": {name: vars(value) for name, value in metrics.items()}})
    # Projection totality and changing live extents use the same retained banks.
    totality = run_live_rows(native, seed)
    minimal = run_minimal_witness(native)
    library = Path(library).resolve(strict=True)
    manifest = json.loads(library.with_name("native-build.json").read_text())
    return {"schema": "gdn-exact-compact-leaf-research-v1", "seed": seed,
            "byte_equivalence_observed": True, "production_qualified": False,
            "saved_work": "dense leaf-state writes; leaf key projections are still computed",
            "codegen_isa": "AVX2 four-sibling/eight-value physical tile",
            "requested_oracle_isa": os.environ.get("LLAMINAR_ISA_LEVEL", "native"),
            "scope": "single CPU thread, supplied FP32 core operands, synthetic cases",
            "build": manifest, "core_sha256": digest(manifest["core_library"]),
            "bridge_sha256": digest(library), "driver_sha256": digest(__file__),
            "cases": results, "live_rows": totality, "minimal_witness": minimal}


def run_minimal_witness(native):
    """Recover both serial results that a sole adjusted-query action cannot distinguish."""
    q = np.ones((1, 1), dtype=np.float32)
    k = q.copy()
    v = np.full((1, 16), 2 ** -25, dtype=np.float32)
    gates = np.array([[0, 100, 0, 0]], dtype=np.float32)
    observed = []
    for value in (0, 1):
        state = np.full((1, 16), value, dtype=np.float32)
        expected_state, expected_output = native.serial(state, q[0], k[0], v[0], gates[0])
        evaluated = evaluate(native, State(state), (-1,), q, k, v, gates)
        comparison = Comparison()
        comparison.observe(expected_output, evaluated.outputs[0], {"parent": value})
        comparison.observe(expected_state, native.materialize(evaluated.commit(0)), {"parent": value})
        if comparison.mismatches:
            raise AssertionError("compact candidate failed the minimal adjusted-query witness")
        observed.append({"parent": value, "serial": float(expected_output[0]),
                         "compact": float(evaluated.outputs[0][0]), "comparison": vars(comparison)})
    return observed


def run_live_rows(native, seed):
    """Prove arbitrary M and large/small/empty replay with persistent poisoned storage."""
    rng, arithmetic = LCG(seed), Arithmetic(Precision.FP32)
    capacity, dk, dv = 63, 7, 16
    q, k, v, gates = inputs(rng, (-1,) * capacity, dk, dv, "ordinary")
    prepared = [native.prepare(q[i], gates[i]) for i in range(capacity)]
    scaled = np.stack([item[0] for item in prepared])
    decay = np.array([item[1][0] for item in prepared], dtype=np.float32)
    beta = np.array([item[1][1] for item in prepared], dtype=np.float32)
    parent = rng.vector(arithmetic, dk * dv, 128).reshape(dk, dv)
    output, output_guard = guarded((capacity, dv))
    corrections, correction_guard = guarded((capacity, dv))
    snapshots, snapshot_guard = guarded((capacity, dk, dv))
    metric = Comparison()
    retained = None
    rows_inventory = [*range(1, 17), 31, 63, 2, 0, 16, 1, 63]
    for rows in rows_inventory:
        for publication in ("outputs", "snapshots"):
            for array in (output, corrections, snapshots):
                array.view(np.uint32).fill(0x7FC12345)
            arguments = [rows, dk, dv, *(native.ptr(x) for x in
                (parent, scaled, k, v, decay, beta, output, corrections))]
            if publication == "snapshots":
                arguments.append(native.ptr(snapshots))
            native.call("gdn_leaf_exact_" + publication, *arguments)
            if retained is not None:
                metric.observe(retained[1], native.materialize(retained[0]), "retained after dirty replay")
            for i in range(rows):
                expected_state, expected_output = native.serial(parent, q[i], k[i], v[i], gates[i])
                metric.observe(expected_output, output[i], {"rows": rows, "node": i, "mode": publication})
                if publication == "snapshots":
                    metric.observe(expected_state, snapshots[i], {"rows": rows, "node": i})
                else:
                    restored = native.materialize(State(parent, Update(decay[i], k[i], corrections[i])))
                    metric.observe(expected_state, restored, {"rows": rows, "node": i})
            if rows and publication == "outputs":
                retained = (State(parent.copy(), Update(decay[rows - 1], k[rows - 1].copy(),
                                  corrections[rows - 1].copy())), expected_state.copy())
            for array, guard in ((output, output_guard), (corrections, correction_guard), (snapshots, snapshot_guard)):
                check_guard(guard)
                first_unused = 0 if array is snapshots and publication == "outputs" else rows
                if np.any(array[first_unused:].view(np.uint32) != 0x7FC12345):
                    raise AssertionError("candidate wrote inactive capacity or compact leaf states")
    if retained is not None:
        native.group(retained[0], scaled[:0], k[:0], v[:0], decay[:0], beta[:0], snapshots=False)
    if metric.mismatches:
        raise AssertionError("live-row replay failed byte equivalence")
    return {"row_counts": rows_inventory, "comparison": vars(metric)}


def main():
    """Run the mandatory strict experiment and retain all comparison counts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=20261024)
    args = parser.parse_args()
    result = run_exact(args.library, args.seed)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"cases": len(result["cases"]), "byte_equivalence_observed": True,
                      "live_row_replay": result["live_rows"]}, indent=2))


if __name__ == "__main__":
    main()
