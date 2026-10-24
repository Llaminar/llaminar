#!/usr/bin/env python3
"""Reproducible GDN leaf-elimination experiment; never a production verifier.

The dense recurrence, full triangular correction solve, and internal-only
factored evaluator share operands, but use independent evaluation algorithms.
Exact Fraction arithmetic checks the algebra and accepted-history lifecycle.
Explicit NumPy FP32/FP64 operations expose reassociation error without BLAS or
implicit mixed precision. Numerical closeness and byte equivalence are separate
observations. This is a core-recurrence experiment, not model certification or
a kernel performance benchmark; convolution and acceptance sampling are inputs.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np


class Precision(str, Enum):
    """Explicit arithmetic domain; exact mode never converts through a float."""

    EXACT = "exact"
    FP64 = "fp64"
    FP32 = "fp32"


@dataclass(frozen=True)
class Arithmetic:
    """Own scalar conversion and fixed, increasing-key reduction order."""

    precision: Precision

    @property
    def dtype(self):
        """Return the storage dtype corresponding to this arithmetic domain."""
        return {Precision.EXACT: object, Precision.FP64: np.float64,
                Precision.FP32: np.float32}[self.precision]

    def scalar(self, numerator, denominator=1):
        """Construct one rounded scalar, or one exact rational."""
        if self.precision is Precision.EXACT:
            return Fraction(int(numerator), int(denominator))
        return self.dtype(numerator / denominator)

    def zeros(self, shape):
        """Allocate diagnostic storage with a correctly typed additive zero."""
        return np.full(shape, self.scalar(0), dtype=self.dtype)

    def dot(self, left, right):
        """Reduce products in increasing index order, without implicit FMA."""
        total = self.scalar(0)
        for x, y in zip(left, right, strict=True):
            total = total + x * y
        return total

    def action(self, state, vector):
        """Apply a transposed-storage state [key, value] to one key vector."""
        out = self.zeros(state.shape[1])
        for row, coefficient in zip(state, vector, strict=True):
            out = out + row * coefficient
        return out


@dataclass(frozen=True)
class Node:
    """Supplied, branch-correct core operands with already-computed gates."""

    parent: int
    q: np.ndarray
    k: np.ndarray
    v: np.ndarray
    decay: Any
    beta: Any


@dataclass(frozen=True)
class Tree:
    """A topologically ordered forest under one implicit checkpoint root."""

    nodes: tuple[Node, ...]
    paths: tuple[tuple[int, ...], ...] = field(init=False)
    internal: frozenset[int] = field(init=False)

    def __post_init__(self):
        """Reject invalid ancestry and incompatible per-head dimensions."""
        paths = []
        internal = set()
        geometry = None
        for i, node in enumerate(self.nodes):
            if node.parent < -1 or node.parent >= i:
                raise ValueError("parent must be -1 or a preceding node")
            if node.q.ndim != 1 or node.k.shape != node.q.shape or node.v.ndim != 1:
                raise ValueError("expected matching key/query vectors and a value vector")
            current = (len(node.k), len(node.v))
            if 0 in current or (geometry is not None and geometry != current):
                raise ValueError("all nodes must have one positive head geometry")
            geometry = current
            paths.append((paths[node.parent] if node.parent >= 0 else ()) + (i,))
            if node.parent >= 0:
                internal.add(node.parent)
        object.__setattr__(self, "paths", tuple(paths))
        object.__setattr__(self, "internal", frozenset(internal))


def dense_step(arithmetic, state, node):
    """Independent five-stage serial oracle, including explicit state rounding."""
    decayed = state * node.decay
    correction = (node.v - arithmetic.action(decayed, node.k)) * node.beta
    updated = decayed + node.k[:, None] * correction[None, :]
    return updated, arithmetic.action(updated, node.q)


def dense_tree(arithmetic, checkpoint, tree):
    """Materialize each branch independently; allowed only in this diagnostic."""
    states, outputs = [], []
    for node in tree.nodes:
        parent = checkpoint if node.parent < 0 else states[node.parent]
        state, output = dense_step(arithmetic, parent, node)
        states.append(state)
        outputs.append(output)
    return states, outputs


@dataclass(frozen=True)
class Record:
    """One resolved additive update, owned by the accepted path."""

    decay: Any
    k: np.ndarray
    correction: np.ndarray


@dataclass(frozen=True)
class Pending:
    """Own the raw update of the accepted leaf; its query is no longer live."""

    decay: Any
    beta: Any
    k: np.ndarray
    v: np.ndarray


@dataclass
class Work:
    """Count actual checkpoint actions, not allocated capacity or estimated FLOPs."""

    output_actions: int = 0
    internal_key_actions: int = 0
    pending_key_actions: int = 0
    resolved_history_terms: int = 0
    flushes: int = 0
    flush_pending_resolutions: int = 0

    @property
    def projection_vectors(self):
        """Return the measured action count for equation (9)."""
        return self.output_actions + self.internal_key_actions + self.pending_key_actions


@dataclass(frozen=True)
class History:
    """Bounded resolved history plus at most one pending accepted leaf.

    Arrays are immutable by convention inside this diagnostic: operations
    allocate new arrays, and rejection returns the original History object.
    This representation is not installed as an inference state authority.
    """

    checkpoint: np.ndarray
    resolved: tuple[Record, ...] = ()
    pending: Pending | None = None
    capacity: int = 8

    def __post_init__(self):
        """Reject impossible storage rather than silently exceeding the bound."""
        if self.capacity < 0 or len(self.resolved) > self.capacity:
            raise ValueError("resolved history exceeds its nonnegative capacity")
        if self.checkpoint.ndim != 2 or 0 in self.checkpoint.shape:
            raise ValueError("checkpoint must have positive key/value dimensions")

    def action(self, arithmetic, vector, records=(), work=None):
        """Read the checkpoint and rank-one records without forming their state."""
        out = arithmetic.action(self.checkpoint, vector)
        for record in self.resolved + tuple(records):
            out = out * record.decay + record.correction * arithmetic.dot(record.k, vector)
            if work is not None:
                work.resolved_history_terms += 1
        return out

    def resolve_pending(self, arithmetic, work):
        """Resolve exactly one old pending key, with no redundant output readout."""
        if self.pending is None:
            return ()
        node = self.pending
        work.pending_key_actions += 1
        correction = node.beta * (node.v - node.decay * self.action(arithmetic, node.k, work=work))
        return (Record(node.decay, node.k.copy(), correction),)

    def materialize(self, arithmetic):
        """Diagnostic/dense-consumer boundary, resolving any remaining pending leaf."""
        state = self.checkpoint.copy()
        for record in self.resolved:
            state = state * record.decay + record.k[:, None] * record.correction[None, :]
        if self.pending is not None:
            # A real dense consumer must pay for this state update. Do not
            # pretend that deferred work disappeared at the end of a run.
            node = self.pending
            state = state * node.decay
            correction = node.beta * (node.v - arithmetic.action(state, node.k))
            state = state + node.k[:, None] * correction[None, :]
        return state

    def flush(self, arithmetic, work):
        """Materialize the complete accepted prefix and retire its update records."""
        work.flushes += 1
        work.flush_pending_resolutions += int(self.pending is not None)
        return History(self.materialize(arithmetic), capacity=self.capacity)


@dataclass(frozen=True)
class Evaluation:
    """Candidate outputs and only those corrections with descendant consumers."""

    tree: Tree
    history: History
    pending_records: tuple[Record, ...]
    corrections: dict[int, Record]
    outputs: tuple[np.ndarray, ...]
    work: Work

    def commit(self, arithmetic, endpoint, *, work=None):
        """Publish only the selected path, compacting resolved storage when full."""
        if endpoint is None:
            return self.history
        if endpoint < 0 or endpoint >= len(self.tree.nodes):
            raise ValueError("accepted endpoint is outside the evaluated tree")
        records = self.history.resolved + self.pending_records
        pending = None
        for i in self.tree.paths[endpoint]:
            if i in self.corrections:
                records += (self.corrections[i],)
            else:
                node = self.tree.nodes[i]
                pending = Pending(node.decay, node.beta, node.k.copy(), node.v.copy())
        state = self.history.checkpoint
        # Flush only a prefix of resolved records. The sole new leaf stays
        # pending; flushing a terminal request explicitly resolves it later.
        excess = max(0, len(records) - self.history.capacity)
        if excess:
            if work is not None:
                work.flushes += 1
            for record in records[:excess]:
                state = state * record.decay + record.k[:, None] * record.correction[None, :]
            records = records[excess:]
        return History(state, records, pending, self.history.capacity)


def leaf_evaluate(arithmetic, history, tree, *, adjust_all=False):
    """Solve internal/pending rows and compute leaves by the adjusted-query identity."""
    work = Work()
    # Empty verification has no next consumer and must preserve pending state.
    pending = history.resolve_pending(arithmetic, work) if tree.nodes else ()
    corrections, outputs = {}, []
    for i, node in enumerate(tree.nodes):
        records = pending + tuple(corrections[j] for j in tree.paths[i][:-1])
        if i in tree.internal:
            work.internal_key_actions += 1
            u = node.beta * (node.v - node.decay * history.action(arithmetic, node.k, records, work))
            corrections[i] = Record(node.decay, node.k.copy(), u)
        work.output_actions += 1
        if i not in tree.internal or adjust_all:
            eta = node.beta * arithmetic.dot(node.k, node.q)
            adjusted = node.decay * (node.q - eta * node.k)
            output = history.action(arithmetic, adjusted, records, work) + eta * node.v
        else:
            output = (node.decay * history.action(arithmetic, node.q, records, work)
                      + corrections[i].correction * arithmetic.dot(node.k, node.q))
        outputs.append(output)
    expected = len(tree.nodes) + len(tree.internal) + int(bool(tree.nodes) and history.pending is not None)
    if work.projection_vectors != expected or len(corrections) != len(tree.internal):
        raise AssertionError("leaf elimination performed unexpected checkpoint/correction work")
    return Evaluation(tree, history, pending, corrections, tuple(outputs), work)


def all_node_evaluate(arithmetic, checkpoint, tree):
    """Independent forward solve of (I+G)U=R, then B+CU; no adjusted queries."""
    corrections, outputs = {}, []
    for i, node in enumerate(tree.nodes):
        path = tree.paths[i]
        decay = arithmetic.scalar(1)
        weights = {}
        for j in reversed(path):
            weights[j] = decay
            decay = decay * tree.nodes[j].decay
        u = node.beta * (node.v - decay * arithmetic.action(checkpoint, node.k))
        for j in path[:-1]:
            coefficient = node.beta * weights[j] * arithmetic.dot(node.k, tree.nodes[j].k)
            u = u - coefficient * corrections[j].correction
        corrections[i] = Record(node.decay, node.k.copy(), u)
        output = decay * arithmetic.action(checkpoint, node.q)
        for j in path:
            coefficient = weights[j] * arithmetic.dot(node.q, tree.nodes[j].k)
            output = output + coefficient * corrections[j].correction
        outputs.append(output)
    return corrections, outputs


class LCG:
    """Fully specified 32-bit generator, independent of NumPy RNG versions."""

    def __init__(self, seed):
        """Initialize the reproducible unsigned state."""
        self.state = seed & 0xFFFFFFFF

    def integer(self, limit):
        """Return a bounded integer after the Numerical Recipes LCG transition."""
        self.state = (1664525 * self.state + 1013904223) & 0xFFFFFFFF
        return (self.state >> 8) % limit

    def vector(self, arithmetic, size, denominator=64):
        """Generate a vector of exactly specified rational source values."""
        return np.array([arithmetic.scalar(self.integer(129) - 64, denominator)
                         for _ in range(size)], dtype=arithmetic.dtype)


def topology(name, rng, *, small=False):
    """Construct chains, forests, early leaves and deterministic irregular trees."""
    count = {"empty": 0, "singleton": 1, "chain": 5 if small else 9,
             "star": 5 if small else 19, "ternary": 12 if small else 39,
             "irregular": 8 if small else 27, "early_leaves": 7}[name]
    if name == "chain":
        return tuple(i - 1 for i in range(count))
    if name == "star":
        return (-1,) * count
    if name == "ternary":
        return tuple((i // 3) - 1 for i in range(count))
    if name == "irregular":
        return tuple(rng.integer(i + 1) - 1 for i in range(count))
    if name == "early_leaves":
        return (-1, 0, 0, 2, 2, 4, 4)
    return (-1,) * count


def operands(arithmetic, rng, parents, dk, dv, regime):
    """Supply normalized floating keys or rational keys and adversarial gate regimes."""
    nodes = []
    parallel = rng.vector(arithmetic, dk)
    for i, parent in enumerate(parents):
        q, k, v = (rng.vector(arithmetic, d) for d in (dk, dk, dv))
        if regime == "parallel":
            k = parallel + k * arithmetic.scalar(1, 4096)
        if arithmetic.precision is not Precision.EXACT:
            q = q / arithmetic.dtype(math.sqrt(float(arithmetic.dot(q, q)) + 1e-6) * math.sqrt(dk))
            k = k / arithmetic.dtype(math.sqrt(float(arithmetic.dot(k, k)) + 1e-6))
        else:
            k, q = k / arithmetic.scalar(dk), q / arithmetic.scalar(dk)
        decay = arithmetic.scalar(rng.integer(15) + 1, 16)
        beta = arithmetic.scalar(rng.integer(15) + 1, 16)
        if regime == "zero" and i % 3 == 0:
            decay = arithmetic.scalar(0)
        if regime == "endpoint":
            decay, beta = arithmetic.scalar(1), arithmetic.scalar(i % 2)
        if regime == "tiny":
            decay = arithmetic.scalar(1, 1 << (120 if i % 2 else 20))
        nodes.append(Node(parent, q, k, v, decay, beta))
    return Tree(tuple(nodes))


@dataclass
class Comparison:
    """Keep exact mismatches separate from finite numeric error diagnostics."""

    elements: int = 0
    mismatches: int = 0
    max_abs: float = 0.0
    max_ulp: int = 0
    first: dict | None = None

    def observe(self, expected, actual, where):
        """Compare every scalar; fail on nonfinite values rather than hiding NaNs."""
        expected, actual = np.asarray(expected), np.asarray(actual)
        if expected.shape != actual.shape or expected.dtype != actual.dtype:
            raise AssertionError("comparison shape/dtype mismatch")
        self.elements += expected.size
        if expected.dtype == object:
            unequal = expected != actual
            errors = np.array([float(abs(a - b)) for a, b in
                               zip(expected.flat, actual.flat, strict=True)])
            bits_expected = bits_actual = None
        else:
            if not np.isfinite(expected).all() or not np.isfinite(actual).all():
                raise AssertionError("nonfinite recurrence result")
            integer = np.uint32 if expected.dtype == np.float32 else np.uint64
            bits_expected, bits_actual = expected.view(integer), actual.view(integer)
            unequal = bits_expected != bits_actual
            errors = np.abs(expected.astype(np.float64) - actual.astype(np.float64))
            # Order IEEE encodings monotonically; retain signed-zero byte drift.
            sign = integer(1 << (expected.itemsize * 8 - 1))
            order_a = np.where(bits_expected & sign, ~bits_expected, bits_expected ^ sign)
            order_b = np.where(bits_actual & sign, ~bits_actual, bits_actual ^ sign)
            distance = np.maximum(order_a, order_b) - np.minimum(order_a, order_b)
            self.max_ulp = max(self.max_ulp, int(distance.max(initial=0)))
        self.max_abs = max(self.max_abs, float(errors.max(initial=0)))
        self.mismatches += int(np.count_nonzero(unequal))
        if self.first is None and np.any(unequal):
            index = tuple(int(x) for x in np.argwhere(unequal)[0])
            self.first = {"where": where, "index": index,
                          "expected": str(expected[index]), "actual": str(actual[index])}
            if bits_expected is not None:
                self.first.update(expected_bits=hex(int(bits_expected[index])),
                                  actual_bits=hex(int(bits_actual[index])))


def run_case(spec):
    """Exercise every accepted endpoint and several rounds with a bounded history."""
    precision, shape, regime, dk, dv, seed, adjust_all, rounds = spec
    arithmetic = Arithmetic(Precision(precision))
    rng = LCG(seed)
    state = rng.vector(arithmetic, dk * dv, 128).reshape(dk, dv)
    history = History(state.copy(), capacity=3)
    metrics = {name: Comparison() for name in ("all_node_outputs", "leaf_outputs",
               "all_node_states", "accepted_states", "continuation_outputs", "flushed_states")}
    counters = {"candidates": 0, "endpoints": 0, "projection_vectors": 0,
                "baseline_projection_vectors": 0, "correction_rows": 0,
                "flushes": 0, "terminal_pending_resolutions": 0}
    for round_index in range(rounds):
        parents = topology(shape, rng, small=precision == "exact")
        tree = operands(arithmetic, rng, parents, dk, dv, regime)
        states, outputs = dense_tree(arithmetic, state, tree)
        all_corrections, all_outputs = all_node_evaluate(arithmetic, state, tree)
        evaluated = leaf_evaluate(arithmetic, history, tree, adjust_all=adjust_all)
        for i in range(len(tree.nodes)):
            where = {"round": round_index, "endpoint": i}
            metrics["all_node_outputs"].observe(outputs[i], all_outputs[i], where)
            metrics["leaf_outputs"].observe(outputs[i], evaluated.outputs[i], where)
        continuation = operands(arithmetic, rng, (-1,), dk, dv, regime)
        for endpoint in (None, *range(len(tree.nodes))):
            where = {"round": round_index, "endpoint": endpoint}
            committed = evaluated.commit(arithmetic, endpoint)
            expected_state = state if endpoint is None else states[endpoint]
            metrics["accepted_states"].observe(expected_state, committed.materialize(arithmetic), where)
            all_state = state.copy()
            if endpoint is not None:
                for j in tree.paths[endpoint]:
                    record = all_corrections[j]
                    all_state = all_state * record.decay + record.k[:, None] * record.correction[None, :]
            metrics["all_node_states"].observe(expected_state, all_state, where)
            _, next_outputs = dense_tree(arithmetic, expected_state, continuation)
            next_evaluation = leaf_evaluate(arithmetic, committed, continuation, adjust_all=adjust_all)
            metrics["continuation_outputs"].observe(next_outputs[0], next_evaluation.outputs[0], where)
            flush_work = Work()
            flushed = committed.flush(arithmetic, flush_work)
            metrics["flushed_states"].observe(expected_state, flushed.checkpoint, where)
            if endpoint is None and committed is not history:
                raise AssertionError("no accepted extension changed the authoritative history")
            counters["endpoints"] += 1
        counters["candidates"] += len(tree.nodes)
        counters["projection_vectors"] += evaluated.work.projection_vectors
        counters["baseline_projection_vectors"] += 2 * len(tree.nodes)
        counters["correction_rows"] += len(evaluated.corrections) + len(evaluated.pending_records)
        # Carry leaves, then reject all drafts, then accept an internal node.
        leaves = sorted(set(range(len(tree.nodes))) - tree.internal)
        endpoint = (None if round_index % 4 == 2 or not leaves else
                    min(tree.internal) if round_index % 4 == 3 and tree.internal else leaves[-1])
        commit_work = Work()
        history = evaluated.commit(arithmetic, endpoint, work=commit_work)
        state = state if endpoint is None else states[endpoint]
        counters["flushes"] += commit_work.flushes
        if round_index % 4 == 3:
            flush_work = Work()
            history = history.flush(arithmetic, flush_work)
            counters["terminal_pending_resolutions"] += flush_work.flush_pending_resolutions
    # Include final pending work even when the last selected endpoint was a leaf.
    work = Work()
    terminal = history.flush(arithmetic, work)
    counters["terminal_pending_resolutions"] += work.flush_pending_resolutions
    metrics["flushed_states"].observe(state, terminal.checkpoint, {"round": "terminal"})
    if precision == "exact" and any(metric.mismatches for metric in metrics.values()):
        raise AssertionError(f"exact algebra/lifecycle failure: {spec}: {metrics}")
    return {"precision": precision, "shape": shape, "regime": regime, "dk": dk, "dv": dv,
            "seed": seed, "adjust_all": adjust_all, "rounds": rounds,
            "counts": counters, "comparisons": {k: vars(v) for k, v in metrics.items()}}


def scalar_witness():
    """An exact-binary counterexample, valid even with fused multiply-add."""
    results = {}
    for precision in Precision:
        arithmetic = Arithmetic(precision)
        one = np.array([arithmetic.scalar(1)], dtype=arithmetic.dtype)
        tiny = np.array([arithmetic.scalar(1, 1 << 25)], dtype=arithmetic.dtype)
        node = Node(-1, one, one, tiny, arithmetic.scalar(1), arithmetic.scalar(1))
        state = one.reshape(1, 1)
        _, ordinary = dense_step(arithmetic, state, node)
        proposed = leaf_evaluate(arithmetic, History(state), Tree((node,))).outputs[0]
        comparison = Comparison()
        comparison.observe(ordinary, proposed, "S=k=q=decay=beta=1, v=2^-25")
        results[precision.value] = {"serial": str(ordinary[0]), "leaf": str(proposed[0]),
                                    **vars(comparison)}
    return results


def physical_workers():
    """Use available physical cores, counting hyperthreads only once on Linux."""
    cpus = os.sched_getaffinity(0) if hasattr(os, "sched_getaffinity") else range(os.cpu_count() or 1)
    cores = set()
    for cpu in cpus:
        directory = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        if directory.is_dir():
            cores.add(((directory / "physical_package_id").read_text().strip(),
                       (directory / "core_id").read_text().strip()))
        else:
            cores.add((0, cpu))
    return max(1, len(cores))


def main(argv=None):
    """Run the retained experiment; an explicit strict gate exits red on byte drift."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=physical_workers())
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261024)
    parser.add_argument("--require-byte-equivalence", action="store_true")
    parser.add_argument("--native-library", type=Path)
    args = parser.parse_args(argv)
    if args.workers < 1 or args.rounds < 1:
        parser.error("workers and rounds must be positive")
    shapes = ("empty", "singleton", "chain", "star", "ternary", "irregular", "early_leaves")
    regimes = ("ordinary", "zero", "endpoint", "tiny", "parallel")
    specs = [(precision.value, shape, regime, dk, dv, args.seed + seed_offset, adjust_all, args.rounds)
             for precision in Precision
             for dk, dv in (((3, 2),) if precision is Precision.EXACT else ((7, 5), (64, 64), (128, 128)))
             for shape in shapes for regime in regimes for seed_offset in range(2)
             for adjust_all in (False, True)]
    with ProcessPoolExecutor(max_workers=min(args.workers, len(specs))) as pool:
        print(f"Running {len(specs)} cases on {min(args.workers, len(specs))} physical workers", flush=True)
        results = []
        for result in pool.map(run_case, specs):
            results.append(result)
            if len(results) % 100 == 0:
                print(f"Completed {len(results)}/{len(specs)} cases", flush=True)
    native = None
    if args.native_library:
        from native import run_native
        native = run_native(args.native_library, args.seed)
    admitted = all(case["comparisons"][name]["mismatches"] == 0 for case in results
                   if case["precision"] == "fp32" for name in
                   ("leaf_outputs", "accepted_states", "continuation_outputs", "flushed_states"))
    report = {"schema": "gdn-leaf-elimination-research-v1", "seed": args.seed,
              "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "numpy_version": np.__version__, "workers": min(args.workers, len(specs)),
              "scope": "supplied core operands; no model, GPU, convolution, sampling, or timing certification",
              "byte_equivalence_observed": admitted, "production_qualified": False,
              "scalar_witness": scalar_witness(), "cases": results, "native_cpu": native}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Completed {len(results)} reproducible cases; FP32 byte equivalence: {admitted}")
    print(f"Report: {args.output}")
    return 1 if args.require_byte_equivalence and not admitted else 0


if __name__ == "__main__":
    sys.exit(main())
