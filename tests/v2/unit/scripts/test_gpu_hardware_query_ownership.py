#!/usr/bin/env python3
"""Keep physical GPU property discovery at its immutable startup authority.

This architecture check complements the compiled inventory/launch replays and
native query observers. Function occupancy, active-context queries and runtime
health checks have different lifetimes and are deliberately not card metadata.
The one native GDN observation adapter authenticates injected records in tests;
its exact caller allowance cannot authorize production launch-path discovery.
"""
from collections import Counter
from pathlib import Path
import re
import unittest

from check_gpu_blocking_sync_policy import (
    enclosing_caller, function_intervals, source_files, strip_comments_and_literals,
)

ROOT = Path(__file__).resolve().parents[4]
PHYSICAL = re.compile(
    r"\b(?:cudaGetDeviceProperties|hipGetDeviceProperties|cudaDeviceGetAttribute|hipDeviceGetAttribute|"
    r"cuDeviceGetAttribute|cuDeviceGetProperties|"
    r"cudaDevAttr(?:MultiProcessorCount|WarpSize|ComputeCapabilityMajor|ComputeCapabilityMinor|"
    r"MaxThreadsPerBlock|MaxBlocksPerMultiprocessor|MaxSharedMemoryPerBlockOptin)|"
    r"hipDeviceAttribute(?:MultiprocessorCount|WarpSize|MaxThreadsPerBlock|"
    r"MaxSharedMemoryPerBlock|WallClockRate|CanUseStreamWaitValue))\b"
)
# Exact call/token budgets, not whole-file exemptions. New property consumers
# must extend the authority's record and use its typed binding instead.
ALLOWED = Counter({
    ("src/v2/backends/cuda/CUDAHardwareMetadata.cu", "readHardware", "cudaGetDeviceProperties"): 1,
    ("src/v2/backends/rocm/ROCmHardwareMetadata.cpp", "readHardware", "hipGetDeviceProperties"): 1,
    ("src/v2/backends/rocm/ROCmHardwareMetadata.cpp", "readHardware", "hipDeviceGetAttribute"): 2,
    ("src/v2/backends/rocm/ROCmHardwareMetadata.cpp", "readHardware", "hipDeviceAttributeWallClockRate"): 1,
    ("src/v2/backends/rocm/ROCmHardwareMetadata.cpp", "readHardware", "hipDeviceAttributeCanUseStreamWaitValue"): 1,
    ("src/v2/kernels/cuda/gdn/CUDADenseCompactSnapshots.cuh",
     "prepareObservedCUDADenseRetentionCapturePlan", "cudaGetDeviceProperties"): 1,
})


def observations(path, text):
    """Attribute native facts to exact callers after stripping comments/literals."""
    if not PHYSICAL.search(text):
        return Counter()
    source = strip_comments_and_literals(text)
    functions = function_intervals(source)
    return Counter((path, enclosing_caller(functions, match.start()), match.group())
                   for match in PHYSICAL.finditer(source))


def failures(actual, allowed=ALLOWED):
    """Reject both unreviewed queries and stale allowances for retired queries."""
    return {key: (actual[key], allowed[key]) for key in actual.keys() | allowed.keys()
            if actual[key] != allowed[key]}


class GPUHardwareQueryOwnership(unittest.TestCase):
    """Prevent independent physical caches while retaining native program probes."""

    def test_production_has_exact_query_owners(self):
        actual = Counter()
        for path in source_files(ROOT):
            actual.update(observations(path.relative_to(ROOT).as_posix(), path.read_text()))
        self.assertEqual(failures(actual), {})

    def test_new_kernel_and_backend_queries_are_rejected(self):
        for path in ("src/v2/kernels/cuda/example.cu", "src/v2/backends/cuda/CUDABackend.cu"):
            for expression in ("cudaGetDeviceProperties(&p, device)",
                               "cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, device)",
                               "hipDeviceGetAttribute(&n, hipDeviceAttributeWallClockRate, device)",
                               "cuDeviceGetAttribute(&n, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device)",
                               "cuDeviceGetProperties(&properties, device)"):
                with self.subTest(path=path, expression=expression):
                    actual = observations(path, "void launch() { " + expression + "; }")
                    self.assertTrue(failures(actual, Counter()))

    def test_allowed_file_does_not_allow_another_caller_or_extra_query(self):
        path, caller, token = next(iter(ALLOWED))
        original = observations(path, f"void {caller}() {{ {token}(&p, i); }}")
        allowed = Counter({(path, caller, token): 1})
        self.assertEqual(failures(original, allowed), {})
        for text in (f"void another() {{ {token}(&p, i); }}",
                     f"void {caller}() {{ {token}(&p, i); {token}(&p, i); }}"):
            self.assertTrue(failures(observations(path, text), allowed))
        self.assertTrue(failures(Counter(), allowed))

    def test_new_attribute_cannot_bypass_generic_query_guard(self):
        for function, attribute in (("cudaDeviceGetAttribute", "cudaDevAttrFutureUnlistedFeature"),
                                    ("hipDeviceGetAttribute", "hipDeviceAttributeFutureUnlistedFeature"),
                                    ("cuDeviceGetAttribute", "CU_DEVICE_ATTRIBUTE_FUTURE_UNLISTED_FEATURE")):
            with self.subTest(function=function):
                actual = observations("src/v2/kernels/new_kernel.cu",
                                      f"void prepare() {{ {function}(&value, {attribute}, ordinal); }}")
                self.assertEqual(sum(actual.values()), 1)
                self.assertTrue(failures(actual, Counter()))

    def test_driver_queries_cannot_borrow_runtime_reader_allowance(self):
        path = "src/v2/backends/cuda/CUDAHardwareMetadata.cu"
        original = "void readHardware() { cudaGetDeviceProperties(&properties, ordinal); }"
        allowed = Counter({(path, "readHardware", "cudaGetDeviceProperties"): 1})
        self.assertEqual(failures(observations(path, original), allowed), {})
        for query in ("cuDeviceGetAttribute(&value, CU_DEVICE_ATTRIBUTE_FUTURE_UNLISTED_FEATURE, ordinal);",
                      "cuDeviceGetProperties(&properties, ordinal);"):
            with self.subTest(query=query):
                extended = original.rsplit("}", 1)[0] + query + "}"
                self.assertTrue(failures(observations(path, extended), allowed))

    def test_native_hip_reader_has_exactly_two_attribute_queries(self):
        path = "src/v2/backends/rocm/ROCmHardwareMetadata.cpp"
        source = """void readHardware() {
            hipDeviceGetAttribute(&clock, hipDeviceAttributeWallClockRate, ordinal);
            const auto capability = observeROCmStreamWaitValueCapability(ordinal,
                [](int *value, int device) {
                    return hipDeviceGetAttribute(value, hipDeviceAttributeCanUseStreamWaitValue, device);
                });
        }"""
        allowed = Counter({(path, "readHardware", "hipDeviceGetAttribute"): 2,
                           (path, "readHardware", "hipDeviceAttributeWallClockRate"): 1,
                           (path, "readHardware", "hipDeviceAttributeCanUseStreamWaitValue"): 1})
        self.assertEqual(failures(observations(path, source), allowed), {})
        third_query = "hipDeviceGetAttribute(&value, hipDeviceAttributeFutureUnlistedFeature, ordinal);"
        extended = source.rsplit("}", 1)[0] + third_query + "}"
        self.assertTrue(failures(observations(path, extended), allowed))
        self.assertTrue(failures(observations(path, source.replace("readHardware", "anotherCaller")), allowed))

    def test_comments_and_program_resource_observations_are_not_physical_queries(self):
        self.assertEqual(observations("src/v2/kernels/example.cu", '''
            // cudaGetDeviceProperties(&properties, ordinal);
            // cuDeviceGetAttribute(&value, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, ordinal);
            const char *diagnostic = "hipDeviceAttributeWallClockRate cuDeviceGetProperties";
            void prepare() {
                cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, kernel, 256, 0);
                hipFuncGetAttributes(&attributes, kernel);
                cuFuncGetAttribute(&registers, CU_FUNC_ATTRIBUTE_NUM_REGS, function);
                cuOccupancyMaxActiveBlocksPerMultiprocessor(&n, function, 256, 0);
                cudaGetDevice(&ordinal);
                hipGetDevice(&ordinal);
                cuCtxGetDevice(&ordinal);
            }
        '''), Counter())


if __name__ == "__main__":
    unittest.main()
