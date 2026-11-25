/**
 * @file Test__NonNUMAKernelClassification.cpp
 * @brief Regression: node-0 binding is skipped only on a kernel without NUMA.
 *
 * Kernels built without NUMA (e.g. Docker Desktop's VM) expose no sysfs node
 * directory and reject memory-policy syscalls; node 0 is then the whole
 * machine. A container that only hides sysfs on a NUMA kernel must keep strict
 * binding, so classification requires every independent observation to agree.
 */

#include "utils/NUMATopology.h"

#include <gtest/gtest.h>

namespace llaminar2::test
{
    TEST(NonNUMAKernelClassification, RequiresEveryObservationToAgree)
    {
        EXPECT_TRUE(classifyNonNUMAKernel({true, true, 1}));
        EXPECT_TRUE(classifyNonNUMAKernel({true, true, 0}));
        // Container hiding sysfs on a NUMA kernel: syscalls still present.
        EXPECT_FALSE(classifyNonNUMAKernel({true, false, 1}));
        EXPECT_FALSE(classifyNonNUMAKernel({true, false, 2}));
        // Syscalls filtered (seccomp) while sysfs shows nodes.
        EXPECT_FALSE(classifyNonNUMAKernel({false, true, 1}));
        // hwloc sees several nodes or could not observe at all.
        EXPECT_FALSE(classifyNonNUMAKernel({true, true, 2}));
        EXPECT_FALSE(classifyNonNUMAKernel({true, true, -1}));
        EXPECT_FALSE(classifyNonNUMAKernel({false, false, 4}));
    }

    TEST(NonNUMAKernelClassification, OnlyNodeZeroAndOneCachedObservation)
    {
        const auto &first = observedNonNUMAKernelEvidence();
        const auto &second = observedNonNUMAKernelEvidence();
        EXPECT_EQ(&first, &second) << "the observation is taken once per process";
        EXPECT_FALSE(nonNUMAKernelSingleNode(1));
        EXPECT_FALSE(nonNUMAKernelSingleNode(-1));
        EXPECT_EQ(nonNUMAKernelSingleNode(0), classifyNonNUMAKernel(first));
    }
} // namespace llaminar2::test
