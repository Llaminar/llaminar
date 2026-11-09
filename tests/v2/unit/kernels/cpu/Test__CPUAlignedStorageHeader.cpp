/**
 * @file Test__CPUAlignedStorageHeader.cpp
 * @brief Standalone aligned-storage ownership-header compilation without transitive prerequisites.
 *
 * Under the production C++23 standard, the real public header must declare
 * its fatal deallocation boundary itself;
 * unrelated tensor, MPI or NEON includes must not supply std::terminate by
 * accident. This interface test allocates no storage, initializes no backend
 * and does not substitute for PhysicalMemoryAuthority allocation/lifetime gates.
 */
#include "tensors/AlignedVector.h"
#include <gtest/gtest.h>
#include <type_traits>

static_assert(std::is_default_constructible_v<llaminar2::AlignedVector<float>>);
static_assert(std::is_destructible_v<llaminar2::AlignedVector<float>>);
static_assert(std::is_nothrow_move_constructible_v<llaminar2::AlignedVector<float>>);

/** @test Complete owner declarations parse independently of architecture-specific include order. */
TEST(CPUAlignedStorageHeader, OwnsItsFatalDeallocationDeclaration)
{
    SUCCEED(); // Missing ownership declarations fail translation before this test can run.
}
