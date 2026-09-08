/**
 * @file Test__AlignedVector.cpp
 * @brief Regression coverage for the cache/page alignment guarantees provided
 *        by the AlignedVector tensor utility.
 *
 * MoE CPU expert preparation uses strict NUMA page-placement checks for the
 * large NativeVNNI interleaved weight buffers stored in AlignedVector.  Those
 * checks operate at page granularity, so large buffers must begin on a page
 * boundary; otherwise the migration range can accidentally include neighboring
 * heap allocations on the first or last page and fail for reasons unrelated to
 * the expert weight buffer being prepared.
 */

#include "tensors/AlignedVector.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <limits>
#include <fstream>
#include <sstream>
#include <string>
#ifdef __linux__
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace
{
    /**
     * @brief Return the operating-system page size for alignment assertions.
     *
     * A non-positive return value means the platform did not expose a usable
     * runtime page size, in which case Linux-specific page-alignment assertions
     * should be skipped rather than guessed.
     */
    long runtimePageSize()
    {
#ifdef __linux__
        return sysconf(_SC_PAGESIZE);
#else
        return -1;
#endif
    }

    /**
     * @brief Prove pre-allocation heap geometry against the materialized owner.
     * @tparam T Element type whose byte multiplication is part of the contract.
     * @param elements Logical element capacity.
     * @param alignment Requested minimum address alignment.
     */
    template <typename T>
    void expectHeapGeometryMatchesAllocation(
        size_t elements,
        size_t alignment)
    {
        using Vector = llaminar2::AlignedVector<T>;
        const size_t planned = Vector::requiredAllocationBytes(
            elements, alignment, Vector::StorageKind::Heap);
        Vector materialized;
        materialized.resize_uninitialized_aligned(elements, alignment);
        EXPECT_EQ(materialized.allocationBytes(), planned);
        EXPECT_TRUE(materialized.is_aligned_to(
            std::max(alignment, Vector::ALIGNMENT)));
    }
} // namespace

TEST(Test__AlignedVector, SmallAllocationsRemainCacheLineAligned)
{
    llaminar2::AlignedVector<uint8_t> buffer;
    buffer.resize_uninitialized(257);

    ASSERT_NE(buffer.data(), nullptr);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(buffer.data()) %
                  llaminar2::AlignedVector<uint8_t>::ALIGNMENT,
              0u);
}

TEST(Test__AlignedVector, LargeAllocationsArePageAlignedForStrictNUMAPlacement)
{
    const long page_size = runtimePageSize();
    if (page_size <= 0)
    {
        GTEST_SKIP() << "runtime page size is unavailable on this platform";
    }

    llaminar2::AlignedVector<uint8_t> buffer;
    buffer.resize_uninitialized(static_cast<size_t>(page_size) * 3u + 137u);

    ASSERT_NE(buffer.data(), nullptr);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(buffer.data()) %
                  static_cast<uintptr_t>(page_size),
              0u);
}

TEST(Test__AlignedVector, DedicatedPageMappingMovesWithoutChangingItsOwner)
{
#ifndef __linux__
    GTEST_SKIP() << "anonymous page mappings require Linux";
#else
    const long page_size = runtimePageSize();
    ASSERT_GT(page_size, 0);

    using Vector = llaminar2::AlignedVector<uint16_t>;
    Vector source = Vector::pageMappedUninitialized(
        static_cast<size_t>(page_size) + 37u);
    ASSERT_NE(source.data(), nullptr);
    EXPECT_EQ(source.storageKind(), Vector::StorageKind::AnonymousPageMapping);
    EXPECT_EQ(
        reinterpret_cast<uintptr_t>(source.data()) %
            static_cast<uintptr_t>(page_size),
        0u);
    EXPECT_EQ(
        source.allocationBytes() % static_cast<size_t>(page_size),
        0u);

    auto *const stable_address = source.data();
    Vector destination(std::move(source));
    EXPECT_EQ(destination.data(), stable_address);
    EXPECT_EQ(
        destination.storageKind(),
        Vector::StorageKind::AnonymousPageMapping);
    EXPECT_EQ(source.data(), nullptr);
    EXPECT_EQ(source.storageKind(), Vector::StorageKind::Heap);

    // The moved owner must retain ordinary read/write vector semantics.
    destination.front() = 0x1234u;
    destination.back() = 0x5678u;
    EXPECT_EQ(destination.front(), 0x1234u);
    EXPECT_EQ(destination.back(), 0x5678u);
#endif
}

TEST(Test__AlignedVector, DedicatedPageMappingIsUnmappedAtOwnerRetirement)
{
#ifndef __linux__
    GTEST_SKIP() << "anonymous page mappings require Linux";
#else
    const long page_size = runtimePageSize();
    ASSERT_GT(page_size, 0);

    void *retired_address = nullptr;
    {
        auto mapping =
            llaminar2::AlignedVector<uint8_t>::pageMappedUninitialized(
                static_cast<size_t>(page_size));
        retired_address = mapping.data();
        mapping.front() = 0xa5u;
    }

    unsigned char residency = 0;
    errno = 0;
    EXPECT_EQ(::mincore(
                  retired_address,
                  static_cast<size_t>(page_size),
                  &residency),
              -1);
    EXPECT_EQ(errno, ENOMEM)
        << "mapped owner retirement must revoke the virtual range completely";
    // Retirement includes both virtual guards, not only writable payload.
    for (const auto address : {
             reinterpret_cast<uintptr_t>(retired_address) - static_cast<size_t>(page_size),
             reinterpret_cast<uintptr_t>(retired_address) + static_cast<size_t>(page_size)})
    {
        errno = 0;
        EXPECT_EQ(::mincore(reinterpret_cast<void *>(address),
                            static_cast<size_t>(page_size), &residency), -1);
        EXPECT_EQ(errno, ENOMEM);
    }
#endif
}

TEST(Test__AlignedVector, DedicatedMappingGuardsPreventHugePageOwnershipSharing)
{
#ifndef __linux__
    GTEST_SKIP() << "anonymous page mappings require Linux";
#else
    using Vector = llaminar2::AlignedVector<uint8_t>;
    constexpr size_t huge_page = 2 * 1024 * 1024;
    const size_t page = static_cast<size_t>(runtimePageSize());
    auto storage = Vector::pageMappedUninitialized(huge_page + page);
    const auto address = reinterpret_cast<uintptr_t>(storage.data());
    EXPECT_EQ(address % huge_page, 0u);
    // Virtual guards must not inflate the canonical physical capacity BOM.
    EXPECT_EQ(storage.allocationBytes(), huge_page + page);
    std::ifstream maps("/proc/self/maps");
    ASSERT_TRUE(maps.good());
    bool before = false, payload = false, after = false;
    for (std::string line; std::getline(maps, line);)
    {
        std::istringstream fields(line);
        uintptr_t start = 0, end = 0;
        char separator = 0;
        std::string permission;
        fields >> std::hex >> start >> separator >> end >> permission;
        if (start <= address - page && end >= address)
            before = permission == "---p";
        if (start == address && end == address + storage.allocationBytes())
            payload = permission == "rw-p";
        if (start <= address + storage.allocationBytes() &&
            end >= address + storage.allocationBytes() + page)
            after = permission == "---p";
    }
    EXPECT_TRUE(before);
    EXPECT_TRUE(payload);
    EXPECT_TRUE(after);
#endif
}

TEST(Test__AlignedVector, HeapAdmissionGeometryMatchesEveryExpertElementWidth)
{
    constexpr size_t requested_alignment = 4096u;
    expectHeapGeometryMatchesAllocation<uint8_t>(37u, requested_alignment);
    expectHeapGeometryMatchesAllocation<uint16_t>(2051u, requested_alignment);
    expectHeapGeometryMatchesAllocation<float>(4099u, requested_alignment);
}

TEST(Test__AlignedVector, PageMappingAdmissionGeometryMatchesMaterialization)
{
#ifndef __linux__
    GTEST_SKIP() << "anonymous page mappings require Linux";
#else
    const long page_size = runtimePageSize();
    ASSERT_GT(page_size, 0);

    using ByteVector = llaminar2::AlignedVector<uint8_t>;
    using HalfVector = llaminar2::AlignedVector<uint16_t>;
    using FloatVector = llaminar2::AlignedVector<float>;
    const size_t page = static_cast<size_t>(page_size);

    const auto byte = ByteVector::pageMappedUninitialized(page + 3u);
    EXPECT_EQ(
        byte.allocationBytes(),
        ByteVector::requiredAllocationBytes(
            page + 3u,
            page,
            ByteVector::StorageKind::AnonymousPageMapping));

    const auto half = HalfVector::pageMappedUninitialized(page / 2u + 3u);
    EXPECT_EQ(
        half.allocationBytes(),
        HalfVector::requiredAllocationBytes(
            page / 2u + 3u,
            page,
            HalfVector::StorageKind::AnonymousPageMapping));

    const auto fp32 = FloatVector::pageMappedUninitialized(page / 4u + 3u);
    EXPECT_EQ(
        fp32.allocationBytes(),
        FloatVector::requiredAllocationBytes(
            page / 4u + 3u,
            page,
            FloatVector::StorageKind::AnonymousPageMapping));
#endif
}

TEST(Test__AlignedVector, AdmissionGeometryRejectsInvalidOrOverflowingRequests)
{
    using ByteVector = llaminar2::AlignedVector<uint8_t>;
    using HalfVector = llaminar2::AlignedVector<uint16_t>;

    EXPECT_THROW(
        (void)ByteVector::requiredAllocationBytes(1u, 96u),
        std::invalid_argument);
    EXPECT_THROW(
        (void)HalfVector::requiredAllocationBytes(
            std::numeric_limits<size_t>::max() / sizeof(uint16_t) + 1u),
        std::overflow_error);
    EXPECT_THROW(
        (void)ByteVector::requiredAllocationBytes(
            std::numeric_limits<size_t>::max()),
        std::overflow_error);
}
