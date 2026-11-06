/**
 * @file Test__W4A4WeightStorage.cpp
 * @brief W4A4 physical ownership, retained mapping and complete-row slice lifecycles.
 *
 * These device-free fixtures use the actual PMA/CPUStorageRoot and MmapRegion
 * authorities. They prove zero-copy adoption, one live claim through aliases,
 * source retirement after the last reader, and failure before publication.
 * They are lifecycle/contract checks, not recorded numerical replay.
 */
#include "tensors/W4A4WeightStorage.h"
#include "memory/CPUStorageRoot.h"
#include "loaders/MmapRegion.h"

#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <optional>
#include <string>
#include <unistd.h>

namespace llaminar2::w4a4
{
    namespace
    {
        constexpr int kRows = 3, kColumns = 512;
        constexpr std::size_t kBytes = kRows * (kColumns / kBlockElements) * sizeof(SourceBlock);
        const DeviceId kCPU{DeviceType::CPU, 1};
        constexpr auto kOwner = PhysicalMemoryOwner::PrimaryModelWeights;

        /** @return One bounded allocation authority, without inventory discovery or live sampling. */
        std::shared_ptr<PhysicalMemoryAuthority> authority(
            std::size_t bytes, PhysicalMemoryOwner owner = kOwner)
        {
            PhysicalMemoryPlanBuilder builder;
            builder.add(PhysicalMemoryResource{.world_rank = 0, .device = kCPU,
                .total_bytes = bytes, .admission_available_bytes = bytes}, owner, bytes);
            return std::make_shared<PhysicalMemoryAuthority>(
                std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(builder.build()), 0);
        }

        /** @return The real ledger's live byte count for this fixture's allocation owner. */
        std::size_t live(const std::shared_ptr<PhysicalMemoryAuthority> &memory,
                         PhysicalMemoryOwner owner = kOwner)
        {
            return memory->claimedBytes(kCPU, owner, PhysicalMemoryMaterializationKind::NewAllocation);
        }

        /** @return Complete native rows, with distinct row units and code patterns. */
        std::array<SourceBlock, kRows * (kColumns / kBlockElements)> source()
        {
            std::array<SourceBlock, kRows * (kColumns / kBlockElements)> result{};
            for (std::size_t block = 0; block < result.size(); ++block)
            {
                result[block].unit = (block / 2 + 1) * 0.125f;
                result[block].descriptor = 0x60004321;
                result[block].rotation_length = kColumns;
                result[block].clip_bits = 0;
                std::fill_n(result[block].codes, 128, static_cast<std::uint8_t>(block * 17));
            }
            return result;
        }

        /** @return Actual admitted native bytes, initialized before immutable publication. */
        AlignedVector<std::uint8_t> allocation(
            const std::shared_ptr<PhysicalMemoryAuthority> &memory, PhysicalMemoryOwner owner = kOwner)
        {
            auto bytes = CPUStorageRoot<std::uint8_t>(kBytes, owner, kCPU,
                CPUStoragePlacement::local(), CPUStorageInitialization::Uninitialized).allocate(memory);
            const auto blocks = source();
            std::memcpy(bytes.data(), blocks.data(), kBytes);
            return bytes;
        }

        /** @brief Fresh bounded file owned and removed by this fixture alone. */
        class SourceFile final
        {
        public:
            static constexpr std::size_t offset = 16;
            /** @brief Write an exact native interval with unrelated guarded bytes around it. */
            SourceFile()
            {
                std::string name = ::testing::TempDir() + "/w4a4-storage-XXXXXX";
                const int descriptor = ::mkstemp(name.data());
                if (descriptor < 0) throw std::runtime_error("Cannot create native storage fixture");
                std::array<std::uint8_t, offset + kBytes + 32> contents;
                contents.fill(0x5a);
                const auto blocks = source();
                std::memcpy(contents.data() + offset, blocks.data(), kBytes);
                const auto written = ::write(descriptor, contents.data(), contents.size());
                const int closed = ::close(descriptor);
                if (written != static_cast<ssize_t>(contents.size()) || closed != 0)
                {
                    ::unlink(name.c_str());
                    throw std::runtime_error("Cannot publish complete native storage fixture");
                }
                path_ = std::move(name);
            }
            /** @brief Retire only this fixture's exclusively created file. */
            ~SourceFile() { ::unlink(path_.c_str()); }
            SourceFile(const SourceFile &) = delete;
            SourceFile &operator=(const SourceFile &) = delete;
            /** @return The exact path consumed by the real mapping authority. */
            const std::string &path() const noexcept { return path_; }
        private:
            std::string path_; ///< Exclusive file name; no process-wide directory cleanup.
        };
    }

    TEST(W4A4WeightStorageLifecycle, RowLeasesRetainOneExactPhysicalOwner)
    {
        const auto physical = CPUStorageRoot<std::uint8_t>(kBytes, kOwner, kCPU).physicalBytes();
        auto memory = authority(physical);
        auto bytes = allocation(memory);
        const auto *address = bytes.data();
        std::optional<WeightStorage> root(WeightStorage::adopt(
            Transform::Identity, kRows, kColumns, std::move(bytes), kOwner));
        EXPECT_TRUE(bytes.empty());
        EXPECT_EQ(root->allocator(), kCPU);
        EXPECT_FALSE(root->isMapped());
        EXPECT_EQ(reinterpret_cast<const std::uint8_t *>(root->view().blocks().data()), address);
        EXPECT_EQ(live(memory), physical);
        std::optional<WeightStorage> tail(root->rowSlice(1, 2));
        std::optional<WeightStorage> last(tail->rowSlice(1, 1));
        EXPECT_EQ(last->view().blocks().data(), root->view().blocks().data() + 4);
        EXPECT_EQ(last->view().contract().sourceBytes(), 2 * sizeof(SourceBlock));
        EXPECT_EQ(last->view().contract().rotationLength(), kColumns);
        EXPECT_EQ(last->view().blocks().front().unit, 0.375f);
        EXPECT_EQ(live(memory), physical);
        root.reset();
        tail.reset();
        EXPECT_EQ(live(memory), physical);
        EXPECT_EQ(last->view().blocks().front().codes[0], 68);
        last.reset();
        EXPECT_EQ(live(memory), 0u);
    }

    TEST(W4A4WeightStorageLifecycle, CopyAndRvalueCopyCannotInvalidateExistingReaders)
    {
        const auto physical = CPUStorageRoot<std::uint8_t>(kBytes, kOwner, kCPU).physicalBytes();
        auto memory = authority(physical);
        {
            auto root = WeightStorage::adopt(Transform::Identity, kRows, kColumns, allocation(memory), kOwner);
            const auto copied = root;
            const auto retained = std::move(root);
            EXPECT_EQ(root.view().blocks().data(), copied.view().blocks().data());
            EXPECT_EQ(root.view().blocks().data(), retained.view().blocks().data());
            EXPECT_EQ(live(memory), physical);
            EXPECT_THROW(static_cast<void>(root.rowSlice(3, 1)), std::out_of_range);
            EXPECT_THROW(static_cast<void>(root.rowSlice(0, 0)), std::out_of_range);
            EXPECT_THROW(static_cast<void>(root.rowSlice(1, 3)), std::out_of_range);
            EXPECT_THROW(static_cast<void>(root.rowSlice(std::numeric_limits<std::size_t>::max(), 1)), std::out_of_range);
            EXPECT_EQ(live(memory), physical);
        }
        EXPECT_EQ(live(memory), 0u);
    }

    TEST(W4A4WeightStorageContract, MissingOrMismatchedAdmissionCannotPublishStorage)
    {
        const auto physical = CPUStorageRoot<std::uint8_t>(kBytes, kOwner, kCPU).physicalBytes();
        auto memory = authority(physical);
        EXPECT_THROW(static_cast<void>(WeightStorage::adopt(Transform::Identity, kRows, kColumns,
            AlignedVector<std::uint8_t>(kBytes), kOwner)), std::logic_error);
        EXPECT_EQ(live(memory), 0u);
        constexpr auto workspace = PhysicalMemoryOwner::ExecutionWorkspace;
        auto wrong = authority(physical, workspace);
        EXPECT_THROW(static_cast<void>(WeightStorage::adopt(Transform::Identity, kRows, kColumns,
            allocation(wrong, workspace), kOwner)), std::logic_error);
        EXPECT_EQ(live(wrong, workspace), 0u);
        EXPECT_THROW(static_cast<void>(WeightStorage::adopt(Transform::Identity, kRows, kColumns,
            allocation(wrong, workspace), workspace)), std::invalid_argument);
        EXPECT_EQ(live(wrong, workspace), 0u);
        auto malformed = allocation(memory);
        // Exact block position is preserved in the public admission diagnostic.
        SourceBlock damaged = source()[3];
        damaged.unit = -1.0f;
        std::memcpy(malformed.data() + 3 * sizeof(SourceBlock), &damaged, sizeof(damaged));
        try
        {
            static_cast<void>(WeightStorage::adopt(Transform::Identity, kRows, kColumns,
                                                   std::move(malformed), kOwner));
            FAIL() << "Malformed source unexpectedly published";
        }
        catch (const std::invalid_argument &error)
        {
            EXPECT_NE(std::string(error.what()).find("row 1, block 1"), std::string::npos);
        }
        EXPECT_EQ(live(memory), 0u);
    }

    TEST(W4A4WeightStorageLifecycle, MappedRowsRetainExactFileCoordinatesAndFinalReaderLifetime)
    {
        const SourceFile file;
        std::shared_ptr<MmapRegion> region = MmapRegion::create(
            file.path(), -1, true, MmapRegion::PrefaultPolicy::DemandPaged);
        ASSERT_NE(region, nullptr);
        std::weak_ptr<MmapRegion> retired = region;
        const auto *address = region->data() + SourceFile::offset;
        std::optional<WeightStorage> root(WeightStorage::map(
            Transform::Identity, kRows, kColumns, region, SourceFile::offset));
        EXPECT_TRUE(root->isMapped());
        EXPECT_EQ(root->allocator(), DeviceId::cpu());
        EXPECT_EQ(reinterpret_cast<const std::uint8_t *>(root->view().blocks().data()), address);
        std::optional<WeightStorage> last(root->rowSlice(2, 1));
        const auto provenance = MmapRegion::resolveFileSource(last->view().blocks().data(), 2 * sizeof(SourceBlock));
        ASSERT_TRUE(provenance.has_value());
        EXPECT_EQ(provenance->offset, SourceFile::offset + 4 * sizeof(SourceBlock));
        region.reset();
        root.reset();
        EXPECT_FALSE(retired.expired());
        EXPECT_EQ(last->view().blocks().front().unit, 0.375f);
        last.reset();
        EXPECT_TRUE(retired.expired());
        EXPECT_FALSE(MmapRegion::resolveFileSource(address, kBytes).has_value());
    }

    TEST(W4A4WeightStorageContract, MappingsRejectAbsentMisalignedAndIncompleteNativeIntervals)
    {
        const SourceFile file;
        std::shared_ptr<MmapRegion> region = MmapRegion::create(
            file.path(), -1, true, MmapRegion::PrefaultPolicy::DemandPaged);
        ASSERT_NE(region, nullptr);
        EXPECT_THROW(static_cast<void>(WeightStorage::map(Transform::Identity, kRows, kColumns, nullptr, 0)), std::invalid_argument);
        EXPECT_THROW(static_cast<void>(WeightStorage::map(Transform::Identity, kRows, kColumns, region, 1)), std::invalid_argument);
        EXPECT_THROW(static_cast<void>(WeightStorage::map(Transform::Identity, kRows, kColumns, region, region->size())), std::invalid_argument);
        EXPECT_THROW(static_cast<void>(WeightStorage::map(Transform::Identity, kRows, kColumns, region,
            std::numeric_limits<std::size_t>::max())), std::invalid_argument);
        EXPECT_THROW(static_cast<void>(WeightStorage::map(Transform::Identity, kRows + 1, kColumns, region, SourceFile::offset)), std::invalid_argument);
    }
} // namespace llaminar2::w4a4
