#include <gtest/gtest.h>

#include "loaders/MmapRegion.h"

#include <filesystem>
#include <fstream>
#include <string>

#include <unistd.h>

namespace llaminar2::test
{
    namespace
    {
        /**
         * @brief Creates a small temporary file that is safe to mmap in unit tests.
         *
         * The contents are irrelevant; the test is about the mmap prefault policy
         * chosen by MmapRegion::create(), not filesystem throughput.
         */
        class TemporaryMmapFile
        {
        public:
            TemporaryMmapFile()
            {
                path_ = std::filesystem::temp_directory_path() /
                        ("llaminar_mmap_region_test_" + std::to_string(::getpid()) + ".bin");
                std::ofstream out(path_, std::ios::binary | std::ios::trunc);
                std::string page(4096, '\x5a');
                out.write(page.data(), static_cast<std::streamsize>(page.size()));
            }

            ~TemporaryMmapFile()
            {
                std::error_code ignored;
                std::filesystem::remove(path_, ignored);
            }

            const std::filesystem::path &path() const { return path_; }

        private:
            std::filesystem::path path_;
        };
    } // namespace

    TEST(Test__MmapRegion, DemandPagedPolicyDoesNotEagerPrefault)
    {
        TemporaryMmapFile file;

        auto region = MmapRegion::create(
            file.path().string(),
            /*numa_node=*/-1,
            /*skip_cache_eviction=*/false,
            MmapRegion::PrefaultPolicy::DemandPaged);

        ASSERT_NE(region, nullptr);
        EXPECT_FALSE(region->wasEagerPrefaulted())
            << "GPU-target mmap must not use MAP_POPULATE or NUMA first-touch";
    }

    TEST(Test__MmapRegion, AutoPolicyPreservesHistoricalEagerPrefault)
    {
        TemporaryMmapFile file;

        auto region = MmapRegion::create(
            file.path().string(),
            /*numa_node=*/-1,
            /*skip_cache_eviction=*/false,
            MmapRegion::PrefaultPolicy::Auto);

        ASSERT_NE(region, nullptr);
        EXPECT_TRUE(region->wasEagerPrefaulted())
            << "CPU/non-GPU mmap keeps the historical eager-prefault behavior";
    }
} // namespace llaminar2::test
