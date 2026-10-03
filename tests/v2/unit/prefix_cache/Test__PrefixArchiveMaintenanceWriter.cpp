/**
 * @file Test__PrefixArchiveMaintenanceWriter.cpp
 * @brief Native, device-free proof of bounded durable background archive writes.
 *
 * The tests use real descriptors and kernel flags, not a successful synthetic
 * range-flush sink. This closes the OverlayFS regression where empty logical
 * mappings made every sync_file_range return successfully without doing I/O.
 * Full publication/concurrent-tail coverage lives beside this focused proof
 * in ProductionTestPreflight's archive backend entry.
 */
#include "execution/prefix_cache/PrefixArchiveMaintenanceWriter.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <filesystem>
#include <span>
#include <sys/stat.h>
#include <vector>

using namespace llaminar2;

namespace
{
    /** @brief Exact temporary inode owner; the production writer only borrows it. */
    class NativeTestFile final
    {
    public:
        /** @brief Create one private path with exactly the requested write mode. */
        explicit NativeTestFile(int mode = O_RDWR | O_DSYNC)
        {
            std::string pattern = (std::filesystem::temp_directory_path() /
                "llaminar-maintenance-write.XXXXXXXX").string();
            if (!::mkdtemp(pattern.data()))
                throw std::runtime_error("failed to create test directory");
            directory_ = pattern;
            path_ = directory_ + "/archive";
            fd_ = ::open(path_.c_str(), mode | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
            if (fd_ < 0)
            {
                ::rmdir(directory_.c_str());
                throw std::runtime_error("failed to create test archive");
            }
        }

        /** @brief Close and remove only this test's exact file and empty directory. */
        ~NativeTestFile()
        {
            close();
            ::unlink(path_.c_str());
            ::rmdir(directory_.c_str());
        }
        NativeTestFile(const NativeTestFile &) = delete;
        NativeTestFile &operator=(const NativeTestFile &) = delete;

        /** @return Borrowed native descriptor for policy and byte-level checks. */
        int fd() const noexcept { return fd_; }

        /** @brief Retire the owned descriptor for the native-failure regression. */
        void close() noexcept
        {
            if (fd_ >= 0)
            {
                ::close(fd_);
                fd_ = -1;
            }
        }
    private:
        std::string directory_;
        std::string path_;
        int fd_ = -1;
    };
}

/** @test Actual kernel flags, not advertised support, certify native completion. */
TEST(Test__PrefixArchiveMaintenanceWriter, RequiresWritableNativeDataSynchronousMode)
{
    NativeTestFile durable;
    PrefixArchiveMaintenanceWriter writer(durable.fd());
    EXPECT_EQ(writer.evidence().native_open_flags, ::fcntl(durable.fd(), F_GETFL));
    EXPECT_EQ(writer.evidence().native_open_flags & O_DSYNC, O_DSYNC);
    EXPECT_EQ(writer.evidence().durable_writes, 0u);
    NativeTestFile buffered(O_RDWR);
    EXPECT_THROW(PrefixArchiveMaintenanceWriter{buffered.fd()}, std::runtime_error);
    NativeTestFile read_only(O_RDONLY | O_DSYNC);
    EXPECT_THROW(PrefixArchiveMaintenanceWriter{read_only.fd()}, std::runtime_error);
    EXPECT_THROW(PrefixArchiveMaintenanceWriter{-1}, std::runtime_error);
}

/** @test A partial live extent never writes adjacent sentinel or unused capacity. */
TEST(Test__PrefixArchiveMaintenanceWriter, WritesOnlyExactLiveBytesWithNativeEvidence)
{
    NativeTestFile file;
    PrefixArchiveMaintenanceWriter writer(file.fd());
    const std::array<std::byte, 7> input{
        std::byte{1}, std::byte{2}, std::byte{3}, std::byte{4},
        std::byte{5}, std::byte{6}, std::byte{99}};
    ASSERT_TRUE(writer.write(std::span(input).first(3)));
    ASSERT_TRUE(writer.write(std::span(input).subspan(3, 3)));
    std::array<std::byte, 7> actual{};
    EXPECT_EQ(::pread(file.fd(), actual.data(), actual.size(), 0), 6);
    EXPECT_TRUE(std::equal(actual.begin(), actual.begin() + 6,
                           input.begin(), input.begin() + 6));
    EXPECT_EQ(writer.evidence().durable_writes, 2u);
    EXPECT_EQ(writer.evidence().durable_bytes, 6u);
    EXPECT_EQ(writer.evidence().largest_write_bytes, 3u);
}

/** @test The largest supported chunk completes without accumulating a dirty tail. */
TEST(Test__PrefixArchiveMaintenanceWriter, AcceptsCanonicalMaximumChunk)
{
    NativeTestFile file;
    PrefixArchiveMaintenanceWriter writer(file.fd());
    std::vector<std::byte> chunk(PrefixArchiveIOGeometry::compactionBytes(), std::byte{0x53});
    ASSERT_TRUE(writer.write(chunk));
    EXPECT_EQ(writer.evidence().durable_bytes, chunk.size());
    EXPECT_LE(writer.evidence().largest_write_bytes, chunk.size());
    std::byte last{};
    EXPECT_EQ(::pread(file.fd(), &last, 1, chunk.size() - 1), 1);
    EXPECT_EQ(last, std::byte{0x53});
}

/** @test Invalid extents fail before native I/O and cannot be retried into success. */
TEST(Test__PrefixArchiveMaintenanceWriter, GeometryFailureIsTerminalWithoutNativeWrites)
{
    const std::array<std::byte, 1> valid{std::byte{1}};
    std::vector<std::byte> oversized(PrefixArchiveIOGeometry::compactionBytes() + 1);
    for (const auto bytes : {std::span<const std::byte>{}, std::span<const std::byte>(oversized)})
    {
        NativeTestFile file;
        PrefixArchiveMaintenanceWriter writer(file.fd());
        std::string error;
        EXPECT_FALSE(writer.write(bytes, &error));
        EXPECT_FALSE(error.empty());
        EXPECT_FALSE(writer.write(valid));
        EXPECT_EQ(writer.evidence().durable_writes, 0u);
        EXPECT_EQ(::lseek(file.fd(), 0, SEEK_END), 0);
    }
}

/** @test A real descriptor failure must not produce a completion or publication. */
TEST(Test__PrefixArchiveMaintenanceWriter, NativeFailurePermanentlyRejectsFurtherWork)
{
    NativeTestFile file;
    PrefixArchiveMaintenanceWriter writer(file.fd());
    file.close();
    const std::array<std::byte, 1> bytes{std::byte{1}};
    std::string error;
    EXPECT_FALSE(writer.write(bytes, &error));
    EXPECT_NE(error.find("Bad file descriptor"), std::string::npos);
    EXPECT_FALSE(writer.write(bytes, &error));
    EXPECT_EQ(error, "prefix maintenance writer already failed");
    EXPECT_EQ(writer.evidence().durable_writes, 0u);
    EXPECT_EQ(writer.evidence().durable_bytes, 0u);
}

/** @test The O_SYNC superset also has the required native O_DSYNC contract. */
TEST(Test__PrefixArchiveMaintenanceWriter, FullSynchronousDescriptorIsAccepted)
{
    NativeTestFile file(O_RDWR | O_SYNC);
    PrefixArchiveMaintenanceWriter writer(file.fd());
    const std::array<std::byte, 1> bytes{std::byte{0x71}};
    EXPECT_TRUE(writer.write(bytes));
    EXPECT_EQ(writer.evidence().native_open_flags & O_DSYNC, O_DSYNC);
    EXPECT_EQ(writer.evidence().durable_bytes, bytes.size());
}
