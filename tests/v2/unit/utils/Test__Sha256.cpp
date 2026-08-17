/**
 * @file Test__Sha256.cpp
 * @brief Device-free correctness tests for content identity and shared digest caching.
 *
 * Production parity authenticates large GGUF files against the Hugging Face
 * reference metadata. These tests lock down the campaign optimization that
 * lets cooperating processes share that expensive byte scan: an unchanged
 * file reuses one atomically published digest, normal replacement creates a
 * new identity, and malformed cache state fails closed.
 */

#include <gtest/gtest.h>

#include "utils/Sha256.h"

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <sys/stat.h>
#include <unistd.h>

namespace llaminar2::test
{
    namespace
    {
        /** @brief Own one explicit test directory and remove only that directory. */
        class ScopedTestDirectory
        {
        public:
            /** @brief Create a process-unique directory beneath the system temp root. */
            ScopedTestDirectory()
                : path_(std::filesystem::temp_directory_path() /
                        ("llaminar-sha256-" + std::to_string(::getpid()) + "-" +
                         std::to_string(next_id_++)))
            {
                std::error_code error;
                std::filesystem::create_directories(path_, error);
                EXPECT_FALSE(error) << error.message();
            }

            ScopedTestDirectory(const ScopedTestDirectory &) = delete;
            ScopedTestDirectory &operator=(const ScopedTestDirectory &) = delete;

            /** @brief Restore owner permissions and remove the owned test tree. */
            ~ScopedTestDirectory()
            {
                std::error_code ignored;
                std::filesystem::permissions(
                    path_,
                    std::filesystem::perms::owner_all,
                    std::filesystem::perm_options::add,
                    ignored);
                std::filesystem::remove_all(path_, ignored);
            }

            /** @brief Return the exact directory owned by this scope. */
            const std::filesystem::path &path() const
            {
                return path_;
            }

        private:
            inline static unsigned int next_id_ = 0;
            std::filesystem::path path_;
        };

        /** @brief Replace a small fixture file with the supplied exact bytes. */
        void writeBytes(
            const std::filesystem::path &path,
            const std::string &bytes)
        {
            std::ofstream output(path, std::ios::binary | std::ios::trunc);
            ASSERT_TRUE(output.is_open());
            output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
            output.close();
            ASSERT_TRUE(output);
        }

        /** @brief Return all complete digest entries, excluding advisory locks. */
        std::vector<std::filesystem::path> digestEntries(
            const std::filesystem::path &cache)
        {
            std::vector<std::filesystem::path> entries;
            for (const auto &entry : std::filesystem::directory_iterator(cache))
            {
                if (entry.path().extension() == ".sha256")
                    entries.push_back(entry.path());
            }
            return entries;
        }
    } // namespace

    TEST(Test__Sha256, SharedCacheReusesUnchangedAuthenticatedIdentity)
    {
        ScopedTestDirectory directory;
        const auto model = directory.path() / "model.gguf";
        const auto cache = directory.path() / "digest-cache";
        writeBytes(model, "exact-model-bytes");

        std::string error;
        const auto first = sha256FileHexShared(model, cache, &error);
        ASSERT_TRUE(first.has_value()) << error;
        ASSERT_EQ(digestEntries(cache).size(), 1u);

        /*
         * A cache hit only needs to open the existing lock and entry. Making
         * the directory non-writable means a hidden rescan/publication attempt
         * would fail, while a legitimate hit remains usable.
         */
        ASSERT_EQ(::chmod(cache.c_str(), S_IRUSR | S_IXUSR), 0);
        const auto second = sha256FileHexShared(model, cache, &error);
        EXPECT_EQ(second, first) << error;
        EXPECT_EQ(::chmod(cache.c_str(), S_IRWXU), 0);
    }

    TEST(Test__Sha256, SharedCacheRejectsAReplacedSameSizeFileIdentity)
    {
        ScopedTestDirectory directory;
        const auto model = directory.path() / "model.gguf";
        const auto cache = directory.path() / "digest-cache";
        writeBytes(model, "alpha");

        std::string error;
        const auto first = sha256FileHexShared(model, cache, &error);
        ASSERT_TRUE(first.has_value()) << error;

        // Same byte length with a new inode defeats path/size-only memoization.
        const auto replacement = directory.path() / "replacement.gguf";
        writeBytes(replacement, "bravo");
        std::filesystem::rename(replacement, model);
        const auto second = sha256FileHexShared(model, cache, &error);
        ASSERT_TRUE(second.has_value()) << error;
        EXPECT_NE(second, first);
        EXPECT_EQ(digestEntries(cache).size(), 2u);
    }

    TEST(Test__Sha256, SharedCacheFailsClosedOnMalformedPublishedEntry)
    {
        ScopedTestDirectory directory;
        const auto model = directory.path() / "model.gguf";
        const auto cache = directory.path() / "digest-cache";
        writeBytes(model, "exact-model-bytes");

        std::string error;
        ASSERT_TRUE(sha256FileHexShared(model, cache, &error).has_value()) << error;
        const auto entries = digestEntries(cache);
        ASSERT_EQ(entries.size(), 1u);
        writeBytes(entries.front(), "truncated-cache-entry\n");

        const auto digest = sha256FileHexShared(model, cache, &error);
        EXPECT_FALSE(digest.has_value());
        EXPECT_NE(error.find("invalid authenticated SHA-256 cache entry"),
                  std::string::npos)
            << error;
    }
} // namespace llaminar2::test
