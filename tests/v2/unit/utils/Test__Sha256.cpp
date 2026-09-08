/**
 * @file Test__Sha256.cpp
 * @brief Device-free correctness tests for compact content and file-set identity.
 *
 * Large model artifacts use filesystem identity rather than rereading every
 * payload byte. These tests prove that an ordered split-model descriptor
 * changes when any member is replaced or mutated.
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

    } // namespace

    TEST(Test__Sha256, FileSetIdentityCoversEveryShardWithoutReadingPayload)
    {
        ScopedTestDirectory directory;
        const auto first = directory.path() / "model-00001.gguf";
        const auto second = directory.path() / "model-00002.gguf";
        writeBytes(first, "first-shard");
        writeBytes(second, "second-shard");

        std::string error;
        const auto initial =
            sha256FileSetIdentityHex({first, second}, &error);
        ASSERT_TRUE(initial.has_value()) << error;
        EXPECT_EQ(
            sha256FileSetIdentityHex({first, second}, &error),
            initial)
            << error;
        EXPECT_NE(
            sha256FileSetIdentityHex({second, first}, &error),
            initial)
            << "Split-file order is part of the loaded artifact identity";

        // A sparse payload this large would make a hidden content scan violate
        // the fast unit gate; filesystem-identity capture performs only stat.
        const auto sparse = directory.path() / "model-00003.gguf";
        writeBytes(sparse, {});
        std::filesystem::resize_file(sparse, 64ull * 1024ull * 1024ull * 1024ull);
        const auto with_sparse =
            sha256FileSetIdentityHex({first, second, sparse}, &error);
        ASSERT_TRUE(with_sparse.has_value()) << error;
        EXPECT_NE(with_sparse, initial);

        writeBytes(second, "changed-shard");
        const auto changed =
            sha256FileSetIdentityHex({first, second}, &error);
        ASSERT_TRUE(changed.has_value()) << error;
        EXPECT_NE(changed, initial)
            << "Mutating any GGUF shard must select a new archive namespace";
    }
} // namespace llaminar2::test
