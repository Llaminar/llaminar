/**
 * @file Sha256.cpp
 * @brief Bounded streaming SHA-256 implementation backed by OpenSSL EVP.
 */

#include "utils/Sha256.h"

#include <openssl/evp.h>

#include <array>
#include <cerrno>
#include <cstring>
#include <fstream>
#include <future>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <unordered_map>
#include <vector>

#include <sys/stat.h>

namespace llaminar2
{
    namespace
    {
        /**
         * @brief Result retained by the in-process coalescing table.
         *
         * Keeping the error beside the optional digest lets every waiter
         * receive the same useful failure instead of only the thread that
         * performed the file scan.
         */
        struct DigestResult
        {
            std::optional<std::string> digest;
            std::string error;
        };

        /** @brief Serialize every stable filesystem field used by digest memoization. */
        std::string fileIdentityKey(
            const std::filesystem::path &path,
            std::string *error = nullptr)
        {
            std::error_code ec;
            const auto canonical = std::filesystem::weakly_canonical(path, ec);
            if (ec)
            {
                if (error)
                    *error = "failed to canonicalize file identity: " +
                             path.string() + ": " + ec.message();
                return {};
            }

            struct stat status {};
            if (::stat(canonical.c_str(), &status) != 0)
            {
                if (error)
                    *error = "failed to stat file identity: " +
                             canonical.string() + ": " + std::strerror(errno);
                return {};
            }
            if (!S_ISREG(status.st_mode))
            {
                if (error)
                    *error = "file identity requires a regular file: " +
                             canonical.string();
                return {};
            }

            /*
             * Device/inode distinguish replacement files even when a caller
             * preserves the pathname. ctime closes the ordinary same-size,
             * restored-mtime hole; mtime and nanoseconds retain cheap change
             * detection on filesystems that preserve inode identity.
             */
            return canonical.string() + "\n" +
                   std::to_string(static_cast<unsigned long long>(status.st_dev)) + "\n" +
                   std::to_string(static_cast<unsigned long long>(status.st_ino)) + "\n" +
                   std::to_string(static_cast<unsigned long long>(status.st_size)) + "\n" +
                   std::to_string(static_cast<long long>(status.st_mtim.tv_sec)) + "\n" +
                   std::to_string(static_cast<long long>(status.st_mtim.tv_nsec)) + "\n" +
                   std::to_string(static_cast<long long>(status.st_ctim.tv_sec)) + "\n" +
                   std::to_string(static_cast<long long>(status.st_ctim.tv_nsec));
        }

        DigestResult computeDigest(const std::filesystem::path &path)
        {
            std::ifstream input(path, std::ios::binary);
            if (!input)
            {
                return {std::nullopt, "failed to open model file for SHA-256: " + path.string()};
            }

            using ContextOwner = std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)>;
            ContextOwner context(EVP_MD_CTX_new(), &EVP_MD_CTX_free);
            if (!context || EVP_DigestInit_ex(context.get(), EVP_sha256(), nullptr) != 1)
            {
                return {std::nullopt, "failed to initialize OpenSSL SHA-256 context"};
            }

            constexpr size_t kReadBytes = 8ull * 1024ull * 1024ull;
            std::vector<unsigned char> buffer(kReadBytes);
            while (input)
            {
                input.read(
                    reinterpret_cast<char *>(buffer.data()),
                    static_cast<std::streamsize>(buffer.size()));
                const std::streamsize bytes_read = input.gcount();
                if (bytes_read > 0 &&
                    EVP_DigestUpdate(
                        context.get(),
                        buffer.data(),
                        static_cast<size_t>(bytes_read)) != 1)
                {
                    return {std::nullopt, "OpenSSL rejected SHA-256 input bytes"};
                }
            }
            if (!input.eof())
            {
                return {std::nullopt, "failed while reading model file for SHA-256: " + path.string()};
            }

            std::array<unsigned char, EVP_MAX_MD_SIZE> digest{};
            unsigned int digest_bytes = 0;
            if (EVP_DigestFinal_ex(context.get(), digest.data(), &digest_bytes) != 1 ||
                digest_bytes != 32)
            {
                return {std::nullopt, "failed to finalize SHA-256 digest"};
            }

            std::ostringstream hex;
            hex << std::hex << std::setfill('0');
            for (unsigned int index = 0; index < digest_bytes; ++index)
                hex << std::setw(2) << static_cast<unsigned int>(digest[index]);
            return {hex.str(), {}};
        }

        DigestResult computeBytesDigest(std::string_view bytes)
        {
            using ContextOwner = std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)>;
            ContextOwner context(EVP_MD_CTX_new(), &EVP_MD_CTX_free);
            if (!context || EVP_DigestInit_ex(context.get(), EVP_sha256(), nullptr) != 1)
            {
                return {std::nullopt, "failed to initialize OpenSSL SHA-256 context"};
            }

            if (!bytes.empty() &&
                EVP_DigestUpdate(context.get(), bytes.data(), bytes.size()) != 1)
            {
                return {std::nullopt, "OpenSSL rejected SHA-256 input bytes"};
            }

            std::array<unsigned char, EVP_MAX_MD_SIZE> digest{};
            unsigned int digest_bytes = 0;
            if (EVP_DigestFinal_ex(context.get(), digest.data(), &digest_bytes) != 1 ||
                digest_bytes != 32)
            {
                return {std::nullopt, "failed to finalize SHA-256 digest"};
            }

            std::ostringstream hex;
            hex << std::hex << std::setfill('0');
            for (unsigned int index = 0; index < digest_bytes; ++index)
                hex << std::setw(2) << static_cast<unsigned int>(digest[index]);
            return {hex.str(), {}};
        }

    } // namespace

    std::optional<std::string> sha256FileHex(
        const std::filesystem::path &path,
        std::string *error)
    {
        const std::string identity = fileIdentityKey(path, error);
        if (identity.empty())
        {
            if (error && error->empty())
                *error = "model file does not have a readable stable identity: " + path.string();
            return std::nullopt;
        }

        static std::mutex cache_mutex;
        static std::unordered_map<std::string, std::shared_future<DigestResult>> cache;

        std::shared_future<DigestResult> shared_result;
        std::optional<std::promise<DigestResult>> producer;
        {
            std::lock_guard<std::mutex> lock(cache_mutex);
            auto existing = cache.find(identity);
            if (existing != cache.end())
            {
                shared_result = existing->second;
            }
            else
            {
                producer.emplace();
                shared_result = producer->get_future().share();
                cache.emplace(identity, shared_result);
            }
        }

        if (producer)
        {
            try
            {
                DigestResult computed = computeDigest(path);
                if (computed.digest)
                {
                    std::string final_identity_error;
                    const std::string final_identity =
                        fileIdentityKey(path, &final_identity_error);
                    if (final_identity != identity)
                    {
                        computed = {
                            std::nullopt,
                            final_identity.empty()
                                ? std::move(final_identity_error)
                                : "file identity changed during SHA-256 "
                                  "authentication: " +
                                      path.string()};
                    }
                }
                producer->set_value(std::move(computed));
            }
            catch (...)
            {
                producer->set_exception(std::current_exception());
            }
        }

        try
        {
            const DigestResult result = shared_result.get();
            if (error)
                *error = result.error;
            return result.digest;
        }
        catch (const std::exception &exception)
        {
            if (error)
                *error = std::string("SHA-256 calculation failed: ") + exception.what();
            return std::nullopt;
        }
    }

    std::optional<std::string> sha256FileSetIdentityHex(
        const std::vector<std::filesystem::path> &paths,
        std::string *error)
    {
        if (error)
            error->clear();
        if (paths.empty())
        {
            if (error)
                *error = "model artifact identity requires at least one file";
            return std::nullopt;
        }

        std::string material = "llaminar-model-artifact-filesystem-v1\n";
        material += std::to_string(paths.size()) + "\n";
        for (std::size_t index = 0; index < paths.size(); ++index)
        {
            std::string identity_error;
            const std::string identity =
                fileIdentityKey(paths[index], &identity_error);
            if (identity.empty())
            {
                if (error)
                {
                    *error = "failed to capture model artifact file " +
                             std::to_string(index) + ": " + identity_error;
                }
                return std::nullopt;
            }

            /*
             * Length-prefix every descriptor so path newlines and adjacent
             * split identities cannot produce an ambiguous concatenation.
             */
            material += std::to_string(index) + "\n";
            material += std::to_string(identity.size()) + "\n";
            material += identity;
            material += '\n';
        }

        const DigestResult digest = computeBytesDigest(material);
        if (error)
            *error = digest.error;
        return digest.digest;
    }

    std::optional<std::string> sha256BytesHex(
        std::string_view bytes,
        std::string *error)
    {
        const DigestResult result = computeBytesDigest(bytes);
        if (error)
            *error = result.error;
        return result.digest;
    }
} // namespace llaminar2
