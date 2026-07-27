/**
 * @file Sha256.cpp
 * @brief Bounded streaming SHA-256 implementation backed by OpenSSL EVP.
 */

#include "utils/Sha256.h"

#include <openssl/evp.h>

#include <array>
#include <fstream>
#include <future>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <unordered_map>
#include <vector>

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

        std::string fileIdentityKey(const std::filesystem::path &path)
        {
            std::error_code ec;
            const auto canonical = std::filesystem::weakly_canonical(path, ec);
            const std::filesystem::path normalized = ec ? path.lexically_normal() : canonical;

            ec.clear();
            const auto size = std::filesystem::file_size(path, ec);
            if (ec)
                return {};

            ec.clear();
            const auto modified = std::filesystem::last_write_time(path, ec);
            if (ec)
                return {};

            /*
             * This key only deduplicates work during one process lifetime.
             * The digest itself still comes from every model byte. Including
             * size and mtime prevents a normal in-place replacement from
             * reusing an earlier process-local result.
             */
            return normalized.string() + "\n" +
                   std::to_string(size) + "\n" +
                   std::to_string(modified.time_since_epoch().count());
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
    } // namespace

    std::optional<std::string> sha256FileHex(
        const std::filesystem::path &path,
        std::string *error)
    {
        const std::string identity = fileIdentityKey(path);
        if (identity.empty())
        {
            if (error)
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
                producer->set_value(computeDigest(path));
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
} // namespace llaminar2
