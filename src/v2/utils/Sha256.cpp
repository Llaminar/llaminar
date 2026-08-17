/**
 * @file Sha256.cpp
 * @brief Bounded streaming SHA-256 implementation backed by OpenSSL EVP.
 */

#include "utils/Sha256.h"

#include <openssl/evp.h>

#include <algorithm>
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

#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

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
                    *error = "failed to canonicalize file for SHA-256 identity: " +
                             path.string() + ": " + ec.message();
                return {};
            }

            struct stat status {};
            if (::stat(canonical.c_str(), &status) != 0)
            {
                if (error)
                    *error = "failed to stat file for SHA-256 identity: " +
                             canonical.string() + ": " + std::strerror(errno);
                return {};
            }
            if (!S_ISREG(status.st_mode))
            {
                if (error)
                    *error = "SHA-256 identity requires a regular file: " +
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

        /** @brief RAII owner for one advisory cross-process cache lock. */
        class ScopedFileLock
        {
        public:
            ScopedFileLock() = default;
            ScopedFileLock(const ScopedFileLock &) = delete;
            ScopedFileLock &operator=(const ScopedFileLock &) = delete;

            /** @brief Transfer the unique lock-file descriptor. */
            ScopedFileLock(ScopedFileLock &&other) noexcept
                : descriptor_(other.descriptor_)
            {
                other.descriptor_ = -1;
            }

            /** @brief Release any owned descriptor before accepting another. */
            ScopedFileLock &operator=(ScopedFileLock &&other) noexcept
            {
                if (this != &other)
                {
                    release();
                    descriptor_ = other.descriptor_;
                    other.descriptor_ = -1;
                }
                return *this;
            }

            /** @brief Release the advisory lock by closing its descriptor. */
            ~ScopedFileLock()
            {
                release();
            }

            /**
             * @brief Open and exclusively lock one identity-specific file.
             * @param path Lock-file path inside the trusted cache directory.
             * @param error Receives an actionable system diagnostic.
             * @return Owning lock on success.
             */
            static std::optional<ScopedFileLock> acquire(
                const std::filesystem::path &path,
                std::string *error)
            {
                int flags = O_CREAT | O_RDWR | O_CLOEXEC;
#ifdef O_NOFOLLOW
                flags |= O_NOFOLLOW;
#endif
                const int descriptor = ::open(path.c_str(), flags, S_IRUSR | S_IWUSR);
                if (descriptor < 0)
                {
                    if (error)
                        *error = "failed to open SHA-256 cache lock " +
                                 path.string() + ": " + std::strerror(errno);
                    return std::nullopt;
                }

                while (::flock(descriptor, LOCK_EX) != 0)
                {
                    if (errno == EINTR)
                        continue;
                    if (error)
                        *error = "failed to acquire SHA-256 cache lock " +
                                 path.string() + ": " + std::strerror(errno);
                    ::close(descriptor);
                    return std::nullopt;
                }

                ScopedFileLock lock;
                lock.descriptor_ = descriptor;
                return lock;
            }

        private:
            /** @brief Close the descriptor; `flock` releases with the open description. */
            void release() noexcept
            {
                if (descriptor_ >= 0)
                {
                    ::close(descriptor_);
                    descriptor_ = -1;
                }
            }

            int descriptor_ = -1; ///< Unique open description carrying the lock.
        };

        enum class SharedCacheReadStatus
        {
            Missing,
            Hit,
            Invalid,
        };

        /** @brief Validate and read one atomically published shared-cache entry. */
        SharedCacheReadStatus readSharedCacheEntry(
            const std::filesystem::path &entry_path,
            const std::string &identity_digest,
            std::string *digest,
            std::string *error)
        {
            std::error_code exists_error;
            const bool exists = std::filesystem::exists(entry_path, exists_error);
            if (exists_error)
            {
                if (error)
                    *error = "failed to inspect SHA-256 cache entry " +
                             entry_path.string() + ": " + exists_error.message();
                return SharedCacheReadStatus::Invalid;
            }
            if (!exists)
                return SharedCacheReadStatus::Missing;

            std::ifstream input(entry_path);
            std::string magic;
            std::string cached_identity;
            std::string cached_digest;
            std::string extra;
            if (!input || !std::getline(input, magic) ||
                !std::getline(input, cached_identity) ||
                !std::getline(input, cached_digest) || std::getline(input, extra) ||
                magic != "llaminar-sha256-cache-v1" ||
                cached_identity != identity_digest ||
                cached_digest.size() != 64 ||
                !std::all_of(
                    cached_digest.begin(), cached_digest.end(),
                    [](unsigned char character)
                    {
                        return (character >= '0' && character <= '9') ||
                               (character >= 'a' && character <= 'f');
                    }))
            {
                if (error)
                    *error = "invalid authenticated SHA-256 cache entry: " +
                             entry_path.string();
                return SharedCacheReadStatus::Invalid;
            }

            *digest = std::move(cached_digest);
            return SharedCacheReadStatus::Hit;
        }

        /** @brief Atomically publish one complete shared-cache entry. */
        bool writeSharedCacheEntry(
            const std::filesystem::path &entry_path,
            const std::string &identity_digest,
            const std::string &digest,
            std::string *error)
        {
            const std::filesystem::path temporary =
                entry_path.string() + ".tmp." + std::to_string(::getpid());
            {
                std::ofstream output(temporary, std::ios::trunc);
                if (!output)
                {
                    if (error)
                        *error = "failed to open temporary SHA-256 cache entry: " +
                                 temporary.string();
                    return false;
                }
                output << "llaminar-sha256-cache-v1\n"
                       << identity_digest << '\n'
                       << digest << '\n';
                output.close();
                if (!output)
                {
                    if (error)
                        *error = "failed to write temporary SHA-256 cache entry: " +
                                 temporary.string();
                    std::error_code remove_error;
                    std::filesystem::remove(temporary, remove_error);
                    return false;
                }
            }

            std::error_code rename_error;
            std::filesystem::rename(temporary, entry_path, rename_error);
            if (rename_error)
            {
                if (error)
                    *error = "failed to publish SHA-256 cache entry " +
                             entry_path.string() + ": " + rename_error.message();
                std::error_code remove_error;
                std::filesystem::remove(temporary, remove_error);
                return false;
            }
            return true;
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

    std::optional<std::string> sha256FileHexShared(
        const std::filesystem::path &path,
        const std::filesystem::path &cache_directory,
        std::string *error)
    {
        if (error)
            error->clear();
        if (cache_directory.empty())
        {
            if (error)
                *error = "shared SHA-256 cache directory is empty";
            return std::nullopt;
        }

        std::string identity_error;
        const std::string identity = fileIdentityKey(path, &identity_error);
        if (identity.empty())
        {
            if (error)
                *error = std::move(identity_error);
            return std::nullopt;
        }
        const DigestResult identity_hash = computeBytesDigest(identity);
        if (!identity_hash.digest)
        {
            if (error)
                *error = "failed to hash SHA-256 cache identity: " +
                         identity_hash.error;
            return std::nullopt;
        }

        std::error_code directory_error;
        std::filesystem::create_directories(cache_directory, directory_error);
        std::error_code type_error;
        const bool is_directory =
            std::filesystem::is_directory(cache_directory, type_error);
        if (directory_error || type_error || !is_directory)
        {
            if (error)
                *error = "failed to prepare shared SHA-256 cache directory " +
                         cache_directory.string() +
                         (directory_error
                              ? ": " + directory_error.message()
                              : type_error
                                    ? ": " + type_error.message()
                                    : ": path is not a directory");
            return std::nullopt;
        }

        const std::filesystem::path entry_path =
            cache_directory / (*identity_hash.digest + ".sha256");
        const std::filesystem::path lock_path =
            cache_directory / (*identity_hash.digest + ".lock");
        auto lock = ScopedFileLock::acquire(lock_path, error);
        if (!lock)
            return std::nullopt;

        /*
         * A process may have waited behind the producer for a large model.
         * Re-stat after acquiring the lock so the cache key cannot describe
         * bytes replaced during that wait.
         */
        std::string locked_identity_error;
        const std::string locked_identity =
            fileIdentityKey(path, &locked_identity_error);
        if (locked_identity != identity)
        {
            if (error)
                *error = locked_identity.empty()
                             ? std::move(locked_identity_error)
                             : "file identity changed while waiting for shared SHA-256 authentication: " +
                                   path.string();
            return std::nullopt;
        }

        std::string cached_digest;
        const SharedCacheReadStatus read_status = readSharedCacheEntry(
            entry_path, *identity_hash.digest, &cached_digest, error);
        if (read_status == SharedCacheReadStatus::Hit)
        {
            std::string final_identity_error;
            const std::string final_identity =
                fileIdentityKey(path, &final_identity_error);
            if (final_identity != identity)
            {
                if (error)
                    *error = final_identity.empty()
                                 ? std::move(final_identity_error)
                                 : "file identity changed while consuming shared "
                                   "SHA-256 authentication: " +
                                       path.string();
                return std::nullopt;
            }
            return cached_digest;
        }
        if (read_status == SharedCacheReadStatus::Invalid)
            return std::nullopt;

        const DigestResult computed = computeDigest(path);
        if (!computed.digest)
        {
            if (error)
                *error = computed.error;
            return std::nullopt;
        }

        std::string final_identity_error;
        const std::string final_identity = fileIdentityKey(path, &final_identity_error);
        if (final_identity != identity)
        {
            if (error)
                *error = final_identity.empty()
                             ? std::move(final_identity_error)
                             : "file identity changed during shared SHA-256 authentication: " +
                                   path.string();
            return std::nullopt;
        }
        if (!writeSharedCacheEntry(
                entry_path, *identity_hash.digest, *computed.digest, error))
        {
            return std::nullopt;
        }
        return computed.digest;
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
