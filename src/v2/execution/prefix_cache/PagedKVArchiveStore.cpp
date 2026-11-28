/**
 * @file PagedKVArchiveStore.cpp
 * @brief Durable finite-quota disk store implementation.
 */
#include "execution/prefix_cache/PagedKVArchiveStore.h"

#include "execution/prefix_cache/BlockHash.h"

#include <array>
#include <cerrno>
#include <cstdio>
#include <fcntl.h>
#include <filesystem>
#include <stdexcept>
#include <system_error>
#include <unistd.h>

namespace llaminar2
{
namespace
{
/** Rounding/block overhead charged per extent (metadata + rounding), so the
 * quota is complete and inclusive (never a payload-only budget). */
constexpr std::size_t kExtentOverheadBytes = 4096;
/** Engine-owned coefficient that makes the temp pattern unambiguous: any file
 * that does not match this name belongs to someone else and is never touched. */
constexpr const char *kTempPrefix = ".llaminar-kv-";

std::size_t roundUp(std::size_t value, std::size_t block)
{
    if (block == 0) return value;
    const auto rem = value % block;
    return rem == 0 ? value : value + (block - rem);
}

std::uint64_t archiveDigest(const PagedKVArchive &archive)
{
    const auto bytes = archive.serialize();
    return hashPrefixBytes(bytes.data(), bytes.size());
}

bool writeAll(int fd, std::span<const std::byte> bytes)
{
    std::size_t done = 0;
    while (done < bytes.size())
    {
        const auto n = ::write(fd, bytes.data() + done, bytes.size() - done);
        if (n < 0)
        {
            if (errno == EINTR) continue;
            return false;
        }
        done += static_cast<std::size_t>(n);
    }
    return true;
}
} // namespace

PagedKVArchiveStore::PagedKVArchiveStore(const PagedKVDiskRoute &route)
    : quota_(makePagedKVDiskQuota(route)), directory_(route.directory)
{
    if (!route.valid)
        throw std::invalid_argument("Paged KV archive store requires a valid route: " + route.refusal);
    if (quota_.capacity() == 0)
        throw std::invalid_argument("Paged KV archive store requires a positive journal membership");
    // A DISABLED tier is the representable off state: no directory exists and
    // the quota already refuses every admit (quota 0); no directory check is
    // applied because no store ever touches storage.
    if (route.directory.empty())
        return;
    std::error_code ec;
    const auto status = std::filesystem::status(directory_, ec);
    if (ec || !std::filesystem::is_directory(status))
        throw std::invalid_argument(
            "Paged KV archive store directory is not an existing directory: " + directory_);
}

PagedKVArchiveStore::PagedKVArchiveStore(PagedKVArchiveStore &&) noexcept = default;
PagedKVArchiveStore &PagedKVArchiveStore::operator=(PagedKVArchiveStore &&) noexcept = default;
PagedKVArchiveStore::~PagedKVArchiveStore() = default;

std::string PagedKVArchiveStore::pathFor(const PagedKVArchiveExtentKey &key) const
{
    // Exact engine-ownable file name from the typed identity (never a caller
    // path input, never a traversal — the identity is fully encoded).
    return directory_ + "/" + prefixHashHex(key.digest) + "." +
        std::to_string(key.namespace_id) + "." + std::to_string(key.generation);
}

std::optional<PagedKVArchiveExtentKey> PagedKVArchiveStore::put(
    const PagedKVArchive &archive, std::uint64_t generation,
    std::uint64_t namespace_id, std::string *refusal)
{
    const auto clear_refusal = [&]() {
        if (refusal) refusal->clear();
    };
    // Provenance gate before any quota mutation or byte: only an FP32
    // attention section is a storable extent (the plan's typed archive rule).
    if (archive.empty() || archive.meta().format != KVStorageFormat::FP32 ||
        archive.meta().family != KVCacheFamily::AttentionOnly)
    {
        if (refusal) *refusal = "only nonempty FP32 attention sections are storable";
        return std::nullopt;
    }
    const auto serialized = archive.serialize();
    const std::size_t bytes = roundUp(serialized.size(), kExtentOverheadBytes) +
        kExtentOverheadBytes;
    const std::uint64_t digest = archiveDigest(archive);
    const PagedKVArchiveExtentKey key{generation, namespace_id, digest, bytes};
    // Admit BEFORE any byte reaches storage; refused = no file, no admission.
    if (!quota_.admit({generation, namespace_id, digest, bytes}))
    {
        if (refusal) *refusal = "disk quota cannot admit the rounded extent (quota full)";
        return std::nullopt;
    }
    // Durable completion: same-directory temp, write all, fsync, rename,
    // directory fsync — a crash never publishes a partial extent.
    const std::string final_path = pathFor(key);
    const std::string temp_path =
        directory_ + "/" + kTempPrefix + std::to_string(::getpid()) + ".tmp";
    int fd = -1;
    bool ok = false;
    do
    {
        fd = ::open(temp_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
        if (fd < 0) break;
        if (!writeAll(fd, std::as_bytes(std::span(serialized)))) break;
        if (::fsync(fd) != 0) break;
        if (::close(fd) != 0) { fd = -1; break; }
        fd = -1;
        if (::rename(temp_path.c_str(), final_path.c_str()) != 0) break;
        // Make the rename durable (the directory entry, not the data).
        const int dir_fd = ::open(directory_.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
        if (dir_fd >= 0)
        {
            (void)::fsync(dir_fd);
            (void)::close(dir_fd);
        }
        ok = true;
    } while (false);
    if (fd >= 0) (void)::close(fd);
    if (!ok)
    {
        (void)::unlink(temp_path.c_str());
        (void)::unlink(final_path.c_str());
        quota_.retire({generation, namespace_id, digest, bytes});
        if (refusal)
            *refusal = "disk write failed (durable completion not achieved): " +
                std::string(std::strerror(errno));
        return std::nullopt;
    }
    clear_refusal();
    return key;
}

std::optional<PagedKVArchive> PagedKVArchiveStore::get(
    const PagedKVArchiveExtentKey &key, std::string *refusal) const
{
    if (!quota_.contains({key.generation, key.namespace_id, key.digest, key.bytes}))
    {
        if (refusal) *refusal = "extent identity was never admitted";
        return std::nullopt;
    }
    std::error_code ec;
    const auto size = std::filesystem::file_size(pathFor(key), ec);
    if (ec || size > (1ull << 32))
    {
        if (refusal) *refusal = "extent file is missing or unreadably large";
        return std::nullopt;
    }
    std::vector<std::byte> bytes(static_cast<std::size_t>(size));
    const int fd = ::open(pathFor(key).c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0)
    {
        if (refusal) *refusal = "extent file cannot be opened";
        return std::nullopt;
    }
    std::size_t done = 0;
    while (done < bytes.size())
    {
        const auto n = ::read(fd, bytes.data() + done, bytes.size() - done);
        if (n < 0)
        {
            if (errno == EINTR) continue;
            (void)::close(fd);
            if (refusal) *refusal = "extent read failed";
            return std::nullopt;
        }
        if (n == 0) break;
        done += static_cast<std::size_t>(n);
    }
    (void)::close(fd);
    if (done != bytes.size())
    {
        if (refusal) *refusal = "extent file is truncated";
        return std::nullopt;
    }
    try
    {
        auto archive = PagedKVArchive::parse(bytes);
        // Content is provenance-authenticated AND digest-bound: a tampered or
        // partially overwritten extent refuses (never a silent byte change).
        if (archiveDigest(archive) != key.digest)
        {
            if (refusal) *refusal = "extent digest mismatch (corrupt or tampered)";
            return std::nullopt;
        }
        return archive;
    }
    catch (const std::exception &e)
    {
        if (refusal) *refusal = std::string("extent parse refused: ") + e.what();
        return std::nullopt;
    }
}

bool PagedKVArchiveStore::retire(const PagedKVArchiveExtentKey &key)
{
    if (!quota_.contains({key.generation, key.namespace_id, key.digest, key.bytes}))
        return false;
    std::error_code ec;
    const auto removed = std::filesystem::remove(pathFor(key), ec);
    if (ec || !removed) return false;
    quota_.retire({key.generation, key.namespace_id, key.digest, key.bytes});
    return true;
}

std::size_t PagedKVArchiveStore::cleanOrphanedTemps() const
{
    std::size_t removed = 0;
    std::error_code ec;
    for (std::filesystem::directory_iterator it(directory_, ec),
             end = std::filesystem::directory_iterator();
         !ec && it != end; it.increment(ec))
    {
        const auto &entry = *it;
        const auto name = entry.path().filename().string();
        // Engine-only namespace rule: ONLY the exact temp pattern is
        // ever cleaned; named extents and foreign files are never touched.
        if (name.rfind(kTempPrefix, 0) != 0) continue;
        if (!entry.is_regular_file(ec)) continue;
        if (std::filesystem::remove(entry.path(), ec) && !ec) ++removed;
    }
    return removed;
}
} // namespace llaminar2
