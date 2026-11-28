/**
 * @file PagedKVArchiveStore.cpp
 * @brief Durable finite-quota disk store implementation.
 */
#include "execution/prefix_cache/PagedKVArchiveStore.h"

#include "execution/prefix_cache/BlockHash.h"

#include <array>
#include <atomic>
#include <cerrno>
#include <cstdio>
#include <fcntl.h>
#include <filesystem>
#include <signal.h>
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
/** Engine-owned temp name shape: `.llaminar-kv-<pid>-<ordinal>.tmp` (fresh
 * exclusive per-operation names; a foreign file never matches this pattern
 * and a live process's temps are never removed). */
constexpr const char *kTempPrefix = ".llaminar-kv-";
constexpr const char *kTempSuffix = ".tmp";

std::atomic<std::uint64_t> gTempOrdinal{0};

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

/** @return True when the pid is not provably alive in this pid namespace. */
bool pidIsGone(pid_t pid)
{
    if (pid <= 0) return true;
    if (::kill(pid, 0) != 0) return errno == ESRCH;
    return false; // Signal 0 delivered: the process (or a reused one) is alive.
}

/**
 * @brief Parse one engine temp name; returns the owning pid or nullopt when
 * the name does not match the exact engine temp pattern (`<pid>-<ordinal>`).
 */
std::optional<pid_t> parseTempOwner(const std::string &name)
{
    if (name.rfind(kTempPrefix, 0) != 0) return std::nullopt;
    if (name.size() <= std::string(kTempPrefix).size() + std::string(kTempSuffix).size())
        return std::nullopt;
    const std::string body =
        name.substr(std::string(kTempPrefix).size(),
            name.size() - std::string(kTempPrefix).size() - std::string(kTempSuffix).size());
    const auto dash = body.find('-');
    if (dash == std::string::npos || dash == 0) return std::nullopt;
    const std::string pid_text = body.substr(0, dash);
    const std::string ord_text = body.substr(dash + 1);
    if (pid_text.empty() || ord_text.empty()) return std::nullopt;
    if (pid_text.find_first_not_of("0123456789") != std::string::npos) return std::nullopt;
    if (ord_text.find_first_not_of("0123456789") != std::string::npos) return std::nullopt;
    const long pid = std::strtol(pid_text.c_str(), nullptr, 10);
    if (pid <= 0 || pid > 4194304) return std::nullopt;
    return static_cast<pid_t>(pid);
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
    // FRESH EXCLUSIVE TEMP: a per-operation unique name is created with
    // O_CREAT|O_EXCL (mkstemp-equivalent), so two concurrent puts can never
    // collide on (or truncate) each other's temp file, and a retry after a
    // stale temp collision gets a NEW name.
    const std::string final_path = pathFor(key);
    std::string temp_path;
    int fd = -1;
    bool ok = false;
    for (unsigned attempt = 0; attempt < 64; ++attempt)
    {
        const std::uint64_t ordinal = gTempOrdinal.fetch_add(1, std::memory_order_relaxed);
        temp_path = directory_ + "/" + kTempPrefix + std::to_string(::getpid()) +
            "-" + std::to_string(ordinal) + kTempSuffix;
        fd = ::open(temp_path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0644);
        if (fd >= 0) break;
        if (errno != EEXIST) break;
        fd = -1;
    }
    do
    {
        if (fd < 0) break;
        if (!writeAll(fd, std::as_bytes(std::span(serialized)))) break;
        if (::fsync(fd) != 0) break;
        if (::close(fd) != 0) { fd = -1; break; }
        fd = -1;
        if (::rename(temp_path.c_str(), final_path.c_str()) != 0) break;
        // Make the rename durable (the directory entry, not the data): a
        // dir-fsync failure REFUSES publication (never a voided durability
        // claim — the operation reports failure and cleans up).
        const int dir_fd = ::open(directory_.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
        if (dir_fd < 0) break;
        if (::fsync(dir_fd) != 0) { (void)::close(dir_fd); break; }
        (void)::close(dir_fd);
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
    const auto path = pathFor(key);
    std::error_code ec;
    if (!std::filesystem::exists(path, ec) || ec)
    {
        // H2: an admitted-but-missing extent has NO storage to hold the
        // admission; the defined healing is to release the quota and report
        // success (the extent is gone, its charge must not stick until
        // restart). A second retire then no-ops on the withdrawn identity.
        quota_.retire({key.generation, key.namespace_id, key.digest, key.bytes});
        return true;
    }
    const auto removed = std::filesystem::remove(path, ec);
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
        // Engine-only namespace rule: ONLY the exact engine temp shape is
        // ever considered; named extents and foreign files are never touched.
        const auto owner = parseTempOwner(name);
        if (!owner.has_value()) continue;
        // OWNERSHIP PROOF: a temp owned by a LIVE process is a concurrent
        // in-flight operation of another job/thread — never deleted (an
        // orphan is only ever a temp whose owner pid is provably gone).
        if (!pidIsGone(*owner)) continue;
        if (!entry.is_regular_file(ec)) continue;
        if (std::filesystem::remove(entry.path(), ec) && !ec) ++removed;
    }
    return removed;
}
} // namespace llaminar2
