/**
 * @file DiskPrefixStorageBackend.cpp
 * @brief Crash-tolerant implementation of the model-specific prefix archive.
 *
 * The format is deliberately simple:
 *
 *   archive header
 *   [record preamble][JSON metadata][raw payload sections][commit footer]
 *   [record preamble][JSON metadata][raw payload sections][commit footer]
 *
 * Metadata is parsed with nlohmann::json rather than ad-hoc string searching.
 * A record is visible only when its footer is present and its metadata
 * checksum matches. Payload sections are verified lazily during hydration.
 */

#include "execution/prefix_cache/DiskPrefixStorageBackend.h"

#include "execution/prefix_cache/BlockHash.h"
#include "execution/prefix_cache/RamPrefixStorageBackend.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstring>
#include <exception>
#include <fcntl.h>
#include <limits>
#include <sstream>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

namespace llaminar2
{
    namespace
    {
        using json = nlohmann::json;

        constexpr std::array<char, 8> kArchiveMagic = {'L', 'L', 'K', 'V', 'C', '0', '0', '1'};
        constexpr std::array<char, 8> kRecordMagic = {'L', 'L', 'K', 'V', 'R', '0', '0', '1'};
        constexpr std::array<char, 8> kCommitMagic = {'L', 'L', 'K', 'V', 'D', 'O', 'N', 'E'};
        constexpr uint32_t kArchiveVersion = 1;
        constexpr uint32_t kRecordVersion = 1;
        constexpr uint32_t kPutRecord = 1;
        constexpr uint32_t kDeleteRecord = 2;
        constexpr uint32_t kTouchRecord = 3;
        constexpr uint64_t kArchiveHeaderBytes = 80;
        constexpr uint64_t kRecordPreambleBytes = 56;
        constexpr uint64_t kRecordFooterBytes = 16;
        constexpr uint64_t kMaximumMetadataBytes = 1024ull * 1024ull;
        constexpr uint64_t kMaximumRecordBytes = 1ull << 40;
        constexpr std::array<const char *, 6> kSectionNames = {
            "kv",
            "hybrid",
            "mtp",
            "terminal_hidden",
            "terminal_logits",
            "model_runtime_state",
        };

        class FileDescriptor
        {
        public:
            explicit FileDescriptor(int value = -1) : value_(value) {}
            ~FileDescriptor()
            {
                if (value_ >= 0)
                    ::close(value_);
            }
            FileDescriptor(const FileDescriptor &) = delete;
            FileDescriptor &operator=(const FileDescriptor &) = delete;
            FileDescriptor(FileDescriptor &&other) noexcept : value_(other.value_)
            {
                other.value_ = -1;
            }
            int get() const { return value_; }
            explicit operator bool() const { return value_ >= 0; }

        private:
            int value_ = -1;
        };

        class AdvisoryLock
        {
        public:
            explicit AdvisoryLock(const std::filesystem::path &path)
                : fd_(::open(path.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0600))
            {
                if (fd_ && ::flock(fd_.get(), LOCK_EX) != 0)
                    locked_ = false;
                else
                    locked_ = static_cast<bool>(fd_);
            }
            ~AdvisoryLock()
            {
                if (locked_)
                    (void)::flock(fd_.get(), LOCK_UN);
            }
            bool locked() const { return locked_; }

        private:
            FileDescriptor fd_;
            bool locked_ = false;
        };

        void storeU32(uint8_t *destination, uint32_t value)
        {
            for (int byte = 0; byte < 4; ++byte)
                destination[byte] = static_cast<uint8_t>((value >> (byte * 8)) & 0xffu);
        }

        void storeU64(uint8_t *destination, uint64_t value)
        {
            for (int byte = 0; byte < 8; ++byte)
                destination[byte] = static_cast<uint8_t>((value >> (byte * 8)) & 0xffu);
        }

        uint32_t loadU32(const uint8_t *source)
        {
            uint32_t value = 0;
            for (int byte = 0; byte < 4; ++byte)
                value |= static_cast<uint32_t>(source[byte]) << (byte * 8);
            return value;
        }

        uint64_t loadU64(const uint8_t *source)
        {
            uint64_t value = 0;
            for (int byte = 0; byte < 8; ++byte)
                value |= static_cast<uint64_t>(source[byte]) << (byte * 8);
            return value;
        }

        bool writeAll(int fd, const void *bytes, size_t byte_count)
        {
            const auto *cursor = static_cast<const uint8_t *>(bytes);
            size_t remaining = byte_count;
            while (remaining > 0)
            {
                const ssize_t written = ::write(fd, cursor, remaining);
                if (written < 0 && errno == EINTR)
                    continue;
                if (written <= 0)
                    return false;
                cursor += static_cast<size_t>(written);
                remaining -= static_cast<size_t>(written);
            }
            return true;
        }

        bool preadAll(int fd, void *bytes, size_t byte_count, uint64_t offset)
        {
            auto *cursor = static_cast<uint8_t *>(bytes);
            size_t remaining = byte_count;
            while (remaining > 0)
            {
                const ssize_t read_bytes = ::pread(
                    fd,
                    cursor,
                    remaining,
                    static_cast<off_t>(offset));
                if (read_bytes < 0 && errno == EINTR)
                    continue;
                if (read_bytes <= 0)
                    return false;
                cursor += static_cast<size_t>(read_bytes);
                offset += static_cast<uint64_t>(read_bytes);
                remaining -= static_cast<size_t>(read_bytes);
            }
            return true;
        }

        uint64_t fileSize(int fd)
        {
            struct stat state
            {
            };
            return ::fstat(fd, &state) == 0
                       ? static_cast<uint64_t>(state.st_size)
                       : 0;
        }

        bool isLowerHexDigest(const std::string &value)
        {
            if (value.size() != 64)
                return false;
            return std::all_of(
                value.begin(),
                value.end(),
                [](char character)
                {
                    return (character >= '0' && character <= '9') ||
                           (character >= 'a' && character <= 'f');
                });
        }

        uint64_t checksumBytes(const void *bytes, size_t byte_count)
        {
            return byte_count == 0
                       ? hashPrefixBytes("", 0)
                       : hashPrefixBytes(bytes, byte_count);
        }

        uint64_t recordCommitChecksum(
            uint32_t type,
            uint64_t sequence,
            uint64_t metadata_checksum,
            const json &sections)
        {
            uint64_t checksum = combinePrefixHash(metadata_checksum, type);
            checksum = combinePrefixHash(checksum, sequence);
            for (const auto &section : sections)
            {
                checksum = combinePrefixHash(
                    checksum,
                    section.value("bytes", uint64_t{0}));
                checksum = combinePrefixHash(
                    checksum,
                    section.value("checksum", uint64_t{0}));
            }
            return checksum;
        }

        json keyToJson(const PrefixCacheKey &key)
        {
            return {
                {"fingerprint", key.fingerprint},
                {"parent_hash", key.parent_hash},
                {"token_hash", key.token_hash},
                {"block_index", key.block_index},
                {"token_start", key.token_start},
                {"token_count", key.token_count},
            };
        }

        bool keyFromJson(const json &value, PrefixCacheKey *key)
        {
            if (!key || !value.is_object())
                return false;
            try
            {
                key->fingerprint = value.at("fingerprint").get<uint64_t>();
                key->parent_hash = value.at("parent_hash").get<uint64_t>();
                key->token_hash = value.at("token_hash").get<uint64_t>();
                key->block_index = value.at("block_index").get<int>();
                key->token_start = value.at("token_start").get<int>();
                key->token_count = value.at("token_count").get<int>();
                return key->valid();
            }
            catch (const json::exception &)
            {
                return false;
            }
        }

        json layoutToJson(const PrefixPayloadLayout &layout)
        {
            return {
                {"device_type", static_cast<int>(layout.device.type)},
                {"device_ordinal", layout.device.ordinal},
                {"block_size", layout.block_size},
                {"first_layer_index", layout.first_layer_index},
                {"total_layers", layout.total_layers},
                {"fa_layers", layout.fa_layers},
                {"gdn_layers", layout.gdn_layers},
                {"local_kv_heads", layout.local_kv_heads},
                {"kv_head_start", layout.kv_head_start},
                {"head_dim", layout.head_dim},
                {"k_precision", static_cast<int>(layout.k_precision)},
                {"v_precision", static_cast<int>(layout.v_precision)},
                {"kv_layout", static_cast<int>(layout.kv_layout)},
                {"bytes_per_fa_layer_k", layout.bytes_per_fa_layer_k},
                {"bytes_per_fa_layer_v", layout.bytes_per_fa_layer_v},
                {"hybrid_host_state_bytes", layout.hybrid_host_state_bytes},
                {"hybrid_device_state_bytes", layout.hybrid_device_state_bytes},
                {"hybrid_state_bytes", layout.hybrid_state_bytes},
                {"mtp_layers", layout.mtp_layers},
                {"mtp_local_kv_heads", layout.mtp_local_kv_heads},
                {"mtp_kv_head_start", layout.mtp_kv_head_start},
                {"mtp_head_dim", layout.mtp_head_dim},
                {"mtp_k_precision", static_cast<int>(layout.mtp_k_precision)},
                {"mtp_v_precision", static_cast<int>(layout.mtp_v_precision)},
                {"mtp_kv_layout", static_cast<int>(layout.mtp_kv_layout)},
                {"bytes_per_mtp_layer_k", layout.bytes_per_mtp_layer_k},
                {"bytes_per_mtp_layer_v", layout.bytes_per_mtp_layer_v},
                {"mtp_kv_bytes", layout.mtp_kv_bytes},
                {"terminal_hidden_bytes", layout.terminal_hidden_bytes},
                {"terminal_logits_bytes", layout.terminal_logits_bytes},
                {"includes_hybrid_state", layout.includes_hybrid_state},
                {"includes_mtp_state", layout.includes_mtp_state},
                {"includes_terminal_hidden", layout.includes_terminal_hidden},
                {"includes_terminal_logits", layout.includes_terminal_logits},
            };
        }

        bool layoutFromJson(const json &value, PrefixPayloadLayout *layout)
        {
            if (!layout || !value.is_object())
                return false;
            try
            {
                layout->device = DeviceId(
                    static_cast<DeviceType>(value.at("device_type").get<int>()),
                    value.at("device_ordinal").get<int>());
                layout->block_size = value.at("block_size").get<int>();
                layout->first_layer_index = value.at("first_layer_index").get<int>();
                layout->total_layers = value.at("total_layers").get<int>();
                layout->fa_layers = value.at("fa_layers").get<int>();
                layout->gdn_layers = value.at("gdn_layers").get<int>();
                layout->local_kv_heads = value.at("local_kv_heads").get<int>();
                layout->kv_head_start = value.at("kv_head_start").get<int>();
                layout->head_dim = value.at("head_dim").get<int>();
                layout->k_precision =
                    static_cast<ActivationPrecision>(value.at("k_precision").get<int>());
                layout->v_precision =
                    static_cast<ActivationPrecision>(value.at("v_precision").get<int>());
                layout->kv_layout =
                    static_cast<TensorLayout>(value.at("kv_layout").get<int>());
                layout->bytes_per_fa_layer_k =
                    value.at("bytes_per_fa_layer_k").get<size_t>();
                layout->bytes_per_fa_layer_v =
                    value.at("bytes_per_fa_layer_v").get<size_t>();
                layout->hybrid_host_state_bytes =
                    value.at("hybrid_host_state_bytes").get<size_t>();
                layout->hybrid_device_state_bytes =
                    value.at("hybrid_device_state_bytes").get<size_t>();
                layout->hybrid_state_bytes =
                    value.at("hybrid_state_bytes").get<size_t>();
                layout->mtp_layers = value.at("mtp_layers").get<int>();
                layout->mtp_local_kv_heads =
                    value.at("mtp_local_kv_heads").get<int>();
                layout->mtp_kv_head_start =
                    value.at("mtp_kv_head_start").get<int>();
                layout->mtp_head_dim = value.at("mtp_head_dim").get<int>();
                layout->mtp_k_precision =
                    static_cast<ActivationPrecision>(value.at("mtp_k_precision").get<int>());
                layout->mtp_v_precision =
                    static_cast<ActivationPrecision>(value.at("mtp_v_precision").get<int>());
                layout->mtp_kv_layout =
                    static_cast<TensorLayout>(value.at("mtp_kv_layout").get<int>());
                layout->bytes_per_mtp_layer_k =
                    value.at("bytes_per_mtp_layer_k").get<size_t>();
                layout->bytes_per_mtp_layer_v =
                    value.at("bytes_per_mtp_layer_v").get<size_t>();
                layout->mtp_kv_bytes = value.at("mtp_kv_bytes").get<size_t>();
                layout->terminal_hidden_bytes =
                    value.at("terminal_hidden_bytes").get<size_t>();
                layout->terminal_logits_bytes =
                    value.at("terminal_logits_bytes").get<size_t>();
                layout->includes_hybrid_state =
                    value.at("includes_hybrid_state").get<bool>();
                layout->includes_mtp_state =
                    value.at("includes_mtp_state").get<bool>();
                layout->includes_terminal_hidden =
                    value.at("includes_terminal_hidden").get<bool>();
                layout->includes_terminal_logits =
                    value.at("includes_terminal_logits").get<bool>();
                return layout->device.is_valid() &&
                       layout->block_size > 0 &&
                       layout->totalBytes() > 0;
            }
            catch (const json::exception &)
            {
                return false;
            }
        }

        std::array<std::pair<const void *, size_t>, 6> payloadSections(
            const PrefixBlockHandle &handle)
        {
            const auto runtime_bytes =
                handle.model_runtime_state_storage
                    ? handle.model_runtime_state_storage->size()
                    : 0u;
            return {{
                {handle.kv_payload, handle.kvBytes()},
                {handle.hybrid_payload, handle.hybridBytes()},
                {handle.mtp_payload, handle.layout.mtpKVBytes()},
                {handle.terminal_hidden, handle.terminalHiddenBytes()},
                {handle.terminal_logits, handle.terminalLogitsBytes()},
                {runtime_bytes > 0 ? handle.model_runtime_state_storage->data() : nullptr,
                 runtime_bytes},
            }};
        }

        bool fullLayoutMatch(
            const PrefixPayloadLayout &lhs,
            const PrefixPayloadLayout &rhs)
        {
            return lhs.compatiblePayloadShape(rhs) &&
                   lhs.includes_hybrid_state == rhs.includes_hybrid_state &&
                   lhs.includes_terminal_hidden == rhs.includes_terminal_hidden &&
                   lhs.includes_terminal_logits == rhs.includes_terminal_logits &&
                   lhs.totalBytes() == rhs.totalBytes();
        }

        bool runtimeLayoutMatch(
            const PrefixPayloadLayout &record,
            const PrefixPayloadLayout &runtime)
        {
            /*
             * Non-terminal blocks deliberately omit hybrid and terminal
             * payloads, but retain the same byte geometry in their layout.
             * compatiblePayloadShape() captures that relationship while still
             * requiring every KV/MTP geometry and precision field to match.
             */
            return record.compatiblePayloadShape(runtime);
        }

        std::string errnoMessage(const std::string &operation)
        {
            return operation + ": " + std::strerror(errno);
        }
    } // namespace

    DiskPrefixStorageBackend::HydrationTicket::~HydrationTicket()
    {
        release();
    }

    DiskPrefixStorageBackend::HydrationTicket::HydrationTicket(
        HydrationTicket &&other) noexcept
        : backend_(std::move(other.backend_)),
          handle_(std::move(other.handle_)),
          archive_device_(other.archive_device_),
          archive_inode_(other.archive_inode_),
          sections_(other.sections_),
          consumed_(other.consumed_)
    {
        other.archive_device_ = 0;
        other.archive_inode_ = 0;
        other.consumed_ = true;
    }

    DiskPrefixStorageBackend::HydrationTicket &
    DiskPrefixStorageBackend::HydrationTicket::operator=(
        HydrationTicket &&other) noexcept
    {
        if (this == &other)
            return *this;
        release();
        backend_ = std::move(other.backend_);
        handle_ = std::move(other.handle_);
        archive_device_ = other.archive_device_;
        archive_inode_ = other.archive_inode_;
        sections_ = other.sections_;
        consumed_ = other.consumed_;
        other.archive_device_ = 0;
        other.archive_inode_ = 0;
        other.consumed_ = true;
        return *this;
    }

    DiskPrefixStorageBackend::HydrationTicket::HydrationTicket(
        std::shared_ptr<DiskPrefixStorageBackend> backend,
        PrefixBlockHandle handle,
        uint64_t archive_device,
        uint64_t archive_inode,
        std::array<SectionSnapshot, 6> sections)
        : backend_(std::move(backend)),
          handle_(std::move(handle)),
          archive_device_(archive_device),
          archive_inode_(archive_inode),
          sections_(std::move(sections))
    {
    }

    bool DiskPrefixStorageBackend::HydrationTicket::valid() const noexcept
    {
        return backend_ && handle_.valid() && !consumed_;
    }

    size_t DiskPrefixStorageBackend::HydrationTicket::totalBytes() const noexcept
    {
        return handle_.total_bytes;
    }

    const PrefixBlockHandle &
    DiskPrefixStorageBackend::HydrationTicket::diskHandle() const noexcept
    {
        return handle_;
    }

    void DiskPrefixStorageBackend::HydrationTicket::release() noexcept
    {
        if (!backend_)
            return;
        backend_->releaseHydrationTicket();
        backend_.reset();
        consumed_ = true;
    }

    DiskPrefixStorageBackend::DiskPrefixStorageBackend(
        std::filesystem::path archive_path,
        size_t budget_bytes,
        std::string model_artifact_identity)
        : DiskPrefixStorageBackend(
              std::move(archive_path),
              budget_bytes,
              std::move(model_artifact_identity),
              nullptr)
    {
    }

    DiskPrefixStorageBackend::DiskPrefixStorageBackend(
        std::filesystem::path archive_path,
        size_t budget_bytes,
        std::string model_artifact_identity,
        std::shared_ptr<PhysicalMemoryAuthority> memory_authority)
        : archive_path_(std::move(archive_path)),
          lock_path_(archive_path_.string() + ".lock"),
          budget_bytes_(budget_bytes),
          model_artifact_identity_(std::move(model_artifact_identity)),
          memory_authority_(std::move(memory_authority))
    {
        /*
         * Claim before allocating.  Direct construction deliberately has no
         * live production topology and remains available only to focused unit
         * tests; openShared() hard-requires and retains the canonical ledger.
         */
        if (memory_authority_)
        {
            archive_scratch_memory_lease_ =
                memory_authority_->claimNewAllocation(
                    DeviceId::cpu(),
                    PhysicalMemoryOwner::PrefixArchiveStaging,
                    PrefixArchiveIOGeometry::scratchBytes());
        }
        archive_scratch_.resize(PrefixArchiveIOGeometry::scratchBytes());

        std::string error;
        ready_ = initialize(&error);
        initialization_error_ = std::move(error);
    }

    std::shared_ptr<DiskPrefixStorageBackend> DiskPrefixStorageBackend::openShared(
        const std::filesystem::path &archive_path,
        size_t budget_bytes,
        const std::string &model_artifact_identity,
        std::shared_ptr<PhysicalMemoryAuthority> memory_authority,
        std::string *error)
    {
        static std::mutex registry_mutex;
        static std::unordered_map<std::string, std::weak_ptr<DiskPrefixStorageBackend>> registry;

        std::error_code path_error;
        const auto absolute = std::filesystem::absolute(archive_path, path_error);
        const std::string key =
            (path_error ? archive_path.lexically_normal() : absolute.lexically_normal()).string();

        std::lock_guard<std::mutex> lock(registry_mutex);
        if (!memory_authority ||
            !memory_authority->contains(DeviceId::cpu()))
        {
            if (error)
                *error = "shared prefix archive requires the rank-local CPU memory authority";
            return nullptr;
        }
        if (auto existing = registry[key].lock())
        {
            if (existing->budgetBytes() != budget_bytes)
            {
                if (error)
                    *error = "archive already opened with a different disk budget";
                return nullptr;
            }
            if (existing->memory_authority_.get() != memory_authority.get())
            {
                if (error)
                    *error = "archive already belongs to a different physical-memory authority";
                return nullptr;
            }
            return existing;
        }

        std::shared_ptr<DiskPrefixStorageBackend> backend;
        try
        {
            backend.reset(new DiskPrefixStorageBackend(
                archive_path,
                budget_bytes,
                model_artifact_identity,
                std::move(memory_authority)));
        }
        catch (const std::exception &exception)
        {
            if (error)
                *error = exception.what();
            return nullptr;
        }
        if (!backend->ready())
        {
            if (error)
                *error = backend->initializationError();
            return nullptr;
        }
        registry[key] = backend;
        return backend;
    }

    bool DiskPrefixStorageBackend::canStore(size_t bytes) const
    {
        return bytes > 0 && (budget_bytes_ == 0 || bytes <= budget_bytes_);
    }

    PrefixBlockHandle DiskPrefixStorageBackend::allocate(
        const PrefixCacheKey &key,
        const PrefixPayloadLayout &layout)
    {
        PrefixBlockHandle handle;
        handle.key = key;
        handle.tier = PrefixStorageTier::Disk;
        handle.layout = layout;
        handle.total_bytes = layout.totalBytes();
        return canStore(handle.total_bytes) ? handle : PrefixBlockHandle{};
    }

    bool DiskPrefixStorageBackend::ready() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return ready_;
    }

    const std::string &DiskPrefixStorageBackend::initializationError() const
    {
        return initialization_error_;
    }

    size_t DiskPrefixStorageBackend::usedBytes() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return static_cast<size_t>(active_bytes_);
    }

    bool DiskPrefixStorageBackend::initialize(std::string *error)
    {
        if (!isLowerHexDigest(model_artifact_identity_))
        {
            if (error)
                *error = "model artifact identity must be 64 lowercase hexadecimal characters";
            return false;
        }
        if (archive_path_.filename() !=
            model_artifact_identity_ + ".kvcache")
        {
            if (error)
                *error = "prefix archive filename must be <model-artifact-identity>.kvcache";
            return false;
        }

        std::error_code filesystem_error;
        std::filesystem::create_directories(
            archive_path_.parent_path(),
            filesystem_error);
        if (filesystem_error)
        {
            if (error)
                *error = filesystem_error.message();
            return false;
        }
        std::filesystem::permissions(
            archive_path_.parent_path(),
            std::filesystem::perms::owner_all,
            std::filesystem::perm_options::replace,
            filesystem_error);
        if (filesystem_error)
        {
            if (error)
                *error = filesystem_error.message();
            return false;
        }

        AdvisoryLock lock(lock_path_);
        if (!lock.locked())
        {
            if (error)
                *error = errnoMessage("failed to lock prefix archive");
            return false;
        }

        FileDescriptor archive(::open(
            archive_path_.c_str(),
            O_RDWR | O_CREAT | O_CLOEXEC,
            0600));
        if (!archive)
        {
            if (error)
                *error = errnoMessage("failed to open prefix archive");
            return false;
        }
        (void)::fchmod(archive.get(), 0600);

        if (fileSize(archive.get()) == 0)
        {
            std::array<uint8_t, kArchiveHeaderBytes> header{};
            std::memcpy(header.data(), kArchiveMagic.data(), kArchiveMagic.size());
            storeU32(header.data() + 8, kArchiveVersion);
            storeU32(header.data() + 12, static_cast<uint32_t>(kArchiveHeaderBytes));
            std::memcpy(
                header.data() + 16,
                model_artifact_identity_.data(),
                model_artifact_identity_.size());
            if (!writeAll(archive.get(), header.data(), header.size()) ||
                ::fsync(archive.get()) != 0)
            {
                if (error)
                    *error = errnoMessage("failed to initialize prefix archive");
                return false;
            }
        }

        return refreshIndexLocked(archive.get(), error);
    }

    bool DiskPrefixStorageBackend::refreshIndexLocked(
        int archive_fd,
        std::string *error)
    {
        struct stat state
        {
        };
        if (::fstat(archive_fd, &state) != 0)
        {
            if (error)
                *error = errnoMessage("failed to stat prefix archive");
            return false;
        }
        const uint64_t bytes = static_cast<uint64_t>(state.st_size);
        if (bytes < kArchiveHeaderBytes)
        {
            if (error)
                *error = "prefix archive header is truncated";
            return false;
        }

        std::array<uint8_t, kArchiveHeaderBytes> header{};
        if (!preadAll(archive_fd, header.data(), header.size(), 0) ||
            !std::equal(kArchiveMagic.begin(), kArchiveMagic.end(), header.begin()) ||
            loadU32(header.data() + 8) != kArchiveVersion ||
            loadU32(header.data() + 12) != kArchiveHeaderBytes ||
            std::string(
                reinterpret_cast<const char *>(header.data() + 16),
                model_artifact_identity_.size()) !=
                model_artifact_identity_)
        {
            if (error)
                *error = "prefix archive header or model artifact identity mismatch";
            return false;
        }

        const bool archive_replaced =
            archive_inode_ != 0 &&
            (archive_device_ != static_cast<uint64_t>(state.st_dev) ||
             archive_inode_ != static_cast<uint64_t>(state.st_ino));
        if (archive_replaced || scan_offset_ < kArchiveHeaderBytes || scan_offset_ > bytes)
        {
            records_.clear();
            active_bytes_ = 0;
            next_sequence_ = 1;
            scan_offset_ = kArchiveHeaderBytes;
        }
        archive_device_ = static_cast<uint64_t>(state.st_dev);
        archive_inode_ = static_cast<uint64_t>(state.st_ino);

        uint64_t valid_end = scan_offset_;
        if (!scanRecordsLocked(
                archive_fd,
                scan_offset_,
                bytes,
                &valid_end,
                error))
        {
            return false;
        }

        /*
         * A missing footer identifies an interrupted final append. It is safe
         * to truncate only that uncommitted tail while retaining every earlier
         * committed record.
         */
        if (valid_end < bytes && ::ftruncate(archive_fd, static_cast<off_t>(valid_end)) != 0)
        {
            if (error)
                *error = errnoMessage("failed to discard incomplete archive tail");
            return false;
        }
        scan_offset_ = valid_end;
        return true;
    }

    bool DiskPrefixStorageBackend::scanRecordsLocked(
        int archive_fd,
        uint64_t start_offset,
        uint64_t file_bytes,
        uint64_t *valid_end,
        std::string *error)
    {
        uint64_t cursor = start_offset;
        while (cursor + kRecordPreambleBytes <= file_bytes)
        {
            std::array<uint8_t, kRecordPreambleBytes> preamble{};
            if (!preadAll(archive_fd, preamble.data(), preamble.size(), cursor))
                break;
            if (!std::equal(kRecordMagic.begin(), kRecordMagic.end(), preamble.begin()))
                break;

            const uint32_t version = loadU32(preamble.data() + 8);
            const uint32_t type = loadU32(preamble.data() + 12);
            const uint64_t record_bytes = loadU64(preamble.data() + 16);
            const uint64_t metadata_bytes = loadU64(preamble.data() + 24);
            const uint64_t payload_bytes = loadU64(preamble.data() + 32);
            const uint64_t sequence = loadU64(preamble.data() + 40);
            const uint64_t metadata_checksum = loadU64(preamble.data() + 48);

            if (version != kRecordVersion ||
                (type != kPutRecord &&
                 type != kDeleteRecord &&
                 type != kTouchRecord) ||
                metadata_bytes == 0 ||
                metadata_bytes > kMaximumMetadataBytes ||
                record_bytes > kMaximumRecordBytes ||
                record_bytes != kRecordPreambleBytes + metadata_bytes +
                                    payload_bytes + kRecordFooterBytes ||
                cursor > std::numeric_limits<uint64_t>::max() - record_bytes ||
                cursor + record_bytes > file_bytes)
            {
                break;
            }

            std::string metadata(metadata_bytes, '\0');
            if (!preadAll(
                    archive_fd,
                    metadata.data(),
                    metadata.size(),
                    cursor + kRecordPreambleBytes) ||
                checksumBytes(metadata.data(), metadata.size()) != metadata_checksum)
            {
                break;
            }

            json parsed = json::parse(metadata, nullptr, false);
            if (parsed.is_discarded())
                break;
            PrefixCacheKey key;
            if (!keyFromJson(parsed.value("key", json::object()), &key))
                break;

            std::array<uint8_t, kRecordFooterBytes> footer{};
            const uint64_t footer_offset = cursor + record_bytes - kRecordFooterBytes;
            if (!preadAll(archive_fd, footer.data(), footer.size(), footer_offset) ||
                !std::equal(kCommitMagic.begin(), kCommitMagic.end(), footer.begin()) ||
                loadU64(footer.data() + 8) != recordCommitChecksum(
                    type,
                    sequence,
                    metadata_checksum,
                    parsed.value("sections", json::array())))
            {
                break;
            }

            if (type == kDeleteRecord)
            {
                applyDeleteRecord(key);
            }
            else if (type == kTouchRecord)
            {
                if (payload_bytes != 0 ||
                    !applyTouchRecord(key, sequence))
                {
                    break;
                }
            }
            else
            {
                PrefixPayloadLayout layout;
                const json sections = parsed.value("sections", json::array());
                if (!layoutFromJson(parsed.value("layout", json::object()), &layout) ||
                    !sections.is_array() ||
                    sections.size() != kSectionNames.size())
                {
                    break;
                }

                RecordIndex record;
                record.handle.key = key;
                record.handle.tier = PrefixStorageTier::Disk;
                record.handle.layout = layout;
                record.handle.has_hybrid_state =
                    parsed.value("has_hybrid_state", false);
                record.handle.has_terminal_hidden =
                    parsed.value("has_terminal_hidden", false);
                record.handle.has_terminal_logits =
                    parsed.value("has_terminal_logits", false);
                record.handle.has_model_runtime_state =
                    parsed.value("has_model_runtime_state", false);
                record.record_offset = cursor;
                record.record_bytes = record_bytes;
                record.sequence = sequence;

                uint64_t section_offset =
                    cursor + kRecordPreambleBytes + metadata_bytes;
                uint64_t section_total = 0;
                bool sections_valid = true;
                for (size_t index = 0; index < sections.size(); ++index)
                {
                    if (sections[index].value("name", std::string{}) !=
                        kSectionNames[index])
                    {
                        sections_valid = false;
                        break;
                    }
                    const uint64_t section_bytes =
                        sections[index].value("bytes", uint64_t{0});
                    record.sections[index] = {
                        section_offset,
                        section_bytes,
                        sections[index].value("checksum", uint64_t{0}),
                    };
                    section_offset += section_bytes;
                    section_total += section_bytes;
                }
                if (!sections_valid ||
                    section_total != payload_bytes ||
                    section_total != parsed.value("total_bytes", uint64_t{0}) ||
                    section_total < layout.totalBytes())
                {
                    break;
                }
                record.handle.total_bytes = static_cast<size_t>(section_total);
                applyPutRecord(std::move(record));
            }

            next_sequence_ = std::max(next_sequence_, sequence + 1);
            cursor += record_bytes;
        }

        if (valid_end)
            *valid_end = cursor;
        if (cursor < file_bytes && error)
            *error = "discarded an incomplete or malformed archive tail";
        return true;
    }

    void DiskPrefixStorageBackend::applyPutRecord(RecordIndex record)
    {
        auto existing = records_.find(record.handle.key);
        if (existing != records_.end())
        {
            active_bytes_ -= std::min<uint64_t>(
                active_bytes_,
                existing->second.handle.total_bytes);
        }
        active_bytes_ += record.handle.total_bytes;
        records_[record.handle.key] = std::move(record);
    }

    void DiskPrefixStorageBackend::applyDeleteRecord(const PrefixCacheKey &key)
    {
        auto existing = records_.find(key);
        if (existing == records_.end())
            return;
        active_bytes_ -= std::min<uint64_t>(
            active_bytes_,
            existing->second.handle.total_bytes);
        records_.erase(existing);
    }

    bool DiskPrefixStorageBackend::applyTouchRecord(
        const PrefixCacheKey &key,
        uint64_t sequence)
    {
        auto existing = records_.find(key);
        if (existing == records_.end())
            return false;
        existing->second.sequence = sequence;
        return true;
    }

    bool DiskPrefixStorageBackend::appendDeleteLocked(
        int archive_fd,
        const PrefixCacheKey &key,
        std::string *error)
    {
        const uint64_t sequence = next_sequence_++;
        const json metadata_json = {
            {"type", "delete"},
            {"key", keyToJson(key)},
            {"sections", json::array()},
        };
        const std::string metadata = metadata_json.dump();
        const uint64_t metadata_checksum =
            checksumBytes(metadata.data(), metadata.size());
        const uint64_t record_bytes =
            kRecordPreambleBytes + metadata.size() + kRecordFooterBytes;

        std::array<uint8_t, kRecordPreambleBytes> preamble{};
        std::memcpy(preamble.data(), kRecordMagic.data(), kRecordMagic.size());
        storeU32(preamble.data() + 8, kRecordVersion);
        storeU32(preamble.data() + 12, kDeleteRecord);
        storeU64(preamble.data() + 16, record_bytes);
        storeU64(preamble.data() + 24, metadata.size());
        storeU64(preamble.data() + 32, 0);
        storeU64(preamble.data() + 40, sequence);
        storeU64(preamble.data() + 48, metadata_checksum);

        std::array<uint8_t, kRecordFooterBytes> footer{};
        std::memcpy(footer.data(), kCommitMagic.data(), kCommitMagic.size());
        storeU64(
            footer.data() + 8,
            recordCommitChecksum(
                kDeleteRecord,
                sequence,
                metadata_checksum,
                metadata_json["sections"]));

        if (::lseek(archive_fd, 0, SEEK_END) < 0 ||
            !writeAll(archive_fd, preamble.data(), preamble.size()) ||
            !writeAll(archive_fd, metadata.data(), metadata.size()) ||
            !writeAll(archive_fd, footer.data(), footer.size()))
        {
            if (error)
                *error = errnoMessage("failed to append archive tombstone");
            return false;
        }
        applyDeleteRecord(key);
        scan_offset_ += record_bytes;
        return true;
    }

    bool DiskPrefixStorageBackend::appendTouchLocked(
        int archive_fd,
        const PrefixCacheKey &key,
        std::string *error)
    {
        if (records_.find(key) == records_.end())
        {
            if (error)
                *error = "cannot touch a missing prefix archive record";
            return false;
        }

        const uint64_t sequence = next_sequence_++;
        const json metadata_json = {
            {"type", "touch"},
            {"key", keyToJson(key)},
            {"sections", json::array()},
        };
        const std::string metadata = metadata_json.dump();
        const uint64_t metadata_checksum =
            checksumBytes(metadata.data(), metadata.size());
        const uint64_t record_bytes =
            kRecordPreambleBytes + metadata.size() + kRecordFooterBytes;

        std::array<uint8_t, kRecordPreambleBytes> preamble{};
        std::memcpy(preamble.data(), kRecordMagic.data(), kRecordMagic.size());
        storeU32(preamble.data() + 8, kRecordVersion);
        storeU32(preamble.data() + 12, kTouchRecord);
        storeU64(preamble.data() + 16, record_bytes);
        storeU64(preamble.data() + 24, metadata.size());
        storeU64(preamble.data() + 32, 0);
        storeU64(preamble.data() + 40, sequence);
        storeU64(preamble.data() + 48, metadata_checksum);

        std::array<uint8_t, kRecordFooterBytes> footer{};
        std::memcpy(footer.data(), kCommitMagic.data(), kCommitMagic.size());
        storeU64(
            footer.data() + 8,
            recordCommitChecksum(
                kTouchRecord,
                sequence,
                metadata_checksum,
                metadata_json["sections"]));

        if (::lseek(archive_fd, 0, SEEK_END) < 0 ||
            !writeAll(archive_fd, preamble.data(), preamble.size()) ||
            !writeAll(archive_fd, metadata.data(), metadata.size()) ||
            !writeAll(archive_fd, footer.data(), footer.size()))
        {
            if (error)
                *error = errnoMessage("failed to append archive LRU touch");
            return false;
        }
        if (!applyTouchRecord(key, sequence))
        {
            if (error)
                *error = "archive LRU touch lost its active payload record";
            return false;
        }
        scan_offset_ += record_bytes;
        return true;
    }

    bool DiskPrefixStorageBackend::writeBlock(
        const PrefixBlockHandle &handle,
        PrefixBlockHandle *disk_handle,
        std::vector<PrefixCacheKey> *evicted_keys,
        std::string *error)
    {
        if (!handle.valid() || !canStore(handle.total_bytes))
        {
            if (error)
                *error = "invalid prefix block or block exceeds disk budget";
            return false;
        }
        if (!handle.waitForPayloadOnHost())
        {
            if (error)
                *error = "prefix payload is not ready for disk serialization";
            return false;
        }

        const auto sections = payloadSections(handle);
        uint64_t payload_bytes = 0;
        json section_metadata = json::array();
        for (size_t index = 0; index < sections.size(); ++index)
        {
            const auto &[pointer, bytes] = sections[index];
            if (bytes > 0 && !pointer)
            {
                if (error)
                    *error = std::string("missing payload for section ") +
                             kSectionNames[index];
                return false;
            }
            payload_bytes += bytes;
            section_metadata.push_back({
                {"name", kSectionNames[index]},
                {"bytes", bytes},
                {"checksum", checksumBytes(pointer, bytes)},
            });
        }
        if (payload_bytes != handle.total_bytes)
        {
            if (error)
                *error = "prefix block total does not equal serialized section bytes";
            return false;
        }

        std::lock_guard<std::mutex> process_lock(mutex_);
        if (!ready_)
        {
            if (error)
                *error = initialization_error_;
            return false;
        }
        AdvisoryLock file_lock(lock_path_);
        if (!file_lock.locked())
        {
            if (error)
                *error = errnoMessage("failed to lock prefix archive");
            return false;
        }
        FileDescriptor archive(::open(archive_path_.c_str(), O_RDWR | O_CLOEXEC));
        if (!archive || !refreshIndexLocked(archive.get(), error))
            return false;

        std::vector<PrefixCacheKey> evicted;
        auto same_key = records_.find(handle.key);
        const uint64_t replace_bytes =
            same_key == records_.end() ? 0 : same_key->second.handle.total_bytes;
        while (budget_bytes_ != 0 &&
               active_bytes_ - std::min(active_bytes_, replace_bytes) +
                       handle.total_bytes >
                   budget_bytes_)
        {
            auto victim = std::min_element(
                records_.begin(),
                records_.end(),
                [&](const auto &lhs, const auto &rhs)
                {
                    if (lhs.first == handle.key)
                        return false;
                    if (rhs.first == handle.key)
                        return true;
                    return lhs.second.sequence < rhs.second.sequence;
                });
            if (victim == records_.end() || victim->first == handle.key)
            {
                if (error)
                    *error = "disk budget cannot admit prefix block";
                return false;
            }
            const PrefixCacheKey victim_key = victim->first;
            if (!appendDeleteLocked(archive.get(), victim_key, error))
                return false;
            evicted.push_back(victim_key);
        }

        const uint64_t sequence = next_sequence_++;
        const json metadata_json = {
            {"type", "put"},
            {"key", keyToJson(handle.key)},
            {"layout", layoutToJson(handle.layout)},
            {"total_bytes", payload_bytes},
            {"has_hybrid_state", handle.has_hybrid_state},
            {"has_terminal_hidden", handle.has_terminal_hidden},
            {"has_terminal_logits", handle.has_terminal_logits},
            {"has_model_runtime_state", handle.has_model_runtime_state},
            {"sections", section_metadata},
        };
        const std::string metadata = metadata_json.dump();
        const uint64_t metadata_checksum =
            checksumBytes(metadata.data(), metadata.size());
        const uint64_t record_bytes =
            kRecordPreambleBytes + metadata.size() +
            payload_bytes + kRecordFooterBytes;

        std::array<uint8_t, kRecordPreambleBytes> preamble{};
        std::memcpy(preamble.data(), kRecordMagic.data(), kRecordMagic.size());
        storeU32(preamble.data() + 8, kRecordVersion);
        storeU32(preamble.data() + 12, kPutRecord);
        storeU64(preamble.data() + 16, record_bytes);
        storeU64(preamble.data() + 24, metadata.size());
        storeU64(preamble.data() + 32, payload_bytes);
        storeU64(preamble.data() + 40, sequence);
        storeU64(preamble.data() + 48, metadata_checksum);

        std::array<uint8_t, kRecordFooterBytes> footer{};
        std::memcpy(footer.data(), kCommitMagic.data(), kCommitMagic.size());
        storeU64(
            footer.data() + 8,
            recordCommitChecksum(
                kPutRecord,
                sequence,
                metadata_checksum,
                section_metadata));

        const off_t record_offset = ::lseek(archive.get(), 0, SEEK_END);
        if (record_offset < 0 ||
            !writeAll(archive.get(), preamble.data(), preamble.size()) ||
            !writeAll(archive.get(), metadata.data(), metadata.size()))
        {
            if (error)
                *error = errnoMessage("failed to append prefix record metadata");
            return false;
        }
        for (const auto &[pointer, bytes] : sections)
        {
            if (bytes > 0 && !writeAll(archive.get(), pointer, bytes))
            {
                if (error)
                    *error = errnoMessage("failed to append prefix payload");
                return false;
            }
        }
        if (!writeAll(archive.get(), footer.data(), footer.size()) ||
            ::fsync(archive.get()) != 0)
        {
            if (error)
                *error = errnoMessage("failed to commit prefix archive record");
            return false;
        }

        RecordIndex record;
        record.handle = handle;
        record.handle.tier = PrefixStorageTier::Disk;
        record.handle.kv_payload = nullptr;
        record.handle.hybrid_payload = nullptr;
        record.handle.mtp_payload = nullptr;
        record.handle.terminal_hidden = nullptr;
        record.handle.terminal_logits = nullptr;
        record.handle.kv_storage.reset();
        record.handle.hybrid_storage.reset();
        record.handle.mtp_storage.reset();
        record.handle.terminal_hidden_storage.reset();
        record.handle.terminal_logits_storage.reset();
        record.handle.model_runtime_state_storage.reset();
        record.handle.ram_payload_memory_lease.reset();
        record.handle.ram_runtime_state_memory_lease.reset();
        record.handle.payload_readiness.reset();
        record.handle.pinned_kv_storage.reset();
        record.handle.pinned_hybrid_storage.reset();
        record.handle.pinned_mtp_storage.reset();
        record.handle.pinned_terminal_hidden_storage.reset();
        record.handle.pinned_terminal_logits_storage.reset();
        record.handle.device_kv_storage.reset();
        record.handle.device_hybrid_storage.reset();
        record.handle.device_mtp_storage.reset();
        record.handle.device_terminal_hidden_storage.reset();
        record.handle.device_terminal_logits_storage.reset();
        record.handle.device_kv_allocation.reset();
        record.handle.device_hybrid_allocation.reset();
        record.handle.device_mtp_allocation.reset();
        record.handle.device_terminal_hidden_allocation.reset();
        record.handle.device_terminal_logits_allocation.reset();
        record.record_offset = static_cast<uint64_t>(record_offset);
        record.record_bytes = record_bytes;
        record.sequence = sequence;
        uint64_t section_offset =
            record.record_offset + kRecordPreambleBytes + metadata.size();
        for (size_t index = 0; index < sections.size(); ++index)
        {
            record.sections[index] = {
                section_offset,
                static_cast<uint64_t>(sections[index].second),
                section_metadata[index]["checksum"].get<uint64_t>(),
            };
            section_offset += sections[index].second;
        }
        applyPutRecord(std::move(record));
        scan_offset_ = static_cast<uint64_t>(record_offset) + record_bytes;

        if (disk_handle)
            *disk_handle = records_.at(handle.key).handle;
        if (evicted_keys)
            *evicted_keys = std::move(evicted);

        /*
         * Compaction runs only after the new record is durable and while the
         * cross-process lock is still held. It rewrites active records when
         * stale append history materially exceeds the configured capacity.
         */
        return compactIfNeededLocked(archive.get(), error);
    }

    bool DiskPrefixStorageBackend::release(const PrefixBlockHandle &handle)
    {
        std::lock_guard<std::mutex> process_lock(mutex_);
        if (!ready_)
            return false;
        AdvisoryLock file_lock(lock_path_);
        if (!file_lock.locked())
            return false;
        FileDescriptor archive(::open(archive_path_.c_str(), O_RDWR | O_CLOEXEC));
        std::string error;
        if (!archive || !refreshIndexLocked(archive.get(), &error))
            return false;
        if (records_.find(handle.key) == records_.end())
            return true;
        if (!appendDeleteLocked(archive.get(), handle.key, &error) ||
            ::fsync(archive.get()) != 0)
            return false;
        return compactIfNeededLocked(archive.get(), &error);
    }

    bool DiskPrefixStorageBackend::readBlock(
        const PrefixCacheKey &key,
        const PrefixPayloadLayout &layout,
        PrefixBlockHandle *ram_handle,
        std::string *error)
    {
        /*
         * This convenience path is only a unit-test oracle.  Production owns
         * a bounded, admitted RamPrefixStorageBackend and calls the typed
         * direct-hydration transaction after beginVerifiedHydration().
         */
        RamPrefixStorageBackend ram(std::numeric_limits<size_t>::max());
        return readBlockIntoRamBackend(
            key,
            layout,
            ram,
            ram_handle,
            error);
    }

    std::optional<DiskPrefixStorageBackend::HydrationTicket>
    DiskPrefixStorageBackend::beginVerifiedHydration(
        const PrefixCacheKey &key,
        const PrefixPayloadLayout &layout,
        std::string *error)
    {
        std::lock_guard<std::mutex> process_lock(mutex_);
        if (!ready_)
        {
            if (error)
                *error = initialization_error_;
            return std::nullopt;
        }
        AdvisoryLock file_lock(lock_path_);
        if (!file_lock.locked())
        {
            if (error)
                *error = errnoMessage("failed to lock prefix archive");
            return std::nullopt;
        }
        FileDescriptor archive(::open(archive_path_.c_str(), O_RDWR | O_CLOEXEC));
        if (!archive || !refreshIndexLocked(archive.get(), error))
            return std::nullopt;

        const auto record_it = records_.find(key);
        if (record_it == records_.end() ||
            !fullLayoutMatch(record_it->second.handle.layout, layout))
        {
            if (error)
                *error = "prefix archive record not found or layout mismatch";
            return std::nullopt;
        }
        if (!verifyRecordPayloadLocked(
                archive.get(), record_it->second, error))
        {
            return std::nullopt;
        }

        std::shared_ptr<DiskPrefixStorageBackend> self = weak_from_this().lock();
        if (!self)
        {
            if (error)
                *error = "verified hydration requires a shared archive lifetime";
            return std::nullopt;
        }

        /*
         * Match ordinary read semantics by making the requested payload most
         * recent before RAM pressure writes a victim.  The physical snapshot
         * below still makes a capacity-one RAM/disk swap correct if that write
         * must evict this logical record anyway.
         */
        if (!appendTouchLocked(archive.get(), key, error))
            return std::nullopt;

        const RecordIndex &touched_record = records_.at(key);
        std::array<HydrationTicket::SectionSnapshot, kSectionCount> sections{};
        for (size_t index = 0; index < sections.size(); ++index)
        {
            sections[index] = {
                .offset = touched_record.sections[index].offset,
                .bytes = touched_record.sections[index].bytes,
                .checksum = touched_record.sections[index].checksum,
            };
        }
        ++active_hydration_tickets_;
        return HydrationTicket(
            std::move(self),
            touched_record.handle,
            archive_device_,
            archive_inode_,
            std::move(sections));
    }

    bool DiskPrefixStorageBackend::hydrateVerified(
        HydrationTicket &ticket,
        RamPrefixStorageBackend &ram_backend,
        PrefixBlockHandle *ram_handle,
        std::string *error)
    {
        if (!ram_handle)
        {
            if (error)
                *error = "RAM hydration output is null";
            return false;
        }

        std::lock_guard<std::mutex> process_lock(mutex_);
        if (!ticket.valid() || ticket.backend_.get() != this)
        {
            if (error)
                *error = "verified hydration ticket is invalid, foreign, or consumed";
            return false;
        }
        ticket.consumed_ = true;

        AdvisoryLock file_lock(lock_path_);
        if (!file_lock.locked())
        {
            if (error)
                *error = errnoMessage("failed to lock prefix archive");
            return false;
        }
        FileDescriptor archive(::open(archive_path_.c_str(), O_RDWR | O_CLOEXEC));
        struct stat archive_state
        {
        };
        if (!archive || ::fstat(archive.get(), &archive_state) != 0 ||
            static_cast<uint64_t>(archive_state.st_dev) != ticket.archive_device_ ||
            static_cast<uint64_t>(archive_state.st_ino) != ticket.archive_inode_)
        {
            if (error)
                *error = "prefix archive changed after hydration verification";
            return false;
        }

        const PrefixBlockHandle &verified = ticket.handle_;
        PrefixBlockHandle out =
            ram_backend.allocate(verified.key, verified.layout);
        if (!out.valid())
        {
            if (error)
                *error = "failed to allocate RAM hydration handle";
            return false;
        }

        const size_t runtime_bytes =
            static_cast<size_t>(ticket.sections_[5].bytes);
        if (runtime_bytes > 0)
        {
            auto runtime_state =
                std::make_shared<std::vector<uint8_t>>(runtime_bytes);
            if (!ram_backend.attachModelRuntimeState(
                    &out, std::move(runtime_state)))
            {
                ram_backend.release(out);
                if (error)
                    *error = "failed to allocate accounted RAM runtime-state payload";
                return false;
            }
        }
        out.has_hybrid_state = verified.has_hybrid_state;
        out.has_terminal_hidden = verified.has_terminal_hidden;
        out.has_terminal_logits = verified.has_terminal_logits;
        if (out.total_bytes != verified.total_bytes ||
            out.has_model_runtime_state != verified.has_model_runtime_state)
        {
            ram_backend.release(out);
            if (error)
                *error = "prefix archive runtime-state accounting mismatch";
            return false;
        }

        const std::array<std::pair<void *, size_t>, 6> destinations = {{
            {out.kv_payload, out.kvBytes()},
            {out.hybrid_payload, out.hybridBytes()},
            {out.mtp_payload, out.layout.mtpKVBytes()},
            {out.terminal_hidden, out.terminalHiddenBytes()},
            {out.terminal_logits, out.terminalLogitsBytes()},
            {runtime_bytes > 0
                 ? out.model_runtime_state_storage->data()
                 : nullptr,
             runtime_bytes},
        }};
        for (size_t index = 0; index < destinations.size(); ++index)
        {
            const auto &[destination, destination_bytes] = destinations[index];
            const auto &section = ticket.sections_[index];
            if (section.bytes != destination_bytes ||
                (destination_bytes > 0 &&
                 (!destination ||
                  !preadAll(
                      archive.get(),
                      destination,
                      destination_bytes,
                      section.offset) ||
                  checksumBytes(destination, destination_bytes) !=
                      section.checksum)))
            {
                ram_backend.release(out);
                if (error)
                    *error = std::string("prefix payload checksum or size mismatch in ") +
                             kSectionNames[index];
                return false;
            }
        }

        /*
         * Refresh the logical index after any intervening demotion.  Touch only
         * when the verified record remains resident on disk; an evicted record
         * now has RAM as its sole payload authority.
         */
        if (!refreshIndexLocked(archive.get(), error))
        {
            ram_backend.release(out);
            return false;
        }
        if (records_.find(verified.key) != records_.end() &&
            !appendTouchLocked(archive.get(), verified.key, error))
        {
            ram_backend.release(out);
            return false;
        }

        *ram_handle = std::move(out);
        return true;
    }

    bool DiskPrefixStorageBackend::readBlockIntoRamBackend(
        const PrefixCacheKey &key,
        const PrefixPayloadLayout &layout,
        RamPrefixStorageBackend &ram_backend,
        PrefixBlockHandle *ram_handle,
        std::string *error)
    {
        if (!ram_handle)
        {
            if (error)
                *error = "RAM hydration output is null";
            return false;
        }

        std::lock_guard<std::mutex> process_lock(mutex_);
        if (!ready_)
        {
            if (error)
                *error = initialization_error_;
            return false;
        }
        AdvisoryLock file_lock(lock_path_);
        if (!file_lock.locked())
        {
            if (error)
                *error = errnoMessage("failed to lock prefix archive");
            return false;
        }
        FileDescriptor archive(::open(archive_path_.c_str(), O_RDWR | O_CLOEXEC));
        if (!archive || !refreshIndexLocked(archive.get(), error))
            return false;

        auto record_it = records_.find(key);
        if (record_it == records_.end() ||
            !fullLayoutMatch(record_it->second.handle.layout, layout))
        {
            if (error)
                *error = "prefix archive record not found or layout mismatch";
            return false;
        }
        const RecordIndex &record = record_it->second;

        PrefixBlockHandle out = ram_backend.allocate(key, layout);
        if (!out.valid())
        {
            if (error)
                *error = "failed to allocate RAM hydration handle";
            return false;
        }

        const size_t runtime_bytes =
            static_cast<size_t>(record.sections[5].bytes);
        if (runtime_bytes > 0)
        {
            auto runtime_state =
                std::make_shared<std::vector<uint8_t>>(runtime_bytes);
            if (!ram_backend.attachModelRuntimeState(
                    &out, std::move(runtime_state)))
            {
                ram_backend.release(out);
                if (error)
                    *error = "failed to allocate accounted RAM runtime-state payload";
                return false;
            }
        }
        out.has_hybrid_state = record.handle.has_hybrid_state;
        out.has_terminal_hidden = record.handle.has_terminal_hidden;
        out.has_terminal_logits = record.handle.has_terminal_logits;
        if (out.total_bytes != record.handle.total_bytes ||
            out.has_model_runtime_state !=
                record.handle.has_model_runtime_state)
        {
            ram_backend.release(out);
            if (error)
                *error = "prefix archive runtime-state accounting mismatch";
            return false;
        }

        const std::array<std::pair<void *, size_t>, 6> destinations = {{
            {out.kv_payload, out.kvBytes()},
            {out.hybrid_payload, out.hybridBytes()},
            {out.mtp_payload, out.layout.mtpKVBytes()},
            {out.terminal_hidden, out.terminalHiddenBytes()},
            {out.terminal_logits, out.terminalLogitsBytes()},
            {runtime_bytes > 0 ? out.model_runtime_state_storage->data() : nullptr,
             runtime_bytes},
        }};

        for (size_t index = 0; index < destinations.size(); ++index)
        {
            const auto &[destination, destination_bytes] = destinations[index];
            const SectionIndex &section = record.sections[index];
            if (section.bytes != destination_bytes ||
                (destination_bytes > 0 &&
                 (!destination ||
                  !preadAll(
                      archive.get(),
                      destination,
                      destination_bytes,
                      section.offset) ||
                  checksumBytes(destination, destination_bytes) !=
                      section.checksum)))
            {
                ram_backend.release(out);
                if (error)
                    *error = std::string("prefix payload checksum or size mismatch in ") +
                             kSectionNames[index];
                return false;
            }
        }

        /*
         * Payload verification completes before recency publication. A failed
         * checksum must never make a corrupt record look newly used. Touches
         * are payload-free append records, so this preserves durable LRU without
         * rewriting a multi-megabyte KV payload on every cache hit.
         */
        if (!appendTouchLocked(archive.get(), key, error))
        {
            ram_backend.release(out);
            return false;
        }
        *ram_handle = std::move(out);
        return compactIfNeededLocked(archive.get(), error);
    }

    bool DiskPrefixStorageBackend::verifyRecordPayloadLocked(
        int archive_fd,
        const RecordIndex &record,
        std::string *error)
    {
        if (archive_scratch_.empty())
        {
            if (error)
                *error = "prefix archive I/O scratch is unavailable";
            return false;
        }

        for (size_t index = 0; index < record.sections.size(); ++index)
        {
            const SectionIndex &section = record.sections[index];
            uint64_t remaining = section.bytes;
            uint64_t source_offset = section.offset;
            uint64_t checksum = hashPrefixBytes("", 0);
            while (remaining > 0)
            {
                const size_t chunk = static_cast<size_t>(
                    std::min<uint64_t>(remaining, archive_scratch_.size()));
                if (!preadAll(
                        archive_fd,
                        archive_scratch_.data(),
                        chunk,
                        source_offset))
                {
                    if (error)
                        *error = std::string("failed to stream prefix payload section ") +
                                 kSectionNames[index];
                    return false;
                }
                checksum = hashPrefixBytes(
                    archive_scratch_.data(), chunk, checksum);
                source_offset += chunk;
                remaining -= chunk;
            }
            if (checksum != section.checksum)
            {
                if (error)
                    *error = std::string("prefix payload checksum mismatch in ") +
                             kSectionNames[index];
                return false;
            }
        }
        return true;
    }

    std::vector<PrefixBlockHandle> DiskPrefixStorageBackend::compatibleEntries(
        uint64_t fingerprint,
        const PrefixPayloadLayout &layout,
        std::string *error)
    {
        std::lock_guard<std::mutex> process_lock(mutex_);
        std::vector<PrefixBlockHandle> entries;
        if (!ready_)
        {
            if (error)
                *error = initialization_error_;
            return entries;
        }
        AdvisoryLock file_lock(lock_path_);
        if (!file_lock.locked())
        {
            if (error)
                *error = errnoMessage("failed to lock prefix archive");
            return entries;
        }
        FileDescriptor archive(::open(archive_path_.c_str(), O_RDWR | O_CLOEXEC));
        if (!archive || !refreshIndexLocked(archive.get(), error))
            return entries;

        std::vector<const RecordIndex *> ordered;
        for (const auto &[key, record] : records_)
        {
            if (key.fingerprint == fingerprint &&
                runtimeLayoutMatch(record.handle.layout, layout))
            {
                ordered.push_back(&record);
            }
        }
        std::sort(
            ordered.begin(),
            ordered.end(),
            [](const RecordIndex *lhs, const RecordIndex *rhs)
            {
                return lhs->sequence < rhs->sequence;
            });
        entries.reserve(ordered.size());
        for (const RecordIndex *record : ordered)
            entries.push_back(record->handle);
        return entries;
    }

    bool DiskPrefixStorageBackend::compactIfNeededLocked(
        int archive_fd,
        std::string *error)
    {
        /*
         * A verified ticket may refer to a put record that was logically
         * evicted during a RAM/disk swap.  Append-only mutation preserves its
         * bytes; compaction is the sole operation that could invalidate those
         * immutable offsets, so defer it until every in-process ticket retires.
         */
        if (active_hydration_tickets_ != 0u)
            return true;

        const uint64_t physical_bytes = fileSize(archive_fd);
        const uint64_t threshold = std::max<uint64_t>(
            64ull * 1024ull * 1024ull,
            kArchiveHeaderBytes +
                (budget_bytes_ == 0
                     ? active_bytes_ * 2
                     : static_cast<uint64_t>(budget_bytes_) * 2));
        if (physical_bytes <= threshold ||
            physical_bytes <= kArchiveHeaderBytes + active_bytes_ * 3 / 2)
        {
            return true;
        }
        return rewriteArchiveLocked(archive_fd, error);
    }

    void DiskPrefixStorageBackend::releaseHydrationTicket() noexcept
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (active_hydration_tickets_ == 0u)
            std::terminate();
        --active_hydration_tickets_;
    }

    bool DiskPrefixStorageBackend::rewriteArchiveLocked(
        int source_fd,
        std::string *error)
    {
        const std::filesystem::path temporary =
            archive_path_.string() + ".compact." + std::to_string(::getpid());
        FileDescriptor destination(::open(
            temporary.c_str(),
            O_RDWR | O_CREAT | O_TRUNC | O_CLOEXEC,
            0600));
        if (!destination)
        {
            if (error)
                *error = errnoMessage("failed to create compact prefix archive");
            return false;
        }

        std::array<uint8_t, kArchiveHeaderBytes> header{};
        if (!preadAll(source_fd, header.data(), header.size(), 0) ||
            !writeAll(destination.get(), header.data(), header.size()))
        {
            if (error)
                *error = "failed to copy prefix archive header during compaction";
            (void)::unlink(temporary.c_str());
            return false;
        }

        std::vector<const RecordIndex *> ordered;
        ordered.reserve(records_.size());
        for (const auto &[key, record] : records_)
        {
            (void)key;
            ordered.push_back(&record);
        }
        std::sort(
            ordered.begin(),
            ordered.end(),
            [](const RecordIndex *lhs, const RecordIndex *rhs)
            {
                return lhs->sequence < rhs->sequence;
            });

        for (const RecordIndex *record : ordered)
        {
            uint64_t remaining = record->record_bytes;
            uint64_t source_offset = record->record_offset;
            while (remaining > 0)
            {
                const size_t chunk = static_cast<size_t>(
                    std::min<uint64_t>(remaining, archive_scratch_.size()));
                if (!preadAll(
                        source_fd,
                        archive_scratch_.data(),
                        chunk,
                        source_offset) ||
                    !writeAll(destination.get(), archive_scratch_.data(), chunk))
                {
                    if (error)
                        *error = "failed to copy active prefix record during compaction";
                    (void)::unlink(temporary.c_str());
                    return false;
                }
                source_offset += chunk;
                remaining -= chunk;
            }
        }

        /*
         * The copied put records retain their original sequence fields. Append
         * compact touch records in current LRU order so a restart reconstructs
         * the same recency ordering instead of reverting to historical put
         * order after compaction.
         */
        for (const RecordIndex *record : ordered)
        {
            if (!appendTouchLocked(
                    destination.get(),
                    record->handle.key,
                    error))
            {
                (void)::unlink(temporary.c_str());
                return false;
            }
        }

        if (::fsync(destination.get()) != 0 ||
            ::rename(temporary.c_str(), archive_path_.c_str()) != 0)
        {
            if (error)
                *error = errnoMessage("failed to publish compact prefix archive");
            (void)::unlink(temporary.c_str());
            return false;
        }

        /*
         * The active index points into the old inode. Mark it invalid so the
         * next operation rescans the newly renamed archive before reading.
         */
        archive_device_ = 0;
        archive_inode_ = 0;
        scan_offset_ = 0;
        return true;
    }
} // namespace llaminar2
