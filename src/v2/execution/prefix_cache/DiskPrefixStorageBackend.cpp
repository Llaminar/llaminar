#include "execution/prefix_cache/DiskPrefixStorageBackend.h"

#include "execution/prefix_cache/BlockHash.h"
#include "execution/prefix_cache/RamPrefixStorageBackend.h"

#include <fstream>
#include <sstream>

namespace llaminar2
{
    namespace
    {
        bool writeBytes(const std::filesystem::path &path, const std::shared_ptr<std::vector<uint8_t>> &bytes)
        {
            std::ofstream out(path, std::ios::binary);
            if (!out)
            {
                return false;
            }
            if (bytes && !bytes->empty())
            {
                out.write(reinterpret_cast<const char *>(bytes->data()), static_cast<std::streamsize>(bytes->size()));
            }
            return out.good();
        }

        bool readBytes(const std::filesystem::path &path, std::vector<uint8_t> *bytes)
        {
            if (!bytes)
            {
                return false;
            }
            std::ifstream in(path, std::ios::binary);
            if (!in)
            {
                return false;
            }
            in.seekg(0, std::ios::end);
            const auto size = in.tellg();
            if (size < 0)
            {
                return false;
            }
            in.seekg(0, std::ios::beg);
            bytes->resize(static_cast<size_t>(size));
            if (!bytes->empty())
            {
                in.read(reinterpret_cast<char *>(bytes->data()), size);
            }
            return in.good() || in.eof();
        }

        uint64_t checksumVector(const std::shared_ptr<std::vector<uint8_t>> &bytes)
        {
            if (!bytes || bytes->empty())
            {
                return hashPrefixBytes("", 0);
            }
            return hashPrefixBytes(bytes->data(), bytes->size());
        }

        bool extractUInt64(const std::string &json, const std::string &name, uint64_t *value)
        {
            const std::string needle = "\"" + name + "\":";
            const size_t pos = json.find(needle);
            if (pos == std::string::npos)
            {
                return false;
            }
            const size_t start = pos + needle.size();
            size_t end = start;
            while (end < json.size() && json[end] >= '0' && json[end] <= '9')
            {
                ++end;
            }
            if (end == start)
            {
                return false;
            }
            *value = static_cast<uint64_t>(std::stoull(json.substr(start, end - start)));
            return true;
        }

        bool extractString(const std::string &json, const std::string &name, std::string *value)
        {
            const std::string needle = "\"" + name + "\":\"";
            const size_t pos = json.find(needle);
            if (pos == std::string::npos)
            {
                return false;
            }
            const size_t start = pos + needle.size();
            const size_t end = json.find('"', start);
            if (end == std::string::npos)
            {
                return false;
            }
            *value = json.substr(start, end - start);
            return true;
        }
    } // namespace

    DiskPrefixStorageBackend::DiskPrefixStorageBackend(std::filesystem::path root_dir, size_t budget_bytes)
        : root_dir_(std::move(root_dir)), budget_bytes_(budget_bytes)
    {
    }

    bool DiskPrefixStorageBackend::canStore(size_t bytes) const
    {
        return budget_bytes_ == 0 || bytes <= budget_bytes_;
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

    bool DiskPrefixStorageBackend::release(const PrefixBlockHandle &handle)
    {
        std::error_code ec;
        std::filesystem::remove(blockMetaPath(handle.key), ec);
        std::filesystem::remove(blockPayloadPath(handle.key, "kv.bin"), ec);
        std::filesystem::remove(blockPayloadPath(handle.key, "hybrid.bin"), ec);
        std::filesystem::remove(blockPayloadPath(handle.key, "mtp.bin"), ec);
        std::filesystem::remove(blockPayloadPath(handle.key, "terminal_hidden.bin"), ec);
        std::filesystem::remove(blockPayloadPath(handle.key, "terminal_logits.bin"), ec);
        return !ec;
    }

    bool DiskPrefixStorageBackend::hydrateToRam(const PrefixBlockHandle &handle,
                                                PrefixBlockHandle *ram_handle)
    {
        return readBlock(handle.key, handle.layout, ram_handle, nullptr);
    }

    std::filesystem::path DiskPrefixStorageBackend::blockMetaPath(const PrefixCacheKey &key) const
    {
        return root_dir_ / "blocks" / (key.toHex() + ".meta.json");
    }

    std::filesystem::path DiskPrefixStorageBackend::blockPayloadPath(const PrefixCacheKey &key, const char *suffix) const
    {
        return root_dir_ / "blocks" / (key.toHex() + "." + std::string(suffix));
    }

    bool DiskPrefixStorageBackend::writeBlock(const PrefixBlockHandle &handle, std::string *error)
    {
        if (!handle.valid() || !canStore(handle.total_bytes))
        {
            if (error)
            {
                *error = "invalid handle or disk budget exceeded";
            }
            return false;
        }

        std::error_code ec;
        std::filesystem::create_directories(root_dir_ / "blocks", ec);
        if (ec)
        {
            if (error)
            {
                *error = ec.message();
            }
            return false;
        }

        if (!writeBytes(blockPayloadPath(handle.key, "kv.bin"), handle.kv_storage) ||
            !writeBytes(blockPayloadPath(handle.key, "hybrid.bin"), handle.hybrid_storage) ||
            !writeBytes(blockPayloadPath(handle.key, "mtp.bin"), handle.mtp_storage) ||
            !writeBytes(blockPayloadPath(handle.key, "terminal_hidden.bin"), handle.terminal_hidden_storage) ||
            !writeBytes(blockPayloadPath(handle.key, "terminal_logits.bin"), handle.terminal_logits_storage))
        {
            if (error)
            {
                *error = "failed to write payload file";
            }
            return false;
        }

        std::ofstream manifest(root_dir_ / "manifest.json");
        manifest << "{\"version\":1}\n";

        std::ofstream meta(blockMetaPath(handle.key));
        if (!meta)
        {
            if (error)
            {
                *error = "failed to write metadata";
            }
            return false;
        }

        meta << "{\n"
             << "\"version\":1,\n"
             << "\"key\":\"" << handle.key.toHex() << "\",\n"
             << "\"total_bytes\":" << handle.total_bytes << ",\n"
             << "\"kv_bytes\":" << (handle.kv_storage ? handle.kv_storage->size() : 0) << ",\n"
             << "\"hybrid_bytes\":" << (handle.hybrid_storage ? handle.hybrid_storage->size() : 0) << ",\n"
             << "\"mtp_bytes\":" << (handle.mtp_storage ? handle.mtp_storage->size() : 0) << ",\n"
             << "\"terminal_hidden_bytes\":" << (handle.terminal_hidden_storage ? handle.terminal_hidden_storage->size() : 0) << ",\n"
             << "\"terminal_logits_bytes\":" << (handle.terminal_logits_storage ? handle.terminal_logits_storage->size() : 0) << ",\n"
             << "\"has_hybrid_state\":" << (handle.has_hybrid_state ? 1 : 0) << ",\n"
             << "\"has_terminal_hidden\":" << (handle.has_terminal_hidden ? 1 : 0) << ",\n"
             << "\"has_terminal_logits\":" << (handle.has_terminal_logits ? 1 : 0) << ",\n"
             << "\"kv_checksum\":" << checksumVector(handle.kv_storage) << ",\n"
             << "\"hybrid_checksum\":" << checksumVector(handle.hybrid_storage) << ",\n"
             << "\"mtp_checksum\":" << checksumVector(handle.mtp_storage) << ",\n"
             << "\"terminal_hidden_checksum\":" << checksumVector(handle.terminal_hidden_storage) << ",\n"
             << "\"terminal_logits_checksum\":" << checksumVector(handle.terminal_logits_storage) << "\n"
             << "}\n";
        return meta.good();
    }

    bool DiskPrefixStorageBackend::readBlock(
        const PrefixCacheKey &key,
        const PrefixPayloadLayout &layout,
        PrefixBlockHandle *ram_handle,
        std::string *error) const
    {
        RamPrefixStorageBackend ram(layout.totalBytes());
        return readBlockIntoRamBackend(key, layout, ram, ram_handle, error);
    }

    bool DiskPrefixStorageBackend::readBlockIntoRamBackend(
        const PrefixCacheKey &key,
        const PrefixPayloadLayout &layout,
        IPrefixStorageBackend &ram_backend,
        PrefixBlockHandle *ram_handle,
        std::string *error) const
    {
        if (!ram_handle)
        {
            return false;
        }

        std::ifstream meta_in(blockMetaPath(key));
        if (!meta_in)
        {
            if (error)
            {
                *error = "metadata not found";
            }
            return false;
        }
        std::stringstream buffer;
        buffer << meta_in.rdbuf();
        const std::string meta = buffer.str();

        std::string key_hex;
        if (!extractString(meta, "key", &key_hex) || key_hex != key.toHex())
        {
            if (error)
            {
                *error = "metadata key mismatch";
            }
            return false;
        }

        PrefixBlockHandle out = ram_backend.allocate(key, layout);
        if (!out.valid())
        {
            if (error)
            {
                *error = "failed to allocate RAM hydration handle";
            }
            return false;
        }

        auto load = [&](const char *suffix,
                        std::shared_ptr<std::vector<uint8_t>> &storage,
                        const char *bytes_field,
                        const char *checksum_field) -> bool
        {
            if (!storage)
            {
                return true;
            }
            std::vector<uint8_t> bytes;
            if (!readBytes(blockPayloadPath(key, suffix), &bytes))
            {
                return false;
            }
            uint64_t expected_size = 0;
            uint64_t expected_checksum = 0;
            if (!extractUInt64(meta, bytes_field, &expected_size) ||
                !extractUInt64(meta, checksum_field, &expected_checksum))
            {
                return false;
            }
            if (bytes.size() != expected_size || checksumVector(std::make_shared<std::vector<uint8_t>>(bytes)) != expected_checksum ||
                bytes.size() != storage->size())
            {
                return false;
            }
            *storage = std::move(bytes);
            return true;
        };

        if (!load("kv.bin", out.kv_storage, "kv_bytes", "kv_checksum") ||
            !load("hybrid.bin", out.hybrid_storage, "hybrid_bytes", "hybrid_checksum") ||
            !load("mtp.bin", out.mtp_storage, "mtp_bytes", "mtp_checksum") ||
            !load("terminal_hidden.bin", out.terminal_hidden_storage, "terminal_hidden_bytes", "terminal_hidden_checksum") ||
            !load("terminal_logits.bin", out.terminal_logits_storage, "terminal_logits_bytes", "terminal_logits_checksum"))
        {
            ram_backend.release(out);
            if (error)
            {
                *error = "payload checksum or size mismatch";
            }
            return false;
        }

        out.kv_payload = out.kv_storage && !out.kv_storage->empty() ? out.kv_storage->data() : nullptr;
        out.hybrid_payload = out.hybrid_storage && !out.hybrid_storage->empty() ? out.hybrid_storage->data() : nullptr;
        out.mtp_payload = out.mtp_storage && !out.mtp_storage->empty() ? out.mtp_storage->data() : nullptr;
        out.terminal_hidden = out.terminal_hidden_storage && !out.terminal_hidden_storage->empty() ? out.terminal_hidden_storage->data() : nullptr;
        out.terminal_logits = out.terminal_logits_storage && !out.terminal_logits_storage->empty() ? out.terminal_logits_storage->data() : nullptr;
        uint64_t flag = 0;
        if (extractUInt64(meta, "has_hybrid_state", &flag))
        {
            out.has_hybrid_state = flag != 0;
        }
        if (extractUInt64(meta, "has_terminal_hidden", &flag))
        {
            out.has_terminal_hidden = flag != 0;
        }
        if (extractUInt64(meta, "has_terminal_logits", &flag))
        {
            out.has_terminal_logits = flag != 0;
        }
        *ram_handle = std::move(out);
        return true;
    }

} // namespace llaminar2
