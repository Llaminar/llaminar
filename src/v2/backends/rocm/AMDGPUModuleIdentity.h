/**
 * @file AMDGPUModuleIdentity.h
 * @brief Identify the loaded Linux amdgpu module for native context admission.
 *
 * DKMS modules can publish MODULE_VERSION. Linux distribution modules may
 * omit it while exposing the linked module's original GNU build ID through
 * sysfs. Both forms bind the loaded kernel release and module source version.
 */
#pragma once

#include <cstring>
#include <elf.h>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace llaminar2 {

/** @brief Original source and version inputs to the ROCm worker's driver digest. */
struct AMDGPUModuleIdentity {
    std::string version;
    std::string source;

    /**
     * @brief Read identity from the loaded module and the running kernel.
     * @param module Sysfs directory of the loaded amdgpu module.
     * @param release Kernel release returned by uname.
     * @return Exact version and source bytes, preserving existing versioned receipts.
     * @throws std::runtime_error When a required identity is absent or malformed.
     */
    static AMDGPUModuleIdentity observe(const std::filesystem::path& module,
                                       const std::string& release) {
        if (release.empty()) throw std::runtime_error("Missing running kernel release");
        auto source = read(module / "srcversion");
        const auto versionPath = module / "version";
        auto version = std::filesystem::exists(versionPath)
            ? read(versionPath) : linkedBuildId(read(module / "notes/.note.gnu.build-id"));
        return {version + ":" + release, std::move(source)};
    }

private:
    /** @brief Read an original module attribute, retaining any terminal newline or binary zeroes. */
    static std::string read(const std::filesystem::path& path) {
        std::ifstream file(path, std::ios::binary);
        if (!file) throw std::runtime_error("Cannot read amdgpu module identity: " + path.string());
        std::string bytes{std::istreambuf_iterator<char>(file), {}};
        if (file.bad() || bytes.empty())
            throw std::runtime_error("Incomplete amdgpu module identity: " + path.string());
        return bytes;
    }

    /** @brief Decode the single native-endian GNU ELF note exported by the kernel. */
    static std::string linkedBuildId(const std::string& note) {
        Elf32_Nhdr header{};
        constexpr char owner[] = "GNU";
        constexpr auto offset = sizeof(header) + sizeof(owner);
        if (note.size() >= offset) std::memcpy(&header, note.data(), sizeof(header));
        if (header.n_type != NT_GNU_BUILD_ID || header.n_namesz != sizeof(owner) ||
            header.n_descsz == 0 || note.size() < offset ||
            (note.size() - offset) != ((static_cast<std::size_t>(header.n_descsz) + 3) & ~std::size_t{3}) ||
            std::memcmp(note.data() + sizeof(header), owner, sizeof(owner)) != 0)
            throw std::runtime_error("Invalid amdgpu GNU build ID note");
        std::string id = "amdgpu-build-id-";
        constexpr char hex[] = "0123456789abcdef";
        for (unsigned char byte : std::string_view(note).substr(offset, header.n_descsz)) {
            id += hex[byte >> 4];
            id += hex[byte & 15];
        }
        return id;
    }
};
}
