/**
 * @file CloverMoESourceFixture.h
 * @brief Owned sparse GGUF encoding of the tiny Clover source-directory fixture.
 *
 * The production parser, ModelContext and planner must see the original Preview
 * inventory, while causal materialization omits diffusion operands. Payloads are
 * sparse holes, not numerical model data; this fixture never executes inference.
 */
#pragma once
#include "CloverMoEMetadataFixture.h"
#include <algorithm>
#include <bit>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <unistd.h>

namespace llaminar2::test
{
    /** @brief Unique temporary source retired even when parsing or admission fails. */
    class CloverMoESourceFixture final
    {
    public:
        /**
         * @brief Encode a bounded synthetic directory in native GGUF wire order.
         * @param model Small metadata fixture; tensor payloads are deliberately absent.
         * @throws std::runtime_error On file creation or writing failure.
         * @throws std::invalid_argument For a nonfixture metadata subtype or excessive extent.
         */
        explicit CloverMoESourceFixture(const GGUFModel &model = cloverMetadataFixture())
        {
            static_assert(std::endian::native == std::endian::little);
            char name[] = "/tmp/llaminar-clover-XXXXXX.gguf";
            const int descriptor = ::mkstemps(name, 5);
            if (descriptor < 0) throw std::runtime_error("Cannot create Clover source fixture");
            ::close(descriptor);
            path_ = name;
            try { write(model); }
            catch (...) { remove(); throw; }
        }
        /** @brief Retire only the unique file owned by this fixture. */
        ~CloverMoESourceFixture() { remove(); }
        CloverMoESourceFixture(const CloverMoESourceFixture &) = delete;
        CloverMoESourceFixture &operator=(const CloverMoESourceFixture &) = delete;
        /** @return Stable original source path for metadata-only production consumers. */
        const std::string &path() const noexcept { return path_; }

    private:
        /** @brief Encode a scalar in the fixture's checked little-endian host order. */
        template<class T> static void scalar(std::ostream &out, T value)
        {
            out.write(reinterpret_cast<const char *>(&value), sizeof(value));
        }
        /** @brief Encode one length-prefixed GGUF string. */
        static void string(std::ostream &out, const std::string &value)
        {
            scalar<uint64_t>(out, value.size());
            out.write(value.data(), value.size());
        }
        /** @brief Publish the original metadata/directory and bounded sparse payload extent. */
        void write(const GGUFModel &model)
        {
            std::ofstream out(path_, std::ios::binary | std::ios::trunc);
            out.exceptions(std::ios::badbit | std::ios::failbit);
            out.write("GGUF", 4);
            scalar<uint32_t>(out, 3);
            scalar<uint64_t>(out, model.tensors.size());
            scalar<uint64_t>(out, model.metadata.size() + 1);
            string(out, "general.architecture");
            scalar<uint32_t>(out, static_cast<uint32_t>(GGUFValueType::STRING));
            string(out, model.architecture);
            std::vector<std::string> keys;
            for (const auto &[key, value] : model.metadata) keys.push_back(key);
            std::sort(keys.begin(), keys.end());
            for (const auto &key : keys)
            {
                const auto &value = model.metadata.at(key);
                string(out, key);
                scalar<uint32_t>(out, static_cast<uint32_t>(value.type));
                if (value.type == GGUFValueType::ARRAY && value.array_element_type == GGUFValueType::BOOL)
                {
                    scalar<uint32_t>(out, static_cast<uint32_t>(*value.array_element_type));
                    scalar<uint64_t>(out, value.array_length);
                }
                else if (value.type != GGUFValueType::UINT32 && value.type != GGUFValueType::FLOAT32)
                    throw std::invalid_argument("Clover fixture only encodes UINT32, FLOAT32 and BOOL arrays");
                out.write(reinterpret_cast<const char *>(value.data.data()), value.data.size());
            }
            uint64_t payload = 0;
            for (const auto &tensor : model.tensors)
            {
                string(out, tensor.name);
                scalar<uint32_t>(out, tensor.dimensions.size());
                auto dimensions = tensor.dimensions;
                // The production loader transposes only rank-two directory shapes.
                if (dimensions.size() == 2) std::swap(dimensions[0], dimensions[1]);
                for (const auto extent : dimensions) scalar<uint64_t>(out, extent);
                scalar<uint32_t>(out, static_cast<uint32_t>(tensor.type));
                scalar<uint64_t>(out, payload);
                if (tensor.size_bytes > 64 * 1024 * 1024 - payload)
                    throw std::invalid_argument("Clover fixture exceeds its 64-MiB sparse extent limit");
                payload = (payload + tensor.size_bytes + 31) & ~uint64_t{31};
            }
            const auto start = (static_cast<uint64_t>(out.tellp()) + 31) & ~uint64_t{31};
            out.seekp(start + payload - 1);
            out.put('\0');
        }
        /** @brief Nonthrowing destruction of this fixture, never another test's files. */
        void remove() noexcept
        {
            std::error_code ignored;
            std::filesystem::remove(path_, ignored);
        }
        std::string path_;
    };
}
