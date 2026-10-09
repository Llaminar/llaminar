/**
 * @file PlanningGGUFFixture.h
 * @brief Tiny sparse GGUF source for planning and explicit loading regressions.
 *
 * The real parser reads metadata and tensor extents. Payload holes may be used
 * by explicit source-loading tests, never as real-model numerical evidence.
 * Each fixture owns one unique file,
 * allowing Unit groups to run concurrently without a shared temporary folder.
 */
#pragma once
#include "loaders/ModelLoader.h"
#include <filesystem>
#include <fstream>
#include <optional>
#include <stdexcept>
#include <span>
#include <string>
#include <vector>
#include <unistd.h>

namespace llaminar2::test
{
    /** @brief Move-forbidden owner of one metadata-only dense or MoE GGUF. */
    class PlanningGGUFFixture final
    {
    public:
        /** @brief Directory-only geometry or complete all-attention graph metadata. */
        enum class Geometry { Directory, FullAttentionGraph };
        /** @return Exact GGUF encoding for a source tensor; runtime-only formats are rejected. */
        static GGUFTensorType sourceType(TensorType type)
        {
            switch (type)
            {
            case TensorType::FP32: return GGUFTensorType::F32;
            case TensorType::FP16: return GGUFTensorType::F16;
            case TensorType::BF16: return GGUFTensorType::BF16;
#define LLAMINAR_PLANNING_GGUF_FORMAT(name) case TensorType::name: return GGUFTensorType::name;
            LLAMINAR_PLANNING_GGUF_FORMAT(Q4_0)
            LLAMINAR_PLANNING_GGUF_FORMAT(Q4_1)
            LLAMINAR_PLANNING_GGUF_FORMAT(Q5_0)
            LLAMINAR_PLANNING_GGUF_FORMAT(Q5_1)
            LLAMINAR_PLANNING_GGUF_FORMAT(Q8_0)
            LLAMINAR_PLANNING_GGUF_FORMAT(Q2_K)
            LLAMINAR_PLANNING_GGUF_FORMAT(Q3_K)
            LLAMINAR_PLANNING_GGUF_FORMAT(Q4_K)
            LLAMINAR_PLANNING_GGUF_FORMAT(Q5_K)
            LLAMINAR_PLANNING_GGUF_FORMAT(Q6_K)
            LLAMINAR_PLANNING_GGUF_FORMAT(Q8_K)
            LLAMINAR_PLANNING_GGUF_FORMAT(IQ1_S)
            LLAMINAR_PLANNING_GGUF_FORMAT(IQ1_M)
            LLAMINAR_PLANNING_GGUF_FORMAT(IQ2_XXS)
            LLAMINAR_PLANNING_GGUF_FORMAT(IQ2_XS)
            LLAMINAR_PLANNING_GGUF_FORMAT(IQ2_S)
            LLAMINAR_PLANNING_GGUF_FORMAT(IQ3_XXS)
            LLAMINAR_PLANNING_GGUF_FORMAT(IQ3_S)
            LLAMINAR_PLANNING_GGUF_FORMAT(IQ4_NL)
            LLAMINAR_PLANNING_GGUF_FORMAT(IQ4_XS)
#undef LLAMINAR_PLANNING_GGUF_FORMAT
            default: throw std::invalid_argument("Tensor format has no supported GGUF source encoding");
            }
        }

        /**
         * @brief Populate an exact interval of this fixture with valid native tensor bytes.
         * @param name Directory tensor whose source bytes are being written.
         * @param byte_offset Relative byte offset, including for one expert in a parent.
         * @param bytes Bounded payload; does not change directory geometry or format.
         */
        void writePayload(const std::string &name, size_t byte_offset, std::span<const uint8_t> bytes)
        {
            ModelLoader directory;
            directory.setUseMmap(false);
            if (!directory.loadModel(path_)) throw std::runtime_error("Cannot read fixture directory");
            const auto &model = directory.getModel();
            const auto *tensor = model.findTensor(name);
            if (!tensor || byte_offset > tensor->size_bytes || bytes.size() > tensor->size_bytes - byte_offset)
                throw std::invalid_argument("Fixture payload exceeds its source tensor");
            std::fstream stream(path_, std::ios::binary | std::ios::in | std::ios::out);
            stream.exceptions(std::ios::badbit | std::ios::failbit);
            stream.seekp(model.data_offset + tensor->offset + byte_offset);
            stream.write(reinterpret_cast<const char *>(bytes.data()), bytes.size());
        }

        /**
         * @brief Write complete main and optional NextN tensor directories.
         * @param moe Select routed/shared expert tensors instead of dense FFNs.
         * @param mtp Append one real-layout predictor directory, not fifteen replicas.
         * @param expert_type Native FFN/expert format; extents use the production directory ABI.
         * @param expert_width Expert intermediate dimension; unequal N/K exposes axis errors.
         * @param gdn_projection Optional alpha/beta source format for canonical runtime-promotion tests.
         * @param attention_heads Integral Q-head count; source KV and norm tensors
         *        use its actual head width, allowing valid high-degree GQA tests.
         * @param geometry Graph loading additionally needs the gated Q projection
         *        and nonzero recurrent arena metadata, even for all-FA layers.
         */
        explicit PlanningGGUFFixture(bool moe = false, bool mtp = false,
            GGUFTensorType expert_type = GGUFTensorType::F32, uint32_t expert_width = 256,
            std::optional<GGUFTensorType> gdn_projection = std::nullopt, uint32_t attention_heads = 8,
            Geometry geometry = Geometry::Directory)
        {
            if (GGUFTensorInfo{.type = expert_type}.getTypeSize() == 0)
                throw std::invalid_argument("Planning fixture requires a supported GGUF format");
            char pattern[] = "/tmp/llaminar-planning-XXXXXX.gguf";
            const int descriptor = ::mkstemps(pattern, 5);
            if (descriptor < 0) throw std::runtime_error("Cannot create planning GGUF fixture");
            ::close(descriptor);
            path_ = pattern;
            try { write(moe, mtp, expert_type, expert_width, gdn_projection, attention_heads, geometry); }
            catch (...) { remove(); throw; }
        }
        /** @brief Remove only this fixture's file, including during exception unwinding. */
        ~PlanningGGUFFixture() { remove(); }
        PlanningGGUFFixture(const PlanningGGUFFixture &) = delete;
        PlanningGGUFFixture &operator=(const PlanningGGUFFixture &) = delete;
        /** @return Stable path consumed by the public metadata loader. */
        const std::string &path() const noexcept { return path_; }

    private:
        /** @brief One on-disk tensor directory entry, before payload offset assignment. */
        struct Tensor
        {
            std::string name;
            std::vector<uint64_t> dimensions;
            GGUFTensorType type = GGUFTensorType::F32;
        };
        /** @brief Encode an unsigned GGUF scalar explicitly in little-endian order. */
        template<class T> static void scalar(std::ostream &stream, T value)
        {
            for (size_t byte = 0; byte < sizeof(T); ++byte)
                stream.put(static_cast<char>(value >> (8 * byte)));
        }
        /** @brief Encode a GGUF length-prefixed string. */
        static void string(std::ostream &stream, const std::string &value)
        {
            scalar<uint64_t>(stream, value.size());
            stream.write(value.data(), static_cast<std::streamsize>(value.size()));
        }
        /** @brief Publish the directory followed by sparse, unread payload extents. */
        void write(bool moe, bool mtp, GGUFTensorType expert_type, uint32_t expert_width,
            std::optional<GGUFTensorType> gdn_projection, uint32_t attention_heads, Geometry geometry)
        {
            if (attention_heads < 2 || 256 % attention_heads != 0)
                throw std::invalid_argument("Planning fixture needs integral attention heads and two KV heads");
            const uint64_t head_columns = 256 / attention_heads;
            const std::string arch = moe ? "qwen35moe" : "qwen35";
            const int layers = mtp ? 3 : 2;
            std::vector<Tensor> tensors{
                {"token_embd.weight", {256, 320}}, {"output.weight", {256, 320}},
                {"output_norm.weight", {256}}};
            for (int layer = 0; layer < layers; ++layer)
            {
                const auto prefix = "blk." + std::to_string(layer) + ".";
                for (const auto name : {"attn_norm.weight", "post_attention_norm.weight"})
                    tensors.push_back({prefix + name, {256}});
                for (const auto name : {"attn_q_norm.weight", "attn_k_norm.weight"})
                    tensors.push_back({prefix + name, {head_columns}});
                tensors.push_back({prefix + "attn_q.weight", {256,
                    geometry == Geometry::FullAttentionGraph ? 512u : 256u}});
                tensors.push_back({prefix + "attn_output.weight", {256, 256}});
                for (const auto name : {"attn_k.weight", "attn_v.weight"})
                    tensors.push_back({prefix + name, {256, 2 * head_columns}});
                if (gdn_projection)
                    for (const auto name : {"ssm_alpha.weight", "ssm_beta.weight"})
                        tensors.push_back({prefix + name, {256, 256}, *gdn_projection});
                if (moe)
                {
                    for (const auto name : {"ffn_gate_exps.weight", "ffn_up_exps.weight"})
                        tensors.push_back({prefix + name, {256, expert_width, 8}, expert_type});
                    tensors.push_back({prefix + "ffn_down_exps.weight", {expert_width, 256, 8}, expert_type});
                    tensors.push_back({prefix + "ffn_gate_inp.weight", {256, 8}});
                    const uint64_t shared_width = geometry == Geometry::FullAttentionGraph ? expert_width : 256;
                    for (const auto name : {"ffn_gate_shexp.weight", "ffn_up_shexp.weight"})
                        tensors.push_back({prefix + name, {256, shared_width}});
                    tensors.push_back({prefix + "ffn_down_shexp.weight", {shared_width, 256}});
                    tensors.push_back({prefix + "ffn_gate_inp_shexp.weight", {256}});
                }
                else
                {
                    for (const auto name : {"ffn_gate.weight", "ffn_up.weight"})
                        tensors.push_back({prefix + name, {256, 512}, expert_type});
                    tensors.push_back({prefix + "ffn_down.weight", {512, 256}, expert_type});
                }
                if (mtp && layer == 2)
                {
                    tensors.push_back({prefix + "nextn.eh_proj.weight", {512, 256}});
                    for (const auto name : {"nextn.hnorm.weight", "nextn.enorm.weight", "nextn.shared_head_norm.weight"})
                        tensors.push_back({prefix + name, {256}});
                }
            }
            std::vector<std::pair<std::string, uint32_t>> metadata{
                {arch + ".block_count", static_cast<uint32_t>(layers)},
                {arch + ".embedding_length", 256},
                {arch + ".feed_forward_length", moe && geometry == Geometry::FullAttentionGraph ? expert_width : 512},
                {arch + ".attention.head_count", attention_heads}, {arch + ".attention.head_count_kv", 2},
                {arch + ".context_length", 1024}, {"tokenizer.ggml.token_count", 320},
                {arch + ".expert_count", moe ? 8u : 0u},
                {arch + ".expert_used_count", moe ? 2u : 0u},
                {arch + ".expert_feed_forward_length", moe ? expert_width : 0u},
                {arch + ".nextn_predict_layers", mtp ? 1u : 0u}};
            if (geometry == Geometry::FullAttentionGraph)
                metadata.insert(metadata.end(), {
                    {arch + ".expert_shared_count", moe ? 1u : 0u},
                    {arch + ".full_attention_interval", 1},
                    {arch + ".ssm.conv_kernel", 4},
                    {arch + ".ssm.state_size", static_cast<uint32_t>(head_columns)},
                    {arch + ".ssm.inner_size", 256},
                    {arch + ".ssm.group_count", attention_heads},
                    {arch + ".ssm.time_step_rank", attention_heads}});
            std::ofstream stream(path_, std::ios::binary | std::ios::trunc);
            stream.exceptions(std::ios::badbit | std::ios::failbit);
            stream.write("GGUF", 4);
            scalar<uint32_t>(stream, 3);
            scalar<uint64_t>(stream, tensors.size());
            scalar<uint64_t>(stream, metadata.size() + 1 + (geometry == Geometry::FullAttentionGraph ? 3 : 0));
            string(stream, "general.architecture");
            scalar<uint32_t>(stream, 8);
            string(stream, arch);
            for (const auto &[key, value] : metadata)
            {
                string(stream, key);
                scalar<uint32_t>(stream, 4);
                scalar<uint32_t>(stream, value);
            }
            if (geometry == Geometry::FullAttentionGraph)
            {
                // Initialization consumes a real tokenizer directory even
                // though this ownership fixture never submits text prompts.
                for (const auto &[key, value] : std::vector<std::pair<std::string, std::string>>{
                    {"tokenizer.ggml.model", "gpt2"}, {"tokenizer.ggml.pre", "qwen2"}})
                {
                    string(stream, key);
                    scalar<uint32_t>(stream, 8);
                    string(stream, value);
                }
                string(stream, "tokenizer.ggml.tokens");
                scalar<uint32_t>(stream, 9);
                scalar<uint32_t>(stream, 8);
                scalar<uint64_t>(stream, 320);
                // GPT-2 byte BPE must cover every byte before extra tokens.
                // Non-printing bytes map to the consecutive U+0100 range.
                int escaped = 256;
                for (int byte = 0; byte < 256; ++byte)
                {
                    const int point = (byte >= 33 && byte <= 126) ||
                        (byte >= 161 && byte <= 172) || byte >= 174 ? byte : escaped++;
                    std::string encoded;
                    if (point < 128) encoded.push_back(static_cast<char>(point));
                    else
                    {
                        encoded.push_back(static_cast<char>(0xc0 | (point >> 6)));
                        encoded.push_back(static_cast<char>(0x80 | (point & 63)));
                    }
                    string(stream, encoded);
                }
                for (int token = 256; token < 320; ++token)
                    string(stream, "fixture_token_" + std::to_string(token));
            }
            uint64_t offset = 0;
            for (const auto &tensor : tensors)
            {
                string(stream, tensor.name);
                scalar<uint32_t>(stream, tensor.dimensions.size());
                uint64_t elements = 1;
                for (const auto dimension : tensor.dimensions)
                {
                    scalar<uint64_t>(stream, dimension);
                    elements *= dimension;
                }
                const GGUFTensorInfo info{.type = tensor.type};
                const auto block = std::max(size_t{1}, info.getBlockSize());
                if (elements % block) throw std::invalid_argument("Unaligned planning fixture payload");
                const uint64_t bytes = elements / block * info.getTypeSize();
                scalar<uint32_t>(stream, static_cast<uint32_t>(tensor.type));
                scalar<uint64_t>(stream, offset);
                offset += bytes;
            }
            const uint64_t aligned = (static_cast<uint64_t>(stream.tellp()) + 31) / 32 * 32;
            stream.seekp(static_cast<std::streamoff>(aligned + offset - 1));
            stream.put('\0');
        }
        /** @brief Best-effort unlink of the exact uniquely owned fixture path. */
        void remove() noexcept
        {
            std::error_code ignored;
            std::filesystem::remove(path_, ignored);
        }
        std::string path_;
    };
}
