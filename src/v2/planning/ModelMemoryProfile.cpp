#include "planning/ModelMemoryProfile.h"
#include "loaders/ModelLoader.h"
#include <regex>
#include <numeric>
#include <cstring>
#include <stdexcept>
#include <string_view>

/**
 * @file ModelMemoryProfile.cpp
 * @brief Builds and serializes compact model weight inventories for placement planning.
 *
 * The memory profile keeps the planning layer independent of loaded Tensor objects.
 * It records GGUF tensor names, element counts, native byte sizes, quantization
 * type strings, and inferred layer ownership so higher-level placement code can
 * estimate per-device weight pressure before constructing the execution graph.
 */

namespace llaminar2
{

    namespace
    {

        /** @brief Stable marker preceding every serialized memory profile. */
        constexpr uint32_t kProfileWireMagic = 0x4C4D5032U; // "LMP2"
        /** @brief Current profile layout, including exact MoE geometry. */
        constexpr uint32_t kProfileWireVersion = 2U;

        std::string ggufTypeToString(GGUFTensorType type)
        {
            switch (type)
            {
            case GGUFTensorType::F32:
                return "F32";
            case GGUFTensorType::F16:
                return "F16";
            case GGUFTensorType::Q4_0:
                return "Q4_0";
            case GGUFTensorType::Q4_1:
                return "Q4_1";
            case GGUFTensorType::Q5_0:
                return "Q5_0";
            case GGUFTensorType::Q5_1:
                return "Q5_1";
            case GGUFTensorType::Q8_0:
                return "Q8_0";
            case GGUFTensorType::Q2_K:
                return "Q2_K";
            case GGUFTensorType::Q3_K:
                return "Q3_K";
            case GGUFTensorType::Q4_K:
                return "Q4_K";
            case GGUFTensorType::Q5_K:
                return "Q5_K";
            case GGUFTensorType::Q6_K:
                return "Q6_K";
            case GGUFTensorType::Q8_K:
                return "Q8_K";
            case GGUFTensorType::IQ2_XXS:
                return "IQ2_XXS";
            case GGUFTensorType::IQ2_XS:
                return "IQ2_XS";
            case GGUFTensorType::IQ3_XXS:
                return "IQ3_XXS";
            case GGUFTensorType::IQ1_S:
                return "IQ1_S";
            case GGUFTensorType::IQ4_NL:
                return "IQ4_NL";
            case GGUFTensorType::IQ3_S:
                return "IQ3_S";
            case GGUFTensorType::IQ2_S:
                return "IQ2_S";
            case GGUFTensorType::IQ4_XS:
                return "IQ4_XS";
            case GGUFTensorType::IQ1_M:
                return "IQ1_M";
            case GGUFTensorType::BF16:
                return "BF16";
            default:
                return "UNKNOWN";
            }
        }

        int parseLayerIndex(const std::string &tensor_name)
        {
            // Match patterns like "blk.0.", "blk.12.", "layers.0.", etc.
            static const std::regex layer_regex(R"((?:blk|layers?)\.(\d+)\.)");
            std::smatch match;
            if (std::regex_search(tensor_name, match, layer_regex))
            {
                return std::stoi(match[1].str());
            }
            return -1;
        }

        size_t computeElements(const std::vector<uint64_t> &dimensions)
        {
            if (dimensions.empty())
                return 0;
            size_t result = 1;
            for (auto d : dimensions)
            {
                result *= static_cast<size_t>(d);
            }
            return result;
        }

        /** @brief Whether a tensor is one packed parent containing every routed expert. */
        bool isRoutedExpertParent(std::string_view name)
        {
            return name.ends_with(".ffn_gate_exps.weight") ||
                   name.ends_with(".ffn_up_exps.weight") ||
                   name.ends_with(".ffn_down_exps.weight");
        }

        /**
         * @brief Recover the K dimension of one logical matrix from GGUF axes.
         *
         * ModelLoader normalizes ordinary 2-D tensors to `[N, K]`. Routed
         * expert parents retain GGUF's three-dimensional `[K, N, experts]`
         * representation because loading slices the outer expert axis before
         * preparing an individual matrix. Treating the final axis as K turns
         * the expert count into a matrix dimension and grossly overprices
         * split-K workspace.
         *
         * @param tensor Parsed tensor directory entry.
         * @param expert_count Model-wide routed expert cardinality.
         * @return Inner dimension of one logical matrix, or zero for a scalar.
         * @throws std::invalid_argument when a routed parent has an invalid
         *         rank, expert axis, or zero matrix geometry.
         */
        size_t logicalMatrixK(
            const GGUFTensorInfo &tensor,
            int expert_count)
        {
            if (tensor.dimensions.empty())
                return 0u;
            if (!isRoutedExpertParent(tensor.name))
                return static_cast<size_t>(tensor.dimensions.back());

            if (tensor.dimensions.size() != 3u || expert_count <= 0 ||
                tensor.dimensions[0] == 0u || tensor.dimensions[1] == 0u ||
                tensor.dimensions[2] != static_cast<uint64_t>(expert_count))
            {
                throw std::invalid_argument(
                    "Routed expert tensor has invalid [K, N, experts] geometry: " +
                    tensor.name);
            }
            return static_cast<size_t>(tensor.dimensions[0]);
        }

        int firstPositiveMetadataInt(
            const GGUFModel &model,
            const std::vector<std::string> &keys)
        {
            for (const auto &key : keys)
            {
                const auto it = model.metadata.find(key);
                if (it == model.metadata.end())
                    continue;
                const uint64_t value = it->second.asUInt64();
                if (value > 0)
                    return static_cast<int>(value);
            }
            return 0;
        }

    } // anonymous namespace

    ModelMemoryProfile ModelMemoryProfile::fromGGUF(const GGUFModel &model)
    {
        ModelMemoryProfile profile;

        profile.architecture = model.architecture;
        profile.n_layers = static_cast<int>(model.block_count);
        profile.d_model = static_cast<int>(model.embedding_length);
        profile.n_heads = static_cast<int>(model.head_count);
        profile.n_kv_heads = static_cast<int>(model.head_count_kv);
        profile.vocab_size = static_cast<int>(model.vocab_size);
        profile.max_seq_len = static_cast<int>(model.context_length);
        profile.expert_count = firstPositiveMetadataInt(
            model,
            {model.architecture + ".expert_count"});
        profile.expert_used_count = firstPositiveMetadataInt(
            model,
            {model.architecture + ".expert_used_count"});
        profile.expert_feed_forward_length = firstPositiveMetadataInt(
            model,
            {model.architecture + ".expert_feed_forward_length"});
        profile.expert_shared_feed_forward_length = firstPositiveMetadataInt(
            model,
            {model.architecture + ".expert_shared_feed_forward_length"});
        profile.mtp_layer_count = firstPositiveMetadataInt(
            model,
            {
                model.architecture + ".nextn_predict_layers",
                model.architecture + ".mtp_num_hidden_layers",
                model.architecture + ".mtp.num_hidden_layers",
                "mtp.num_hidden_layers",
                "mtp_num_hidden_layers",
            });
        profile.full_attention_interval = firstPositiveMetadataInt(
            model,
            {model.architecture + ".full_attention_interval"});
        profile.gdn_conv_kernel_size = firstPositiveMetadataInt(
            model,
            {model.architecture + ".ssm.conv_kernel"});
        profile.gdn_state_size = firstPositiveMetadataInt(
            model,
            {model.architecture + ".ssm.state_size"});
        profile.gdn_inner_size = firstPositiveMetadataInt(
            model,
            {model.architecture + ".ssm.inner_size"});
        profile.gdn_group_count = firstPositiveMetadataInt(
            model,
            {model.architecture + ".ssm.group_count"});
        profile.gdn_time_step_rank = firstPositiveMetadataInt(
            model,
            {model.architecture + ".ssm.time_step_rank"});

        // head_dim from key_length or computed
        if (model.key_length > 0)
        {
            profile.head_dim = static_cast<int>(model.key_length);
        }
        else if (model.head_count > 0)
        {
            profile.head_dim = static_cast<int>(model.embedding_length / model.head_count);
        }

        // d_ff: try metadata, else estimate as 4 * d_model (common default)
        auto ff_it = model.metadata.find("llama.feed_forward_length");
        if (ff_it == model.metadata.end())
            ff_it = model.metadata.find(model.architecture + ".feed_forward_length");
        if (ff_it != model.metadata.end())
        {
            profile.d_ff = static_cast<int>(ff_it->second.asUInt64());
        }
        else
        {
            // Estimate: for most models, d_ff is discoverable from tensor shapes
            // Look for ffn_gate or ffn_up weight to infer d_ff
            for (const auto &t : model.tensors)
            {
                if ((t.name.find("ffn_gate") != std::string::npos ||
                     t.name.find("ffn_up") != std::string::npos) &&
                    t.dimensions.size() == 2)
                {
                    profile.d_ff = static_cast<int>(t.dimensions[0]);
                    break;
                }
            }
            if (profile.d_ff == 0)
            {
                profile.d_ff = profile.d_model * 4; // Fallback heuristic
            }
        }

        // Build tensor inventory
        profile.total_native_bytes = 0;
        profile.tensors.reserve(model.tensors.size());

        for (const auto &t : model.tensors)
        {
            TensorSizeInfo info;
            info.name = t.name;
            info.native_bytes = static_cast<size_t>(t.size_bytes);
            info.quant_type = ggufTypeToString(t.type);
            info.elements = computeElements(t.dimensions);
            info.K = logicalMatrixK(t, profile.expert_count);
            info.layer_index = parseLayerIndex(t.name);

            profile.total_native_bytes += info.native_bytes;
            profile.tensors.push_back(std::move(info));
        }

        return profile;
    }

    size_t ModelMemoryProfile::weightBytesForLayers(int first_layer, int last_layer) const
    {
        size_t total = 0;
        for (const auto &t : tensors)
        {
            if (t.layer_index >= first_layer && t.layer_index <= last_layer)
            {
                total += t.native_bytes;
            }
        }
        return total;
    }

    size_t ModelMemoryProfile::embeddingBytes() const
    {
        for (const auto &t : tensors)
        {
            if (t.name.find("token_embd") != std::string::npos ||
                t.name.find("embed_tokens") != std::string::npos)
            {
                return t.native_bytes;
            }
        }
        return 0;
    }

    size_t ModelMemoryProfile::lmHeadBytes() const
    {
        for (const auto &t : tensors)
        {
            if (t.name == "output.weight" || t.name.find("lm_head") != std::string::npos)
            {
                return t.native_bytes;
            }
        }
        // Many models tie embedding and LM head weights
        return embeddingBytes();
    }

    size_t ModelMemoryProfile::normBytes() const
    {
        size_t total = 0;
        for (const auto &t : tensors)
        {
            if (t.layer_index == -1 &&
                (t.name.find("norm") != std::string::npos ||
                 t.name.find("ln_") != std::string::npos))
            {
                total += t.native_bytes;
            }
        }
        return total;
    }

    size_t ModelMemoryProfile::layerWeightBytes(int layer) const
    {
        size_t total = 0;
        for (const auto &t : tensors)
        {
            if (t.layer_index == layer)
            {
                total += t.native_bytes;
            }
        }
        return total;
    }

    // ================================================================
    // MPI serialization helpers (local to this TU)
    // ================================================================

    namespace
    {

        template <typename T>
        void writeVal(std::vector<uint8_t> &buf, T value)
        {
            const auto *p = reinterpret_cast<const uint8_t *>(&value);
            buf.insert(buf.end(), p, p + sizeof(T));
        }

        void writeStr(std::vector<uint8_t> &buf, const std::string &s)
        {
            writeVal<uint32_t>(buf, static_cast<uint32_t>(s.size()));
            buf.insert(buf.end(), s.begin(), s.end());
        }

        template <typename T>
        T readVal(const uint8_t *&ptr, const uint8_t *end)
        {
            if (ptr + sizeof(T) > end)
                throw std::runtime_error("ModelMemoryProfile deserialization: buffer underflow");
            T v;
            std::memcpy(&v, ptr, sizeof(T));
            ptr += sizeof(T);
            return v;
        }

        std::string readStr(const uint8_t *&ptr, const uint8_t *end)
        {
            uint32_t len = readVal<uint32_t>(ptr, end);
            if (ptr + len > end)
                throw std::runtime_error("ModelMemoryProfile deserialization: string buffer underflow");
            std::string s(reinterpret_cast<const char *>(ptr), len);
            ptr += len;
            return s;
        }

    } // anonymous namespace

    // ================================================================
    // ModelMemoryProfile::serialize / deserialize
    // ================================================================

    std::vector<uint8_t> ModelMemoryProfile::serialize() const
    {
        std::vector<uint8_t> buf;
        buf.reserve(4096);

        // Prefix scalars with a versioned identity so stale layouts fail hard.
        writeVal<uint32_t>(buf, kProfileWireMagic);
        writeVal<uint32_t>(buf, kProfileWireVersion);

        // Scalar fields
        writeStr(buf, architecture);
        writeVal<int32_t>(buf, n_layers);
        writeVal<int32_t>(buf, d_model);
        writeVal<int32_t>(buf, d_ff);
        writeVal<int32_t>(buf, n_heads);
        writeVal<int32_t>(buf, n_kv_heads);
        writeVal<int32_t>(buf, head_dim);
        writeVal<int32_t>(buf, vocab_size);
        writeVal<int32_t>(buf, max_seq_len);
        writeVal<int32_t>(buf, expert_count);
        writeVal<int32_t>(buf, expert_used_count);
        writeVal<int32_t>(buf, expert_feed_forward_length);
        writeVal<int32_t>(buf, expert_shared_feed_forward_length);
        writeVal<int32_t>(buf, mtp_layer_count);
        writeVal<int32_t>(buf, full_attention_interval);
        writeVal<int32_t>(buf, gdn_conv_kernel_size);
        writeVal<int32_t>(buf, gdn_state_size);
        writeVal<int32_t>(buf, gdn_inner_size);
        writeVal<int32_t>(buf, gdn_group_count);
        writeVal<int32_t>(buf, gdn_time_step_rank);
        writeVal<uint64_t>(buf, total_native_bytes);

        // Tensor inventory
        writeVal<uint32_t>(buf, static_cast<uint32_t>(tensors.size()));
        for (const auto &t : tensors)
        {
            writeStr(buf, t.name);
            writeVal<uint64_t>(buf, t.native_bytes);
            writeStr(buf, t.quant_type);
            writeVal<uint64_t>(buf, t.elements);
            writeVal<uint64_t>(buf, t.K);
            writeVal<int32_t>(buf, t.layer_index);
        }

        return buf;
    }

    ModelMemoryProfile ModelMemoryProfile::deserialize(const uint8_t *data, size_t size)
    {
        ModelMemoryProfile p;
        const uint8_t *ptr = data;
        const uint8_t *end = data + size;

        const uint32_t magic = readVal<uint32_t>(ptr, end);
        if (magic != kProfileWireMagic)
        {
            throw std::runtime_error(
                "ModelMemoryProfile deserialization: invalid wire-format magic");
        }
        const uint32_t version = readVal<uint32_t>(ptr, end);
        if (version != kProfileWireVersion)
        {
            throw std::runtime_error(
                "ModelMemoryProfile deserialization: unsupported wire-format version " +
                std::to_string(version));
        }

        p.architecture = readStr(ptr, end);
        p.n_layers = readVal<int32_t>(ptr, end);
        p.d_model = readVal<int32_t>(ptr, end);
        p.d_ff = readVal<int32_t>(ptr, end);
        p.n_heads = readVal<int32_t>(ptr, end);
        p.n_kv_heads = readVal<int32_t>(ptr, end);
        p.head_dim = readVal<int32_t>(ptr, end);
        p.vocab_size = readVal<int32_t>(ptr, end);
        p.max_seq_len = readVal<int32_t>(ptr, end);
        p.expert_count = readVal<int32_t>(ptr, end);
        p.expert_used_count = readVal<int32_t>(ptr, end);
        p.expert_feed_forward_length = readVal<int32_t>(ptr, end);
        p.expert_shared_feed_forward_length = readVal<int32_t>(ptr, end);
        p.mtp_layer_count = readVal<int32_t>(ptr, end);
        p.full_attention_interval = readVal<int32_t>(ptr, end);
        p.gdn_conv_kernel_size = readVal<int32_t>(ptr, end);
        p.gdn_state_size = readVal<int32_t>(ptr, end);
        p.gdn_inner_size = readVal<int32_t>(ptr, end);
        p.gdn_group_count = readVal<int32_t>(ptr, end);
        p.gdn_time_step_rank = readVal<int32_t>(ptr, end);
        p.total_native_bytes = readVal<uint64_t>(ptr, end);

        uint32_t n_tensors = readVal<uint32_t>(ptr, end);
        p.tensors.reserve(n_tensors);
        for (uint32_t i = 0; i < n_tensors; ++i)
        {
            TensorSizeInfo t;
            t.name = readStr(ptr, end);
            t.native_bytes = readVal<uint64_t>(ptr, end);
            t.quant_type = readStr(ptr, end);
            t.elements = readVal<uint64_t>(ptr, end);
            t.K = readVal<uint64_t>(ptr, end);
            t.layer_index = readVal<int32_t>(ptr, end);
            p.tensors.push_back(std::move(t));
        }

        return p;
    }

} // namespace llaminar2
