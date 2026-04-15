/**
 * @file SharedWeightPool.cpp
 * @brief Implementation of SharedWeightPool — single GGUF load, many views
 *
 * @author David Sanftenberg
 * @date April 2026
 */

#include "SharedWeightPool.h"
#include "../utils/Logger.h"
#include <regex>

namespace llaminar2
{

    SharedWeightPool::SharedWeightPool() = default;
    SharedWeightPool::~SharedWeightPool() = default;

    std::unique_ptr<SharedWeightPool> SharedWeightPool::create()
    {
        return std::make_unique<SharedWeightPool>();
    }

    bool SharedWeightPool::loadFromGGUF(const std::string &model_path, bool use_mmap)
    {
        if (loaded_)
        {
            LOG_WARN("[SharedWeightPool] Already loaded, ignoring duplicate load request");
            return true;
        }

        LOG_INFO("[SharedWeightPool] Loading model from: " << model_path);

        // Configure mmap
        loader_.setUseMmap(use_mmap);

        // Parse GGUF header + metadata + tensor info
        if (!loader_.loadModel(model_path))
        {
            LOG_ERROR("[SharedWeightPool] Failed to load model: " << model_path);
            return false;
        }

        // Extract model metadata from parsed GGUF
        extractMetadata();
        metadata_.model_path = model_path;

        // Load all weight tensors into the pool
        if (!loadAllTensors())
        {
            LOG_ERROR("[SharedWeightPool] Failed to load weight tensors");
            return false;
        }

        loaded_ = true;

        LOG_INFO("[SharedWeightPool] Loaded " << tensors_.size() << " tensors"
                                              << " (" << (totalHostBytes() / (1024 * 1024)) << " MB)"
                                              << " for " << metadata_.architecture
                                              << " model with " << metadata_.n_layers << " layers");

        return true;
    }

    void SharedWeightPool::extractMetadata()
    {
        const auto &model = loader_.getModel();

        metadata_.architecture = model.architecture;
        metadata_.n_layers = static_cast<int>(loader_.blockCount());
        metadata_.d_model = static_cast<int>(loader_.embeddingLength());
        metadata_.n_heads = static_cast<int>(loader_.headCount());
        metadata_.n_kv_heads = static_cast<int>(loader_.headCountKV());
        metadata_.d_ff = static_cast<int>(loader_.feedForwardLength());
        metadata_.vocab_size = static_cast<int>(loader_.vocabSize());
        metadata_.context_length = static_cast<int>(loader_.contextLength());

        // Head dim: explicit from GGUF or computed
        int key_len = static_cast<int>(loader_.keyLength());
        metadata_.head_dim = (key_len > 0) ? key_len : (metadata_.d_model / metadata_.n_heads);

        // GDN dimensions (if available in GGUF metadata)
        auto getMetaUInt32 = [&](const std::string &key) -> int
        {
            auto it = model.metadata.find(key);
            if (it != model.metadata.end())
            {
                return static_cast<int>(it->second.asUInt32());
            }
            return 0;
        };

        // Standard GGUF keys for SSM/GDN parameters
        std::string arch_prefix = metadata_.architecture;
        metadata_.gdn_n_k_heads = getMetaUInt32(arch_prefix + ".ssm.group_count");
        metadata_.gdn_n_v_heads = getMetaUInt32(arch_prefix + ".ssm.num_heads");
        metadata_.gdn_d_state = getMetaUInt32(arch_prefix + ".ssm.state_size");
    }

    bool SharedWeightPool::loadAllTensors()
    {
        auto names = loader_.tensorNames();

        if (names.empty())
        {
            LOG_ERROR("[SharedWeightPool] No tensors found in model");
            return false;
        }

        LOG_DEBUG("[SharedWeightPool] Loading " << names.size() << " tensors...");

        tensors_.reserve(names.size());

        for (const auto &name : names)
        {
            // Load tensor to host (CPU) with native precision (no dequantization)
            auto tensor = loader_.loadTensor(name, DeviceId::cpu(), WeightPrecision::NATIVE);
            if (!tensor)
            {
                LOG_WARN("[SharedWeightPool] Failed to load tensor: " << name << " (skipping)");
                continue;
            }

            tensors_.emplace(name, std::move(tensor));
        }

        LOG_DEBUG("[SharedWeightPool] Successfully loaded " << tensors_.size()
                                                            << " of " << names.size() << " tensors");

        return !tensors_.empty();
    }

    WeightViewSet SharedWeightPool::createViewSet(
        int first_layer, int last_layer,
        bool has_embedding, bool has_lm_head) const
    {
        WeightViewSet views(first_layer, last_layer, has_embedding, has_lm_head);

        for (const auto &[name, tensor] : tensors_)
        {
            if (!isWeightInLayerRange(name, first_layer, last_layer, has_embedding, has_lm_head))
            {
                continue;
            }

            WeightView view;
            view.name = name;
            view.tensor = tensor; // Shared pointer — pool retains primary ownership
            view.layer_index = extractLayerIndex(name);
            view.is_gemm_weight = !isNonGemmWeight(name);

            views.addView(std::move(view));
        }

        LOG_DEBUG("[SharedWeightPool] Created view set for layers ["
                  << first_layer << ", " << last_layer << ")"
                  << " embedding=" << has_embedding << " lm_head=" << has_lm_head
                  << " → " << views.size() << " weights");

        return views;
    }

    std::shared_ptr<TensorBase> SharedWeightPool::getTensor(const std::string &name) const
    {
        auto it = tensors_.find(name);
        return (it != tensors_.end()) ? it->second : nullptr;
    }

    bool SharedWeightPool::hasTensor(const std::string &name) const
    {
        return tensors_.find(name) != tensors_.end();
    }

    std::vector<std::string> SharedWeightPool::tensorNames() const
    {
        std::vector<std::string> names;
        names.reserve(tensors_.size());
        for (const auto &[name, _] : tensors_)
        {
            names.push_back(name);
        }
        return names;
    }

    size_t SharedWeightPool::tensorCount() const
    {
        return tensors_.size();
    }

    size_t SharedWeightPool::totalHostBytes() const
    {
        size_t total = 0;
        for (const auto &[_, tensor] : tensors_)
        {
            total += tensor->size_bytes();
        }
        return total;
    }

    bool SharedWeightPool::isGemmWeight(const std::string &name) const
    {
        return !isNonGemmWeight(name);
    }

    // =========================================================================
    // Static Helpers
    // =========================================================================

    bool SharedWeightPool::isWeightInLayerRange(
        const std::string &name,
        int first_layer, int last_layer,
        bool has_embedding, bool has_lm_head)
    {
        // Handle special weights (embedding, output norm, LM head)
        if (name == "token_embd.weight")
        {
            // Allow embedding through if this stage owns the LM head too —
            // tied-embedding models (e.g. Qwen3.5) have no output.weight and
            // the LM head falls back to token_embd.weight.
            return has_embedding || has_lm_head;
        }
        if (name == "output_norm.weight" || name == "output.weight")
        {
            return has_lm_head;
        }

        // Extract layer index from "blk.N.xxx" pattern
        int layer_idx = extractLayerIndex(name);
        if (layer_idx >= 0)
        {
            // Layer range is [first, last) - first inclusive, last exclusive
            return layer_idx >= first_layer && layer_idx < last_layer;
        }

        // Unknown weight pattern - include by default (e.g., custom weights)
        return true;
    }

    int SharedWeightPool::extractLayerIndex(const std::string &name)
    {
        // Pattern: blk.{layer_idx}.{component}
        static const std::regex layer_pattern(R"(blk\.(\d+)\.)");
        std::smatch match;
        if (std::regex_search(name, match, layer_pattern))
        {
            return std::stoi(match[1].str());
        }
        return -1;
    }

    // =========================================================================
    // Non-GEMM weight detection (mirrors GraphSchema::WeightShardingConfig::isNonGemmWeight)
    // =========================================================================

    bool SharedWeightPool::isNonGemmWeight(const std::string &name)
    {
        // Norms are 1D
        if (name.find("_norm.weight") != std::string::npos)
            return true;
        // Biases are 1D
        if (name.find(".bias") != std::string::npos)
            return true;
        // Embeddings are used directly
        if (name.find("token_embd") != std::string::npos)
            return true;
        // SSM/GDN convolution kernels are not GEMM weights
        if (name.find("ssm_conv1d") != std::string::npos)
            return true;
        // SSM/GDN per-head scalar parameters (e.g. ssm_a, ssm_dt) without .weight suffix
        if (name.find(".ssm_a") != std::string::npos && name.find(".weight") == std::string::npos)
            return true;
        return false;
    }

} // namespace llaminar2
