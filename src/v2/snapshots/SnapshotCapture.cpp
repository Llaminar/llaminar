/**
 * @file SnapshotCapture.cpp
 * @brief Implementation of snapshot capture logic
 *
 * Extracted from DeviceGraphOrchestrator.h (Phase 2 of DGO refactor).
 */

#include "SnapshotCapture.h"

namespace llaminar2
{

    // =========================================================================
    // Stage capture routing
    // =========================================================================

    void SnapshotCapture::captureStage(const std::string &name, const StageDumpInfo &dump)
    {
        LOG_TRACE("[Snapshot] Callback invoked for stage: " << name
                                                            << " outputs.size=" << dump.outputs.size());

        // Handle fused QKV stage — split into separate Q, K, V snapshots
        if (name.find("_qkv_proj") != std::string::npos)
        {
            size_t qkv_pos = name.find("_qkv_proj");
            std::string prefix = name.substr(0, qkv_pos);

            if (dump.outputs.size() >= 3)
            {
                storeOutput(prefix + "_Q_PROJECTION", dump.outputs[0]);
                storeOutput(prefix + "_K_PROJECTION", dump.outputs[1]);
                storeOutput(prefix + "_V_PROJECTION", dump.outputs[2]);
            }
            return;
        }

        // Handle fused Gate/Up stage — split into separate GATE and UP snapshots
        if (name.find("_gate_up") != std::string::npos)
        {
            size_t pos = name.find("_gate_up");
            std::string prefix = name.substr(0, pos);

            if (dump.outputs.size() >= 2)
            {
                storeOutput(prefix + "_FFN_GATE", dump.outputs[0]);
                storeOutput(prefix + "_FFN_UP", dump.outputs[1]);
            }
            return;
        }

        // Handle fused RoPE stage — captures Q_ROPE and K_ROPE
        if (name.find("_rope") != std::string::npos &&
            name.find("_q_rope") == std::string::npos &&
            name.find("_k_rope") == std::string::npos)
        {
            size_t pos = name.find("_rope");
            std::string prefix = name.substr(0, pos);

            if (dump.outputs.size() >= 2)
            {
                storeOutput(prefix + "_Q_ROPE", dump.outputs[0]);
                storeOutput(prefix + "_K_ROPE", dump.outputs[1]);
            }
            return;
        }

        // Handle GDN 4-way projection — split into QKV, Z, alpha, beta snapshots
        if (name.find("_gdn_proj") != std::string::npos)
        {
            size_t pos = name.find("_gdn_proj");
            std::string prefix = name.substr(0, pos);

            // outputs: [0]=output_qkv, [1]=output_z, [2]=output_a, [3]=output_b
            if (dump.outputs.size() >= 1 && dump.outputs[0].data)
                storeOutput(prefix + "_QKV_PROJECTION", dump.outputs[0]);
            if (dump.outputs.size() >= 2 && dump.outputs[1].data)
                storeOutput(prefix + "_GDN_Z_PROJECTION", dump.outputs[1]);
            // alpha and beta are small per-head tensors, skip for now
            return;
        }

        // Handle lm_head_allgather — overwrites partial LM_HEAD with full vocab
        if (name == "lm_head_allgather")
        {
            if (!dump.outputs.empty() && dump.outputs[0].data)
            {
                const auto &out = dump.outputs[0];
                auto data = extractFp32FromOutput(out);
                LOG_DEBUG("[Snapshot] lm_head_allgather handler: storing as LM_HEAD (overwriting partial), count=" << data.size());
                if (!data.empty())
                    snapshots_["LM_HEAD"] = {std::move(data), out.rows, out.cols};
            }
            return;
        }

        // Handle FusedResidualNormStage — store outputs[1] (norm_output), not outputs[0] (residual)
        if ((name.find("_attn_norm") != std::string::npos ||
             name.find("_ffn_norm") != std::string::npos) &&
            dump.outputs.size() >= 2)
        {
            std::string key = convertStageNameToSnapshotKey(name);

            if (dump.outputs[1].data)
            {
                auto data = extractFp32FromOutput(dump.outputs[1]);
                LOG_DEBUG("[Snapshot] FusedResidualNorm: storing norm_output as key="
                          << key << " count=" << data.size());
                if (!data.empty())
                    snapshots_[key] = {std::move(data), dump.outputs[1].rows, dump.outputs[1].cols};
            }
            return;
        }

        // Standard single-output stages
        LOG_DEBUG("[Snapshot] Standard path: stage=" << name
                                                     << " outputs.size=" << dump.outputs.size()
                                                     << " out[0].data=" << (dump.outputs.empty() ? nullptr : dump.outputs[0].data));
        if (!dump.outputs.empty() && dump.outputs[0].data)
        {
            const auto &out = dump.outputs[0];
            auto data = extractFp32FromOutput(out);
            std::string key = convertStageNameToSnapshotKey(name);
            LOG_DEBUG("[Snapshot] Storing key=" << key << " count=" << data.size());

            if (data.size() >= 8 && key == "EMBEDDING")
            {
                LOG_DEBUG("[Snapshot] " << key << " first 8 values: "
                                        << data[0] << "," << data[1] << "," << data[2] << "," << data[3] << ","
                                        << data[4] << "," << data[5] << "," << data[6] << "," << data[7]);
            }

            if (!data.empty())
                snapshots_[key] = {std::move(data), out.rows, out.cols};
        }
    }

    // =========================================================================
    // FP32 extraction from various tensor formats
    // =========================================================================

    std::vector<float> SnapshotCapture::extractFp32FromOutput(const StageDumpInfo::OutputBuffer &out)
    {
        if (!out.data)
            return {};

        size_t count = out.rows * out.cols;
        if (count == 0)
            return {};

        std::vector<float> data(count);
        std::string dtype_str = out.dtype ? out.dtype : "FP32";

        LOG_TRACE("[extractFp32FromOutput] name=" << (out.name ? out.name : "?")
                                                  << " dtype=" << dtype_str
                                                  << " rows=" << out.rows << " cols=" << out.cols);

        // FP32: direct copy
        if (dtype_str == "FP32")
        {
            std::memcpy(data.data(), out.data, count * sizeof(float));
            return data;
        }

        // Q8_1: dequantize blocks
        if (dtype_str == "Q8_1")
        {
            const Q8_1Block *blocks = static_cast<const Q8_1Block *>(out.data);
            constexpr int BLOCK_SIZE = 32;
            size_t num_blocks = (count + BLOCK_SIZE - 1) / BLOCK_SIZE;

            for (size_t b = 0; b < num_blocks; ++b)
            {
                const Q8_1Block &block = blocks[b];
                float scale = fp16_to_fp32(block.d);
                for (int i = 0; i < BLOCK_SIZE && b * BLOCK_SIZE + i < count; ++i)
                {
                    data[b * BLOCK_SIZE + i] = static_cast<float>(block.qs[i]) * scale;
                }
            }
            return data;
        }

        // Q16_1 variants: dequantize blocks (block sizes 32, 64, 128)
        if (dtype_str.find("Q16_1") == 0)
        {
            int block_size = 32;
            if (dtype_str.find("_64") != std::string::npos)
                block_size = 64;
            else if (dtype_str.find("_128") != std::string::npos)
                block_size = 128;

            const Q16_1Block *blocks = static_cast<const Q16_1Block *>(out.data);
            size_t num_blocks = (count + block_size - 1) / block_size;

            for (size_t b = 0; b < num_blocks; ++b)
            {
                const Q16_1Block &block = blocks[b];
                float scale = fp16_to_fp32(block.d);
                for (int i = 0; i < block_size && b * block_size + i < count; ++i)
                {
                    data[b * block_size + i] = static_cast<float>(block.qs[i]) * scale;
                }
            }
            return data;
        }

        // BF16 or FP16: convert to FP32
        if (dtype_str == "BF16" || dtype_str == "FP16")
        {
            const uint16_t *half_data = static_cast<const uint16_t *>(out.data);
            for (size_t i = 0; i < count; ++i)
            {
                if (dtype_str == "BF16")
                    data[i] = simd::bf16_to_fp32(half_data[i]);
                else
                    data[i] = simd::fp16_to_fp32(half_data[i]);
            }
            return data;
        }

        // Unknown dtype — warn and try FP32 (may be garbage)
        LOG_WARN("[extractFp32FromOutput] Unknown dtype '" << dtype_str << "', assuming FP32");
        std::memcpy(data.data(), out.data, count * sizeof(float));
        return data;
    }

    // =========================================================================
    // Stage name → snapshot key conversion
    // =========================================================================

    std::string SnapshotCapture::convertStageNameToSnapshotKey(const std::string &stage_name)
    {
        // Ordered vector: longest/most-specific suffixes FIRST to ensure correct
        // prefix extraction. E.g. "_gdn_wo_allreduce" must match before "_wo_allreduce"
        // so the prefix is "layerN" (not "layerN_gdn").
        static const std::vector<std::pair<std::string, std::string>> suffix_map = {
            // GDN (Gated Delta Net) linear attention stages — longest suffixes first
            {"_gdn_wo_allreduce", "_ATTENTION_OUTPUT"},
            {"_gdn_out_proj", "_ATTENTION_OUTPUT"},
            {"_gdn_proj", "_QKV_PROJECTION"},
            {"_gdn_recurrence", "_GDN_DELTA_RULE_OUTPUT"},
            {"_gated_norm", "_GDN_NORM_GATE_OUTPUT"},
            // Standard attention stages
            {"_attn_norm", "_ATTENTION_NORM"},
            {"_attn_residual", "_ATTENTION_RESIDUAL"},
            {"_wo_allreduce", "_ATTENTION_OUTPUT"},
            {"_wo_proj", "_ATTENTION_OUTPUT"},
            {"_q_norm", "_Q_NORM"},
            {"_k_norm", "_K_NORM"},
            {"_q_proj", "_Q_PROJECTION"},
            {"_k_proj", "_K_PROJECTION"},
            {"_v_proj", "_V_PROJECTION"},
            {"_q_rope", "_Q_ROPE"},
            {"_k_rope", "_K_ROPE"},
            {"_attention", "_ATTENTION_CONTEXT"},
            // FFN stages
            {"_down_allreduce", "_FFN_DOWN"},
            {"_ffn_norm", "_FFN_NORM"},
            {"_ffn_gate", "_FFN_GATE"},
            {"_ffn_up", "_FFN_UP"},
            {"_swiglu", "_FFN_SWIGLU"},
            {"_down_proj", "_FFN_DOWN"},
            {"_ffn_residual", "_FFN_RESIDUAL"},
        };

        // Global stages
        if (stage_name == "embedding")
            return "EMBEDDING";
        if (stage_name == "final_norm")
            return "FINAL_NORM";
        if (stage_name == "lm_head")
            return "LM_HEAD";

        // Layer-specific stages: extract layer prefix and convert suffix.
        // Uses ordered iteration so longer/more-specific suffixes match first.
        for (const auto &[suffix, replacement] : suffix_map)
        {
            size_t pos = stage_name.find(suffix);
            if (pos != std::string::npos)
            {
                std::string prefix = stage_name.substr(0, pos);
                return prefix + replacement;
            }
        }

        // Fallback: return original name (uppercase)
        std::string result = stage_name;
        for (char &c : result)
            c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
        return result;
    }

    // =========================================================================
    // Private helpers
    // =========================================================================

    void SnapshotCapture::storeOutput(const std::string &key, const StageDumpInfo::OutputBuffer &out)
    {
        if (!out.data)
            return;
        auto data = extractFp32FromOutput(out);
        if (!data.empty())
            snapshots_[key] = {std::move(data), out.rows, out.cols};
    }

} // namespace llaminar2
