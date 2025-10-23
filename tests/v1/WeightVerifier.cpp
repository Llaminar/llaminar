/**
 * @file WeightVerifier.cpp
 * @brief Implementation of weight verification utility
 * @author David Sanftenberg
 */

#include "WeightVerifier.h"
#include "Logger.h"
#include <cmath>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <filesystem>

namespace llaminar
{

    std::string WeightVerificationResult::toString() const
    {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(6);

        if (passed)
        {
            oss << "[PASS] " << details
                << " (max_diff=" << max_abs_diff
                << ", rel_l2=" << rel_l2_error << ")";
        }
        else
        {
            oss << "[FAIL] " << details
                << " (max_diff=" << max_abs_diff
                << ", mean_diff=" << mean_abs_diff
                << ", rel_l2=" << rel_l2_error
                << ", elements=" << total_elements << ")";
        }

        return oss.str();
    }

    WeightVerifier::WeightVerifier(QwenModelWeightsProvider *provider,
                                   const std::string &snapshot_dir,
                                   float abs_tol,
                                   float rel_tol)
        : provider_(provider), snapshot_dir_(snapshot_dir),
          abs_tol_(abs_tol), rel_tol_(rel_tol), verbose_(false)
    {
        if (!provider_)
        {
            throw std::invalid_argument("WeightVerifier: provider cannot be nullptr");
        }

        if (!std::filesystem::exists(snapshot_dir_))
        {
            throw std::runtime_error("WeightVerifier: snapshot directory does not exist: " +
                                     snapshot_dir_);
        }

        LOG_INFO("[WeightVerifier] Initialized with snapshot_dir=" << snapshot_dir_
                                                                   << " abs_tol=" << abs_tol_
                                                                   << " rel_tol=" << rel_tol_);
    }

    bool WeightVerifier::loadPyTorchWeight(int layer, const std::string &weight_type,
                                           NpyArray &out_array)
    {
        std::ostringstream filename;
        filename << snapshot_dir_ << "/layer" << layer << "_" << weight_type << "_WEIGHT.npy";

        std::string filepath = filename.str();

        if (!std::filesystem::exists(filepath))
        {
            LOG_ERROR("[WeightVerifier] PyTorch weight file not found: " << filepath);
            return false;
        }

        if (!NpzLoader::load_npy(filepath, out_array))
        {
            LOG_ERROR("[WeightVerifier] Failed to load PyTorch weight: " << filepath);
            return false;
        }

        if (verbose_)
        {
            LOG_INFO("[WeightVerifier] Loaded PyTorch " << weight_type << " layer " << layer
                                                        << " shape=[" << out_array.shape[0] << "," << out_array.shape[1] << "]"
                                                        << " dtype=" << out_array.dtype);
        }

        return true;
    }

    WeightVerificationResult WeightVerifier::computeStatistics(
        const float *pytorch_data,
        const float *llaminar_data,
        size_t count,
        const std::string &weight_name)
    {
        WeightVerificationResult result;
        result.total_elements = count;

        if (count == 0)
        {
            return WeightVerificationResult::failure(weight_name + ": zero elements");
        }

        // Compute differences
        std::vector<float> diffs(count);
        float sum_abs_diff = 0.0f;
        float max_diff = 0.0f;

        for (size_t i = 0; i < count; ++i)
        {
            float diff = std::abs(pytorch_data[i] - llaminar_data[i]);
            diffs[i] = diff;
            sum_abs_diff += diff;
            max_diff = std::max(max_diff, diff);
        }

        result.max_abs_diff = max_diff;
        result.mean_abs_diff = sum_abs_diff / count;

        // Compute L2 norms
        float pytorch_l2 = 0.0f;
        float diff_l2 = 0.0f;

        for (size_t i = 0; i < count; ++i)
        {
            pytorch_l2 += pytorch_data[i] * pytorch_data[i];
            diff_l2 += diffs[i] * diffs[i];
        }

        pytorch_l2 = std::sqrt(pytorch_l2);
        diff_l2 = std::sqrt(diff_l2);

        // Relative L2 error
        if (pytorch_l2 > 1e-12f)
        {
            result.rel_l2_error = diff_l2 / pytorch_l2;
        }
        else
        {
            result.rel_l2_error = (diff_l2 < 1e-12f) ? 0.0f : std::numeric_limits<float>::infinity();
        }

        // Check tolerances
        result.passed = (result.max_abs_diff < abs_tol_) && (result.rel_l2_error < rel_tol_);

        // Build details string
        std::ostringstream oss;
        oss << weight_name;
        if (result.passed)
        {
            oss << " OK";
        }
        else
        {
            oss << " MISMATCH";
        }
        result.details = oss.str();

        if (verbose_ || !result.passed)
        {
            LOG_INFO("[WeightVerifier] " << result.toString());
        }

        return result;
    }

    bool WeightVerifier::extractLocalSlice(const NpyArray &pytorch_full,
                                           const std::string &weight_type,
                                           std::vector<float> &out_local)
    {
        if (!provider_->isWeightSliced(weight_type))
        {
            LOG_ERROR("[WeightVerifier] extractLocalSlice called on non-sliced weight: " << weight_type);
            return false;
        }

        auto [offset, count] = provider_->getLocalSliceInfo(weight_type);

        if (pytorch_full.shape.size() != 2)
        {
            LOG_ERROR("[WeightVerifier] Expected 2D PyTorch weight, got " << pytorch_full.shape.size() << "D");
            return false;
        }

        size_t out_features = pytorch_full.shape[0];
        size_t in_features = pytorch_full.shape[1];
        int head_dim = provider_->getConfig().head_dim;

        // O weight: column slicing (slice in_features dimension)
        if (weight_type == "O")
        {
            // PyTorch O layout: [d_model, num_heads * head_dim]
            // Slice columns [offset*head_dim, (offset+count)*head_dim)
            int col_offset = offset * head_dim;
            int col_count = count * head_dim;

            if (col_offset + col_count > static_cast<int>(in_features))
            {
                LOG_ERROR("[WeightVerifier] Column slice out of bounds: offset=" << col_offset
                                                                                 << " count=" << col_count
                                                                                 << " in_features=" << in_features);
                return false;
            }

            // Extract columns (need to copy non-contiguous data)
            out_local.resize(out_features * col_count);
            for (size_t row = 0; row < out_features; ++row)
            {
                const float *src_row = pytorch_full.data.data() + row * in_features + col_offset;
                float *dst_row = out_local.data() + row * col_count;
                std::copy(src_row, src_row + col_count, dst_row);
            }

            if (verbose_)
            {
                LOG_INFO("[WeightVerifier] Extracted column slice for " << weight_type
                                                                        << " rank=" << provider_->getRank()
                                                                        << " heads=[" << offset << "," << (offset + count) << ")"
                                                                        << " cols=[" << col_offset << "," << (col_offset + col_count) << ")"
                                                                        << " shape=[" << out_features << "," << col_count << "]"
                                                                        << " size=" << out_local.size());
            }
        }
        else
        {
            // Q/K/V weights: row slicing (slice out_features dimension)
            // PyTorch weight layout: [out_features, in_features]
            // For K/V/Q: [num_heads * head_dim, d_model]
            // Slicing: extract rows [offset*head_dim, (offset+count)*head_dim)

            int row_offset = offset * head_dim;
            int row_count = count * head_dim;

            if (row_offset + row_count > static_cast<int>(out_features))
            {
                LOG_ERROR("[WeightVerifier] Row slice out of bounds: offset=" << row_offset
                                                                              << " count=" << row_count
                                                                              << " out_features=" << out_features);
                return false;
            }

            // Extract rows (contiguous data)
            out_local.resize(row_count * in_features);
            const float *src = pytorch_full.data.data() + row_offset * in_features;
            std::copy(src, src + row_count * in_features, out_local.begin());

            if (verbose_)
            {
                LOG_INFO("[WeightVerifier] Extracted row slice for " << weight_type
                                                                     << " rank=" << provider_->getRank()
                                                                     << " heads=[" << offset << "," << (offset + count) << ")"
                                                                     << " rows=[" << row_offset << "," << (row_offset + row_count) << ")"
                                                                     << " size=" << out_local.size());
            }
        }

        return true;
    }

    WeightVerificationResult WeightVerifier::compareReplicatedWeight(
        const NpyArray &pytorch_weight,
        std::shared_ptr<TensorBase> llaminar_weight,
        const std::string &weight_name)
    {
        if (!llaminar_weight)
        {
            return WeightVerificationResult::failure(weight_name + ": Llaminar weight is nullptr");
        }

        size_t pytorch_size = pytorch_weight.data.size();
        size_t llaminar_size = llaminar_weight->size();

        if (pytorch_size != llaminar_size)
        {
            std::ostringstream oss;
            oss << weight_name << ": size mismatch (PyTorch=" << pytorch_size
                << " vs Llaminar=" << llaminar_size << ")";
            return WeightVerificationResult::failure(oss.str());
        }

        return computeStatistics(pytorch_weight.data.data(),
                                 llaminar_weight->data(),
                                 pytorch_size,
                                 weight_name);
    }

    WeightVerificationResult WeightVerifier::compareSlicedWeight(
        const NpyArray &pytorch_full_weight,
        std::shared_ptr<TensorBase> llaminar_local_weight,
        const std::string &weight_type,
        const std::string &weight_name)
    {
        if (!llaminar_local_weight)
        {
            return WeightVerificationResult::failure(weight_name + ": Llaminar weight is nullptr");
        }

        // Extract local slice from PyTorch weight
        std::vector<float> pytorch_local;
        if (!extractLocalSlice(pytorch_full_weight, weight_type, pytorch_local))
        {
            return WeightVerificationResult::failure(weight_name + ": failed to extract local slice");
        }

        size_t llaminar_size = llaminar_local_weight->size();
        if (pytorch_local.size() != llaminar_size)
        {
            std::ostringstream oss;
            oss << weight_name << ": local slice size mismatch (PyTorch=" << pytorch_local.size()
                << " vs Llaminar=" << llaminar_size << ")";
            return WeightVerificationResult::failure(oss.str());
        }

        return computeStatistics(pytorch_local.data(),
                                 llaminar_local_weight->data(),
                                 pytorch_local.size(),
                                 weight_name);
    }

    WeightVerificationResult WeightVerifier::verifyQueryWeight(int layer)
    {
        NpyArray pytorch_q;
        if (!loadPyTorchWeight(layer, "Q", pytorch_q))
        {
            return WeightVerificationResult::failure("Layer " + std::to_string(layer) + " Q: load failed");
        }

        auto llaminar_q = provider_->getQueryWeight(layer);
        std::string name = "Layer " + std::to_string(layer) + " Q";

        if (provider_->isWeightSliced("Q"))
        {
            return compareSlicedWeight(pytorch_q, llaminar_q, "Q", name);
        }
        else
        {
            return compareReplicatedWeight(pytorch_q, llaminar_q, name);
        }
    }

    WeightVerificationResult WeightVerifier::verifyKeyWeight(int layer)
    {
        NpyArray pytorch_k;
        if (!loadPyTorchWeight(layer, "K", pytorch_k))
        {
            return WeightVerificationResult::failure("Layer " + std::to_string(layer) + " K: load failed");
        }

        auto llaminar_k = provider_->getKeyWeight(layer);
        std::string name = "Layer " + std::to_string(layer) + " K";

        if (provider_->isWeightSliced("K"))
        {
            return compareSlicedWeight(pytorch_k, llaminar_k, "K", name);
        }
        else
        {
            return compareReplicatedWeight(pytorch_k, llaminar_k, name);
        }
    }

    WeightVerificationResult WeightVerifier::verifyValueWeight(int layer)
    {
        NpyArray pytorch_v;
        if (!loadPyTorchWeight(layer, "V", pytorch_v))
        {
            return WeightVerificationResult::failure("Layer " + std::to_string(layer) + " V: load failed");
        }

        auto llaminar_v = provider_->getValueWeight(layer);
        std::string name = "Layer " + std::to_string(layer) + " V";

        if (provider_->isWeightSliced("V"))
        {
            return compareSlicedWeight(pytorch_v, llaminar_v, "V", name);
        }
        else
        {
            return compareReplicatedWeight(pytorch_v, llaminar_v, name);
        }
    }

    WeightVerificationResult WeightVerifier::verifyOutputWeight(int layer)
    {
        NpyArray pytorch_o;
        if (!loadPyTorchWeight(layer, "O", pytorch_o))
        {
            return WeightVerificationResult::failure("Layer " + std::to_string(layer) + " O: load failed");
        }

        auto llaminar_o = provider_->getOutputWeight(layer);
        std::string name = "Layer " + std::to_string(layer) + " O";

        // O is typically replicated (not sliced)
        if (provider_->isWeightSliced("O"))
        {
            return compareSlicedWeight(pytorch_o, llaminar_o, "O", name);
        }
        else
        {
            return compareReplicatedWeight(pytorch_o, llaminar_o, name);
        }
    }

    WeightVerificationResult WeightVerifier::verifyLayerWeights(int layer)
    {
        // Verify Q, K, V, O in sequence
        // Return first failure, or success if all pass

        auto q_result = verifyQueryWeight(layer);
        if (!q_result.passed)
            return q_result;

        auto k_result = verifyKeyWeight(layer);
        if (!k_result.passed)
            return k_result;

        auto v_result = verifyValueWeight(layer);
        if (!v_result.passed)
            return v_result;

        auto o_result = verifyOutputWeight(layer);
        if (!o_result.passed)
            return o_result;

        // All passed
        WeightVerificationResult combined;
        combined.passed = true;
        combined.details = "Layer " + std::to_string(layer) + " all weights OK";
        return combined;
    }

    WeightVerificationResult WeightVerifier::verifyAllWeights()
    {
        int num_layers = provider_->getNumLayers();
        int failed_count = 0;
        std::vector<int> failed_layers;

        for (int layer = 0; layer < num_layers; ++layer)
        {
            auto result = verifyLayerWeights(layer);
            if (!result.passed)
            {
                failed_count++;
                failed_layers.push_back(layer);
                LOG_ERROR("[WeightVerifier] Layer " << layer << " verification failed: " << result.toString());
            }
        }

        if (failed_count == 0)
        {
            WeightVerificationResult success;
            success.passed = true;
            success.details = "All " + std::to_string(num_layers) + " layers verified successfully";
            LOG_INFO("[WeightVerifier] " << success.details);
            return success;
        }
        else
        {
            std::ostringstream oss;
            oss << failed_count << " of " << num_layers << " layers failed (layers: ";
            for (size_t i = 0; i < failed_layers.size(); ++i)
            {
                oss << failed_layers[i];
                if (i + 1 < failed_layers.size())
                    oss << ",";
            }
            oss << ")";
            return WeightVerificationResult::failure(oss.str());
        }
    }

} // namespace llaminar
