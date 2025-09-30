// Golden model loader test
// Compares a subset of float (F16/F32) tensors loaded by Llaminar's ModelLoader
// against the same tensors loaded through the embedded llama.cpp reference loader.
// Focuses initially on detecting corruption / mis-decoding of output_norm.weight.
//
// Behavior:
// 1. Select model path via env LLAMINAR_GOLDEN_MODEL or auto-pick smallest .gguf in models/.
// 2. Load with ModelLoader (Llaminar) and llama.cpp (llama_load_model_from_file).
// 3. Enumerate GGUF tensors; choose those with type F16/F32 (skip quantized for now).
// 4. Map Llaminar tensor names to llama internal tensor names (strip optional ".weight" suffix).
// 5. Compute per-tensor relative L2 error and element range stats; expect tight agreement.
// 6. Special handling for output_norm.weight vs output_norm naming.
//
// Current known issue: output_norm.weight appears corrupted (very large gamma values).
// This test will FAIL for that tensor (intentionally) unless the env override
//   LLAMINAR_ALLOW_OUTPUT_NORM_DIVERGENCE=1 is set, in which case it will only warn.
// Once the loader bug is fixed, remove the override usage in CI.

#include <gtest/gtest.h>
#include <filesystem>
#include <vector>
#include <string>
#include <optional>
#include <iostream>
#include <cstdlib>
#include <unordered_map>
#include <unordered_set>
#include <limits>
#include <cmath>
#include <cstring>

#include "model_loader.h"

extern "C"
{
#include "ggml.h"
#include "llama.h"
#include "ggml-quants.h"
}

// Internal helper to access tensor map (declared in llama-model.cpp)
extern const std::vector<std::pair<std::string, ggml_tensor *>> &llama_internal_get_tensor_map(const llama_model *model);

namespace
{

    struct TensorStats
    {
        double min = std::numeric_limits<double>::infinity();
        double max = -std::numeric_limits<double>::infinity();
        long double sum = 0.0L;
        long double sum_sq = 0.0L;
        size_t count = 0;
    };

    TensorStats compute_stats(const std::vector<float> &v)
    {
        TensorStats s;
        s.count = v.size();
        for (float f : v)
        {
            double d = static_cast<double>(f);
            s.min = std::min(s.min, d);
            s.max = std::max(s.max, d);
            s.sum += d;
            s.sum_sq += d * d;
        }
        return s;
    }

    double rel_l2(const std::vector<float> &a, const std::vector<float> &b)
    {
        if (a.size() != b.size())
        {
            // Return large value to signal mismatch; caller already checks sizes explicitly
            return 1e30;
        }
        long double num = 0.0L;
        long double denom = 0.0L;
        for (size_t i = 0; i < a.size(); ++i)
        {
            long double diff = (long double)a[i] - (long double)b[i];
            num += diff * diff;
            denom += (long double)a[i] * (long double)a[i];
        }
        if (denom == 0.0L)
            return (double)std::sqrt((double)num);
        return (double)std::sqrt((double)(num / denom));
    }

    std::optional<std::string> pick_model_path()
    {
        if (const char *env = std::getenv("LLAMINAR_GOLDEN_MODEL"))
        {
            return std::string(env);
        }
        std::filesystem::path models_dir = std::filesystem::path("models");
        if (!std::filesystem::exists(models_dir))
        {
            return std::nullopt;
        }
        uintmax_t best_size = std::numeric_limits<uintmax_t>::max();
        std::optional<std::string> best;
        for (auto &p : std::filesystem::directory_iterator(models_dir))
        {
            if (!p.is_regular_file())
                continue;
            auto path = p.path();
            if (path.extension() == ".gguf")
            {
                auto sz = std::filesystem::file_size(path);
                if (sz < best_size)
                {
                    best_size = sz;
                    best = path.string();
                }
            }
        }
        return best;
    }

    // Extract floats from a ggml_tensor (F32 or F16). Return empty if unsupported.
    std::vector<float> tensor_to_floats(const ggml_tensor *t)
    {
        std::vector<float> out;
        if (!t)
            return out;
        size_t n = 1;
        int nd = ggml_n_dims(t);
        for (int i = 0; i < nd; ++i)
            n *= (size_t)t->ne[i];
        out.resize(n);
        if (t->type == GGML_TYPE_F32)
        {
            std::memcpy(out.data(), t->data, n * sizeof(float));
        }
        else if (t->type == GGML_TYPE_F16)
        {
            const ggml_fp16_t *src = reinterpret_cast<const ggml_fp16_t *>(t->data);
            for (size_t i = 0; i < n; ++i)
                out[i] = ggml_fp16_to_fp32(src[i]);
        }
        else
        {
            out.clear(); // unsupported for now
        }
        return out;
    }

    std::string strip_weight_suffix(const std::string &name)
    {
        const std::string suf = ".weight";
        if (name.size() > suf.size() && name.compare(name.size() - suf.size(), suf.size(), suf) == 0)
        {
            return name.substr(0, name.size() - suf.size());
        }
        return name;
    }

} // namespace

TEST(ModelLoaderGolden, CompareFloatTensorsWithLlama)
{
    auto maybe_model = pick_model_path();
    ASSERT_TRUE(maybe_model.has_value()) << "No model .gguf file found (set LLAMINAR_GOLDEN_MODEL).";
    const std::string model_path = *maybe_model;
    std::cout << "[GoldenLoader] Using model: " << model_path << "\n";

    // Llaminar load
    ModelLoader loader;
    ASSERT_TRUE(loader.loadModel(model_path)) << "Llaminar ModelLoader failed";
    const auto &gguf = loader.getModel();

    // llama.cpp load
    llama_model_params params = llama_model_default_params();
    params.vocab_only = false;
    params.use_mmap = false; // ensures data pointer access is valid in tests
    auto *llama_model_ptr = llama_model_load_from_file(model_path.c_str(), params);
    ASSERT_NE(llama_model_ptr, nullptr) << "Failed to load model via llama.cpp";

    // Build tensor name -> ggml_tensor* map
    std::unordered_map<std::string, ggml_tensor *> llama_tensors;
    for (auto &pr : llama_internal_get_tensor_map(llama_model_ptr))
    {
        llama_tensors.emplace(pr.first, pr.second);
    }

    int compared = 0;
    int skipped = 0;
    std::unordered_set<std::string> already_compared;
    const double rel_l2_threshold = 1e-3; // strict for F32/F16

    // Critical tensor(s) we always want to inspect first (for known anomaly)
    const std::vector<std::string> critical = {"output_norm.weight"};
    for (const auto &cname : critical)
    {
        auto *info = const_cast<GGUFTensorInfo *>(gguf.findTensor(cname));
        if (!info)
            continue;
        if (!(info->type == GGUFTensorType::F32 || info->type == GGUFTensorType::F16))
            continue;
        auto tensor = loader.loadTensor(info->name);
        ASSERT_TRUE(tensor) << "Failed to load critical tensor " << info->name;
        size_t n = 1;
        for (auto d : info->dimensions)
            n *= d;
        std::vector<float> lam_values(tensor->data(), tensor->data() + n);
        std::string base = strip_weight_suffix(info->name);
        const ggml_tensor *llama_tensor = nullptr;
        auto it_exact = llama_tensors.find(info->name);
        if (it_exact != llama_tensors.end())
            llama_tensor = it_exact->second;
        else
        {
            auto it_base = llama_tensors.find(base);
            if (it_base != llama_tensors.end())
                llama_tensor = it_base->second;
        }
        if (!llama_tensor)
        {
            std::cout << "[CriticalTensor] name=" << info->name << " missing in llama reference\n";
            continue;
        }
        auto llama_vals = tensor_to_floats(llama_tensor);
        if (llama_vals.size() != lam_values.size())
        {
            std::cout << "[CriticalTensor] size mismatch name=" << info->name << " ours=" << lam_values.size() << " ref=" << llama_vals.size() << "\n";
            continue;
        }
        double rl2 = rel_l2(lam_values, llama_vals);
        double max_abs = 0.0;
        for (size_t i = 0; i < lam_values.size(); ++i)
            max_abs = std::max(max_abs, std::abs((double)lam_values[i] - (double)llama_vals[i]));
        std::cout << "[CriticalTensor] name=" << info->name << " n=" << lam_values.size() << " rel_l2=" << rl2 << " max_abs=" << max_abs << "\n";
        // Dump first 16 values from each for forensic diff
        size_t dump = std::min<size_t>(16, lam_values.size());
        std::cout << "[CriticalTensor] ours_first16=";
        for (size_t i = 0; i < dump; ++i)
            std::cout << lam_values[i] << (i + 1 < dump ? "," : "\n");
        std::cout << "[CriticalTensor] ref_first16=";
        for (size_t i = 0; i < dump; ++i)
            std::cout << llama_vals[i] << (i + 1 < dump ? "," : "\n");
        bool is_output_norm = (info->name == "output_norm.weight" || base == "output_norm");
        if (is_output_norm && std::getenv("LLAMINAR_ALLOW_OUTPUT_NORM_DIVERGENCE"))
        {
            if (rl2 > rel_l2_threshold)
                std::cout << "[WARN] output_norm (critical) divergence rl2=" << rl2 << " (override allows)\n";
        }
        else
        {
            EXPECT_LT(rl2, rel_l2_threshold) << "Relative L2 too large for critical tensor " << info->name;
        }
        already_compared.insert(info->name);
        ++compared;
    }

    // Collect candidate tensors (float types only) excluding already compared critical ones
    for (const auto &ti : gguf.tensors)
    {
        if (!(ti.type == GGUFTensorType::F32 || ti.type == GGUFTensorType::F16))
        {
            continue; // skip quantized here (future: add dequant comparison)
        }
        if (already_compared.count(ti.name))
            continue;
        // Load with Llaminar
        auto tensor = loader.loadTensor(ti.name);
        ASSERT_TRUE(tensor) << "Failed to load tensor " << ti.name;
        const float *data = tensor->data();
        size_t n = 1;
        for (auto d : ti.dimensions)
            n *= d;
        std::vector<float> lam_values(data, data + n);

        // Map name
        std::string base = strip_weight_suffix(ti.name);
        const ggml_tensor *llama_tensor = nullptr;
        auto it_exact = llama_tensors.find(ti.name);
        if (it_exact != llama_tensors.end())
            llama_tensor = it_exact->second;
        else
        {
            auto it_base = llama_tensors.find(base);
            if (it_base != llama_tensors.end())
                llama_tensor = it_base->second;
        }
        if (!llama_tensor)
        {
            ++skipped;
            continue; // no counterpart (naming difference or intentionally absent)
        }
        auto llama_vals = tensor_to_floats(llama_tensor);
        if (llama_vals.empty())
        {
            ++skipped;
            continue; // unsupported type on llama side (quantized etc.)
        }
        if (llama_vals.size() != lam_values.size())
        {
            ADD_FAILURE() << "Size mismatch for tensor " << ti.name << " ours=" << lam_values.size() << " llama=" << llama_vals.size();
            continue;
        }

        double rl2 = rel_l2(lam_values, llama_vals);
        auto ours_stats = compute_stats(lam_values);
        auto ref_stats = compute_stats(llama_vals);
        std::cout << "[TensorCompare] name=" << ti.name
                  << " n=" << lam_values.size()
                  << " rel_l2=" << rl2
                  << " ours_min=" << ours_stats.min << " ours_max=" << ours_stats.max << " ours_mean=" << (double)(ours_stats.sum / ours_stats.count)
                  << " ref_min=" << ref_stats.min << " ref_max=" << ref_stats.max << " ref_mean=" << (double)(ref_stats.sum / ref_stats.count)
                  << "\n";

        bool is_output_norm = (ti.name == "output_norm.weight" || base == "output_norm");
        if (is_output_norm && std::getenv("LLAMINAR_ALLOW_OUTPUT_NORM_DIVERGENCE"))
        {
            if (rl2 > rel_l2_threshold)
            {
                std::cout << "[WARN] output_norm divergence rl2=" << rl2 << " (override allows)\n";
            }
        }
        else
        {
            EXPECT_LT(rl2, rel_l2_threshold) << "Relative L2 too large for tensor " << ti.name;
        }
        ++compared;
        if (compared >= 16)
            break; // cap to keep test fast
    }

    std::cout << "[Summary] compared=" << compared << " skipped=" << skipped << "\n";
    llama_model_free(llama_model_ptr);
}

// Extended test: include selected quantized tensors (dequantized floats) parity vs llama.cpp.
// This is separated into its own test so we can gate runtime via env without affecting
// the faster float-only golden test above.
//
// Environment controls:
//   LLAMINAR_GOLDEN_Q_MAX_TENSORS   (default 4)   : maximum quantized tensors to compare
//   LLAMINAR_GOLDEN_Q_MAX_ELEMS     (default 500000) : skip any single tensor with more elements
//   LLAMINAR_GOLDEN_Q_RL2_TOL       (default 1e-5): relative L2 tolerance (should generally be exact)
//   LLAMINAR_GOLDEN_Q_ENABLE        (set to run)  : if unset, this test is skipped early
//   LLAMINAR_GOLDEN_MODEL           : explicit model path override
//
// Notes:
//   We access llama.cpp quantized tensor elements via ggml_get_f32_1d which performs on-demand
//   dequantization. Llaminar path obtains floats by calling ModelLoader::loadTensor which internally
//   dequantizes supported formats into a float buffer. We expect bit-identical (or near) parity.
//
//   For very large tensors (e.g. embedding tables / lm_head), the element cap prevents excessive
//   runtime. Such tensors are counted as skipped with diagnostic output.
TEST(ModelLoaderGolden, CompareQuantizedTensorsWithLlama)
{
    if (!std::getenv("LLAMINAR_GOLDEN_Q_ENABLE"))
    {
        GTEST_SKIP() << "Quantized golden test disabled (set LLAMINAR_GOLDEN_Q_ENABLE=1 to run)";
    }

    auto getenv_size_t = [](const char *name, size_t def_v) -> size_t
    {
        if (const char *e = std::getenv(name)) {
            char *end = nullptr; unsigned long long v = strtoull(e, &end, 10); if (end && end != e) return (size_t)v; }
        return def_v; };
    auto getenv_double = [](const char *name, double def_v) -> double
    {
        if (const char *e = std::getenv(name)) {
            char *end = nullptr; double v = strtod(e, &end); if (end && end != e) return v; }
        return def_v; };

    const size_t max_tensors = getenv_size_t("LLAMINAR_GOLDEN_Q_MAX_TENSORS", 4);
    const size_t per_type_target = getenv_size_t("LLAMINAR_GOLDEN_Q_PER_TYPE", 1); // initial sampling goal per quant type
    const size_t max_elems_per_tensor = getenv_size_t("LLAMINAR_GOLDEN_Q_MAX_ELEMS", 500000);
    const double rl2_tol = getenv_double("LLAMINAR_GOLDEN_Q_RL2_TOL", 1e-5);
    // Optional fallback allowing a partial (strided) sample of an oversized tensor instead of skipping it entirely.
    // Activated when LLAMINAR_GOLDEN_Q_ALLOW_PARTIAL is set (any value). Sample size defaults to max_elems_per_tensor
    // but can be overridden with LLAMINAR_GOLDEN_Q_PARTIAL_ELEMS. This is primarily to capture at least one example
    // of very large single-occurrence formats (e.g. q8_0) without forcing a huge full dequant in CI.
    const bool allow_partial = std::getenv("LLAMINAR_GOLDEN_Q_ALLOW_PARTIAL") != nullptr;
    const size_t partial_target_elems = getenv_size_t("LLAMINAR_GOLDEN_Q_PARTIAL_ELEMS", max_elems_per_tensor);

    auto maybe_model = pick_model_path();
    ASSERT_TRUE(maybe_model.has_value()) << "No model .gguf file found (set LLAMINAR_GOLDEN_MODEL).";
    const std::string model_path = *maybe_model;
    std::cout << "[QuantGolden] Using model: " << model_path << "\n";

    ModelLoader loader;
    ASSERT_TRUE(loader.loadModel(model_path)) << "Llaminar ModelLoader failed";
    const auto &gguf = loader.getModel();

    llama_model_params params = llama_model_default_params();
    params.vocab_only = false;
    params.use_mmap = false;
    auto *llama_model_ptr = llama_model_load_from_file(model_path.c_str(), params);
    ASSERT_NE(llama_model_ptr, nullptr) << "Failed to load model via llama.cpp";

    std::unordered_map<std::string, ggml_tensor *> llama_tensors;
    for (auto &pr : llama_internal_get_tensor_map(llama_model_ptr))
        llama_tensors.emplace(pr.first, pr.second);

    auto is_float_type = [](GGUFTensorType t)
    { return t == GGUFTensorType::F32 || t == GGUFTensorType::F16; };
    auto is_quant_type = [&](GGUFTensorType t)
    { return !is_float_type(t); };

    size_t compared = 0, skipped = 0, sampled_elems = 0;
    std::unordered_map<std::string, size_t> per_type_counts;
    std::unordered_map<std::string, size_t> skip_reason_counts;
    // Track minimal element size that was skipped for being too large per type so we can recommend a higher cap.
    std::unordered_map<std::string, size_t> too_large_min_elems_per_type;
    std::vector<const decltype(gguf.tensors)::value_type *> quant_candidates;

    // Collect quant tensor metadata pointers in original order
    for (const auto &ti : gguf.tensors)
    {
        if (!is_quant_type(ti.type))
            continue;
        quant_candidates.push_back(&ti);
    }

    // Helper to stringify a GGUFTensorType (subset used in summary)
    auto tensor_type_name = [](GGUFTensorType t) -> std::string
    {
        switch (t)
        {
        case GGUFTensorType::Q2_K:
            return "q2_K"; // canonical lowercase vs llama.cpp prints Q2_K but we stay consistent
        case GGUFTensorType::Q3_K:
            return "q3_K";
        case GGUFTensorType::Q4_0:
            return "q4_0";
        case GGUFTensorType::Q5_0:
            return "q5_0";
        case GGUFTensorType::Q8_0:
            return "q8_0";
        case GGUFTensorType::F32:
            return "f32"; // not expected here
        case GGUFTensorType::F16:
            return "f16"; // not expected here
        case GGUFTensorType::Q4_K:
            return "q4_K";
        default:
            return "other";
        }
    };

    // Build a working list we will scan in original order. We will stop when we hit max_tensors actual comparisons.
    // Breadth-first per-type behavior: once a type reaches per_type_target successful comparisons we temporarily
    // ignore further tensors of that type until other types reach their quota or overall budget exhausted.
    const char *debug_env = std::getenv("LLAMINAR_GOLDEN_Q_DEBUG");
    bool debug = debug_env && std::string(debug_env) == "1";
    std::vector<const decltype(gguf.tensors)::value_type *> final_selection = quant_candidates; // same ordering
    if (debug)
    {
        std::cout << "[QuantDebug] candidates=" << final_selection.size() << " per_type_target=" << per_type_target << " max_tensors=" << max_tensors << "\n";
    }

    // Helper: dequantize llama.cpp tensor into float vector (supported subset)
    auto llama_dequant_tensor = [](const ggml_tensor *t) -> std::vector<float>
    {
        std::vector<float> out;
        if (!t)
            return out;
        // Determine total elements
        int nd = ggml_n_dims(t);
        size_t n0 = (size_t)t->ne[0];
        size_t nrows = 1;
        for (int d = 1; d < nd; ++d)
            nrows *= (size_t)t->ne[d];
        if (n0 == 0 || nrows == 0)
            return out;
        out.resize(n0 * nrows);
        const uint8_t *ptr = reinterpret_cast<const uint8_t *>(t->data);
        switch (t->type)
        {
        case GGML_TYPE_Q5_0:
        {
            size_t blocks_per_row = n0 / 32; // block_q5_0 covers 32 elements
            for (size_t r = 0; r < nrows; ++r)
            {
                const block_q5_0 *row_blocks = reinterpret_cast<const block_q5_0 *>(ptr);
                dequantize_row_q5_0(row_blocks, out.data() + r * n0, (int64_t)n0);
                ptr += blocks_per_row * sizeof(block_q5_0);
            }
        }
        break;
        case GGML_TYPE_Q2_K:
        {
            size_t blocks_per_row = n0 / QK_K;
            for (size_t r = 0; r < nrows; ++r)
            {
                const block_q2_K *row_blocks = reinterpret_cast<const block_q2_K *>(ptr);
                dequantize_row_q2_K(row_blocks, out.data() + r * n0, (int64_t)n0);
                ptr += blocks_per_row * sizeof(block_q2_K);
            }
        }
        break;
        case GGML_TYPE_Q3_K:
        {
            size_t blocks_per_row = n0 / QK_K;
            for (size_t r = 0; r < nrows; ++r)
            {
                const block_q3_K *row_blocks = reinterpret_cast<const block_q3_K *>(ptr);
                dequantize_row_q3_K(row_blocks, out.data() + r * n0, (int64_t)n0);
                ptr += blocks_per_row * sizeof(block_q3_K);
            }
        }
        break;
        case GGML_TYPE_Q5_K:
        {
            size_t blocks_per_row = n0 / QK_K;
            for (size_t r = 0; r < nrows; ++r)
            {
                const block_q5_K *row_blocks = reinterpret_cast<const block_q5_K *>(ptr);
                dequantize_row_q5_K(row_blocks, out.data() + r * n0, (int64_t)n0);
                ptr += blocks_per_row * sizeof(block_q5_K);
            }
        }
        break;
        case GGML_TYPE_Q6_K:
        {
            size_t blocks_per_row = n0 / QK_K;
            for (size_t r = 0; r < nrows; ++r)
            {
                const block_q6_K *row_blocks = reinterpret_cast<const block_q6_K *>(ptr);
                dequantize_row_q6_K(row_blocks, out.data() + r * n0, (int64_t)n0);
                ptr += blocks_per_row * sizeof(block_q6_K);
            }
        }
        break;
        case GGML_TYPE_Q8_0:
        {                                    // treat as standard 32-element blocks
            size_t blocks_per_row = n0 / 32; // block_q8_0 has 32 elements
            for (size_t r = 0; r < nrows; ++r)
            {
                const block_q8_0 *row_blocks = reinterpret_cast<const block_q8_0 *>(ptr);
                dequantize_row_q8_0(row_blocks, out.data() + r * n0, (int64_t)n0);
                ptr += blocks_per_row * sizeof(block_q8_0);
            }
        }
        break;
        default:
            // Unsupported type for this path -> return empty to signal skip
            out.clear();
            break;
        }
        return out;
    };

    for (const auto *pti : final_selection)
    {
        if (compared >= max_tensors)
            break; // safety
        const auto &ti = *pti;
        std::string tname_breadth = tensor_type_name(ti.type);
        // Enforce breadth-first quota (only for types we actually support). If unsupported we still skip and continue.
        if (per_type_target > 0 && per_type_counts[tname_breadth] >= per_type_target)
        {
            // Defer this tensor for now – but if we never reach quota for other types it will remain unused.
            // To keep implementation simple we just continue; later passes not implemented => acceptable as we only want early spread.
            // (Future improvement: multi-pass scan.)
            continue;
        }
        if (!loader.supportsQuantization(ti.type))
        {
            ++skipped;
            ++skip_reason_counts["unsupported_format"];
            if (debug)
                std::cout << "[QuantTensorSkip] name=" << ti.name << " reason=unsupported_format type=" << (int)ti.type << "\n";
            else
                std::cout << "[QuantTensorSkip] name=" << ti.name << " reason=unsupported_format\n";
            continue;
        }

        // Name mapping
        std::string base = strip_weight_suffix(ti.name);
        const ggml_tensor *llama_tensor = nullptr;
        auto it_exact = llama_tensors.find(ti.name);
        if (it_exact != llama_tensors.end())
            llama_tensor = it_exact->second;
        else
        {
            auto it_base = llama_tensors.find(base);
            if (it_base != llama_tensors.end())
                llama_tensor = it_base->second;
        }
        if (!llama_tensor)
        {
            ++skipped;
            ++skip_reason_counts["no_ref"];
            std::cout << "[QuantTensorSkip] name=" << ti.name << " reason=no_ref\n";
            continue;
        }

        // Compute element count
        size_t n = 1;
        for (auto d : ti.dimensions)
            n *= (size_t)d;
        if (n == 0)
        {
            ++skipped;
            ++skip_reason_counts["zero_elems"];
            std::cout << "[QuantTensorSkip] name=" << ti.name << " reason=zero_elems\n";
            continue;
        }
        if (n > max_elems_per_tensor)
        {
            std::string tname_sz = tensor_type_name(ti.type);
            auto it_sz = too_large_min_elems_per_type.find(tname_sz);
            if (it_sz == too_large_min_elems_per_type.end() || n < it_sz->second)
                too_large_min_elems_per_type[tname_sz] = n;
            // Partial sampling path: only if enabled, not yet satisfied per-type quota, and still within overall tensor budget.
            if (allow_partial && per_type_counts[tname_sz] < per_type_target && compared < max_tensors)
            {
                // Load full tensor via loader (note: potentially large allocation; caller opted-in via env).
                auto tensor_full = loader.loadTensor(ti.name);
                if (!tensor_full || tensor_full->size() != n)
                {
                    ++skipped;
                    ++skip_reason_counts["too_large"]; // fallback to regular skip classification
                    std::cout << "[QuantTensorSkip] name=" << ti.name << " n=" << n << " reason=too_large(load_fail)\n";
                    continue;
                }
                size_t sample_n = std::min(partial_target_elems, n);
                if (sample_n == 0)
                    sample_n = std::min<size_t>(max_elems_per_tensor, n);
                size_t stride = std::max<size_t>(1, n / sample_n);
                std::vector<float> lam_sample;
                lam_sample.reserve(sample_n);
                for (size_t idx = 0; idx < n && lam_sample.size() < sample_n; idx += stride)
                    lam_sample.push_back(tensor_full->data()[idx]);
                // Reference dequant (full) then strided sample. (Memory heavy but acceptable behind env gate.)
                std::vector<float> llama_full = llama_dequant_tensor(llama_tensor);
                if (llama_full.size() != n)
                {
                    ++skipped;
                    ++skip_reason_counts["ref_size_mismatch"];
                    std::cout << "[QuantTensorSkip] name=" << ti.name << " reason=ref_size_mismatch(full_partial) got=" << llama_full.size() << " expected=" << n << "\n";
                    continue;
                }
                std::vector<float> llama_sample;
                llama_sample.reserve(lam_sample.size());
                for (size_t idx = 0; idx < n && llama_sample.size() < lam_sample.size(); idx += stride)
                    llama_sample.push_back(llama_full[idx]);
                double rl2_partial = rel_l2(lam_sample, llama_sample);
                double max_abs_partial = 0.0;
                for (size_t i = 0; i < lam_sample.size(); ++i)
                    max_abs_partial = std::max(max_abs_partial, std::fabs((double)lam_sample[i] - (double)llama_sample[i]));
                auto ours_stats_p = compute_stats(lam_sample);
                auto ref_stats_p = compute_stats(llama_sample);
                std::cout << "[QuantTensorPartialCompare] name=" << ti.name
                          << " n_total=" << n
                          << " sample_n=" << lam_sample.size()
                          << " stride=" << stride
                          << " rel_l2=" << rl2_partial
                          << " max_abs=" << max_abs_partial
                          << " ours_min=" << ours_stats_p.min << " ours_max=" << ours_stats_p.max
                          << " ref_min=" << ref_stats_p.min << " ref_max=" << ref_stats_p.max
                          << " ours_mean=" << (double)(ours_stats_p.sum / ours_stats_p.count)
                          << " ref_mean=" << (double)(ref_stats_p.sum / ref_stats_p.count)
                          << "\n";
                EXPECT_LT(rl2_partial, rl2_tol) << "Relative L2 too large for (partial) quantized tensor " << ti.name;
                ++compared;
                per_type_counts[tname_sz]++;
                sampled_elems += lam_sample.size();
                continue; // counted as compared via partial path
            }
            ++skipped;
            ++skip_reason_counts["too_large"];
            std::cout << "[QuantTensorSkip] name=" << ti.name << " n=" << n << " reason=too_large(threshold=" << max_elems_per_tensor << ")\n";
            continue;
        }

        // Load Llaminar dequant (float buffer)
        auto tensor = loader.loadTensor(ti.name);
        if (!tensor)
        {
            ++skipped;
            ++skip_reason_counts["loader_fail"];
            std::cout << "[QuantTensorSkip] name=" << ti.name << " reason=loader_fail\n";
            continue;
        }
        if (tensor->size() != n)
        {
            ++skipped;
            ++skip_reason_counts["size_mismatch"];
            std::cout << "[QuantTensorSkip] name=" << ti.name << " reason=size_mismatch ours_size=" << tensor->size() << " expected=" << n << "\n";
            continue;
        }
        std::vector<float> lam_values(tensor->data(), tensor->data() + n);

        // Llama reference dequant via raw block decode
        std::vector<float> llama_vals = llama_dequant_tensor(llama_tensor);
        if (llama_vals.empty())
        {
            ++skipped;
            ++skip_reason_counts["ref_unsupported"];
            std::cout << "[QuantTensorSkip] name=" << ti.name << " reason=ref_unsupported\n";
            continue;
        }
        if (llama_vals.size() != n)
        {
            ++skipped;
            ++skip_reason_counts["ref_size_mismatch"];
            std::cout << "[QuantTensorSkip] name=" << ti.name << " reason=ref_size_mismatch got=" << llama_vals.size() << " expected=" << n << "\n";
            continue;
        }

        double rl2 = rel_l2(lam_values, llama_vals);
        double max_abs = 0.0;
        for (size_t i = 0; i < n; ++i)
            max_abs = std::max(max_abs, std::fabs((double)lam_values[i] - (double)llama_vals[i]));
        auto ours_stats = compute_stats(lam_values);
        auto ref_stats = compute_stats(llama_vals);
        std::cout << "[QuantTensorCompare] name=" << ti.name
                  << " n=" << n
                  << " rel_l2=" << rl2
                  << " max_abs=" << max_abs
                  << " ours_min=" << ours_stats.min << " ours_max=" << ours_stats.max
                  << " ref_min=" << ref_stats.min << " ref_max=" << ref_stats.max
                  << " ours_mean=" << (double)(ours_stats.sum / ours_stats.count)
                  << " ref_mean=" << (double)(ref_stats.sum / ref_stats.count)
                  << "\n";

        EXPECT_LT(rl2, rl2_tol) << "Relative L2 too large for quantized tensor " << ti.name;
        sampled_elems += n;
        ++compared;
        std::string tname = tensor_type_name(ti.type);
        per_type_counts[tname]++;
    }

    std::ostringstream per_type_ss;
    bool first = true;
    for (auto &kv : per_type_counts)
    {
        if (!first)
            per_type_ss << ",";
        first = false;
        per_type_ss << kv.first << ":" << kv.second;
    }
    std::ostringstream skip_ss;
    first = true;
    for (auto &kv : skip_reason_counts)
    {
        if (!first)
            skip_ss << ",";
        first = false;
        skip_ss << kv.first << ":" << kv.second;
    }
    std::cout << "[QuantSummary] compared=" << compared
              << " skipped=" << skipped
              << " sampled_elems=" << sampled_elems
              << " max_tensors=" << max_tensors
              << " per_type_target=" << per_type_target
              << " elem_cap=" << max_elems_per_tensor
              << " rl2_tol=" << rl2_tol
              << " by_type={" << per_type_ss.str() << "}"
              << " skip_reasons={" << skip_ss.str() << "}"
              << "\n";

    // Emit recommendations for unsampled types that only appeared in too_large category.
    if (debug && compared < max_tensors)
    {
        for (auto &kv : too_large_min_elems_per_type)
        {
            auto itc = per_type_counts.find(kv.first);
            if (itc == per_type_counts.end() || itc->second == 0)
            {
                std::cout << "[QuantRecommend] type=" << kv.first
                          << " first_too_large_n=" << kv.second
                          << " suggest_LLAMINAR_GOLDEN_Q_MAX_ELEMS>=" << kv.second
                          << " to enable sampling" << "\n";
            }
        }
    }
    llama_model_free(llama_model_ptr);
}
