// Golden dequantization correctness test
// Compares Llaminar dequantization outputs against ggml reference implementation
// for implemented quantization formats (Option A: minimal ggml linkage).

#include <gtest/gtest.h>
#include <random>
#include <cmath>
#include <cstring>
#include <iostream>
#include "model_loader.h"

extern "C"
{
#include "ggml.h" // pulled via ggml_min
#include "ggml-quants.h"
}

// We will use *reference* quantize_row_*_ref and dequantize_row_* from ggml.
extern "C"
{
    void quantize_row_q5_0_ref(const float *x, block_q5_0 *y, int64_t k);
    void quantize_row_q2_K_ref(const float *x, block_q2_K *y, int64_t k);
    void quantize_row_q3_K_ref(const float *x, block_q3_K *y, int64_t k);
    void quantize_row_q5_K_ref(const float *x, block_q5_K *y, int64_t k);
    void quantize_row_q6_K_ref(const float *x, block_q6_K *y, int64_t k);
    void dequantize_row_q5_0(const block_q5_0 *x, float *y, int64_t k);
    void dequantize_row_q2_K(const block_q2_K *x, float *y, int64_t k);
    void dequantize_row_q3_K(const block_q3_K *x, float *y, int64_t k);
    void dequantize_row_q5_K(const block_q5_K *x, float *y, int64_t k);
    void dequantize_row_q6_K(const block_q6_K *x, float *y, int64_t k);
}

// Utility: absolute + relative error check
static inline bool within_tolerance(float ref, float got, float abs_tol, float rel_tol)
{
    float ad = std::fabs(ref - got);
    if (ad <= abs_tol)
        return true;
    float denom = std::max(std::fabs(ref), 1e-6f);
    float rd = ad / denom;
    return rd <= rel_tol;
}

struct QuantSpec
{
    GGUFTensorType type;
    int block_elems; // number of real values per block
    const char *name;
};

// Block sizes drawn from ggml quants implementation
static const QuantSpec kSpecs[] = {
    {GGUFTensorType::Q5_0, 32, "Q5_0"},
    {GGUFTensorType::Q2_K, 256, "Q2_K"},
    {GGUFTensorType::Q3_K, 256, "Q3_K"},
    {GGUFTensorType::Q5_K, 256, "Q5_K"},
    {GGUFTensorType::Q6_K, 256, "Q6_K"},
};

// For constructing a synthetic GGUF tensor info object
static GGUFTensorInfo make_info(GGUFTensorType t, size_t n_elements, size_t size_bytes)
{
    GGUFTensorInfo info;
    info.name = "synthetic";
    info.dimensions = {(uint64_t)n_elements};
    info.type = t;
    info.offset = 0;
    info.size_bytes = size_bytes;
    return info;
}

TEST(DequantGoldenTest, CompareAgainstGGMLReference)
{
    ModelLoader loader; // we only need its dequant helpers; no model loading required

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-3.0f, 3.0f);

    // Global tolerances (loosely chosen; can tighten later once implementations stabilize)
    // Start with relaxed tolerances (we will tighten once packing verified)
    // Tightened tolerances after achieving structural parity (all formats produce identical values)
    const float abs_tol = 1e-4f;
    const float rel_tol = 1e-4f;

    // Number of random test rows per format
    const int rows = 16; // keep small for CI speed

    for (const auto &spec : kSpecs)
    {
        SCOPED_TRACE(spec.name);
        // Use 4 blocks per logical row for every format to test multi-block decoding uniformly
        const int64_t row_elems = spec.block_elems * 4; // ensures divisibility
        ASSERT_EQ(row_elems % spec.block_elems, 0) << "Row size must be multiple of block size";
        const int blocks_per_row = row_elems / spec.block_elems;
        const size_t n_elements_total = size_t(rows) * size_t(row_elems);

        // Allocate host buffers of raw floats for generating each row then quantizing
        std::vector<float> src(n_elements_total);
        for (auto &v : src)
            v = dist(rng);

        // Quantization & raw storage handling depends on type
        std::vector<uint8_t> raw_quant;      // serialized bytes to feed into llaminar dequant
        raw_quant.reserve(n_elements_total); // will grow as needed

        std::vector<float> ref_dequant(n_elements_total, 0.f);

        // Quantize with ggml reference and append raw blocks
        for (int r = 0; r < rows; ++r)
        {
            const float *row_in = src.data() + r * row_elems;
            float *ref_row_out = ref_dequant.data() + r * row_elems;
            switch (spec.type)
            {
            case GGUFTensorType::Q5_0:
            {
                const int nb = row_elems / spec.block_elems;
                std::vector<block_q5_0> blocks(nb);
                quantize_row_q5_0_ref(row_in, blocks.data(), row_elems);
                dequantize_row_q5_0(blocks.data(), ref_row_out, row_elems);
                const uint8_t *p = reinterpret_cast<const uint8_t *>(blocks.data());
                raw_quant.insert(raw_quant.end(), p, p + nb * sizeof(block_q5_0));
            }
            break;
            case GGUFTensorType::Q2_K:
            {
                const int nb = row_elems / spec.block_elems;
                std::vector<block_q2_K> blocks(nb);
                quantize_row_q2_K_ref(row_in, blocks.data(), row_elems);
                dequantize_row_q2_K(blocks.data(), ref_row_out, row_elems);
                const uint8_t *p = reinterpret_cast<const uint8_t *>(blocks.data());
                raw_quant.insert(raw_quant.end(), p, p + nb * sizeof(block_q2_K));
            }
            break;
            case GGUFTensorType::Q3_K:
            {
                const int nb = row_elems / spec.block_elems;
                std::vector<block_q3_K> blocks(nb);
                quantize_row_q3_K_ref(row_in, blocks.data(), row_elems);
                dequantize_row_q3_K(blocks.data(), ref_row_out, row_elems);
                const uint8_t *p = reinterpret_cast<const uint8_t *>(blocks.data());
                raw_quant.insert(raw_quant.end(), p, p + nb * sizeof(block_q3_K));
            }
            break;
            case GGUFTensorType::Q5_K:
            {
                const int nb = row_elems / spec.block_elems;
                std::vector<block_q5_K> blocks(nb);
                quantize_row_q5_K_ref(row_in, blocks.data(), row_elems);
                dequantize_row_q5_K(blocks.data(), ref_row_out, row_elems);
                const uint8_t *p = reinterpret_cast<const uint8_t *>(blocks.data());
                raw_quant.insert(raw_quant.end(), p, p + nb * sizeof(block_q5_K));
            }
            break;
            case GGUFTensorType::Q6_K:
            {
                const int nb = row_elems / spec.block_elems;
                std::vector<block_q6_K> blocks(nb);
                quantize_row_q6_K_ref(row_in, blocks.data(), row_elems);
                dequantize_row_q6_K(blocks.data(), ref_row_out, row_elems);
                const uint8_t *p = reinterpret_cast<const uint8_t *>(blocks.data());
                raw_quant.insert(raw_quant.end(), p, p + nb * sizeof(block_q6_K));
            }
            break;
            default:
                FAIL() << "Unsupported quant type in golden test";
            }
        }

        // Build a synthetic tensor info (size_bytes is just raw_quant.size())
        auto info = make_info(spec.type, n_elements_total, raw_quant.size());

        // Dispatch into Llaminar dequant path by calling the specific helper (since selectDequantizer currently only wires subset).
        std::vector<float> test_output;
        // Call actual Llaminar dequant routine
        switch (spec.type)
        {
        case GGUFTensorType::Q5_0:
            test_output = loader.dequantizeQ5_0(raw_quant.data(), info);
            break;
        case GGUFTensorType::Q2_K:
            test_output = loader.dequantizeQ2_K(raw_quant.data(), spec.type, spec.name, info);
            break;
        case GGUFTensorType::Q3_K:
            test_output = loader.dequantizeQ3_K(raw_quant.data(), spec.type, spec.name, info);
            break;
        case GGUFTensorType::Q5_K:
            test_output = loader.dequantizeQ5_K(raw_quant.data(), spec.type, spec.name, info);
            break;
        case GGUFTensorType::Q6_K:
            test_output = loader.dequantizeQ6_K(raw_quant.data(), spec.type, spec.name, info);
            break;
        default:
            FAIL() << "Unsupported dispatch";
        }

        ASSERT_EQ(test_output.size(), ref_dequant.size()) << "Size mismatch for type " << spec.name;

        // Compare element-wise, accumulate error metrics
        double max_abs = 0.0, max_rel = 0.0, sum_abs = 0.0;
        for (size_t i = 0; i < ref_dequant.size(); ++i)
        {
            float ref = ref_dequant[i];
            float got = test_output[i];
            float ad = std::fabs(ref - got);
            float rd = ad / std::max(std::fabs(ref), 1e-6f);
            max_abs = std::max<double>(max_abs, ad);
            max_rel = std::max<double>(max_rel, rd);
            sum_abs += ad;
            if (!within_tolerance(ref, got, abs_tol, rel_tol))
            {
                ADD_FAILURE() << spec.name << " element mismatch idx=" << i << " ref=" << ref << " got=" << got << " abs=" << ad << " rel=" << rd;
                if (i > 32)
                    break; // avoid too much spam
            }
        }

        double mean_abs = sum_abs / ref_dequant.size();
        std::cout << "[Golden] " << spec.name << " max_abs=" << max_abs << " max_rel=" << max_rel << " mean_abs=" << mean_abs << "\n";

        // Temporary relaxed gating (to be tightened to 1e-2 / 5e-2 after alignment)
        EXPECT_LT(max_abs, 5e-4f) << "Max absolute error too large for " << spec.name;
        EXPECT_LT(max_rel, 5e-4f) << "Relative error high for " << spec.name;
    }
}
