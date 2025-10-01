// Clean, minimal includes and helpers
#include "model_loader.h"
#include "logger.h"
#include "quant_dequant.h"
#include "../llama.cpp/ggml/src/ggml-quants.h" // upstream dequantize_row_q*_1
#include <cstring>
#include <numeric>
#include <cstdlib>
#include "utils/debug_env.h"

// Include upstream gguf API for offset comparison diagnostics
extern "C"
{
#include "gguf.h"
}
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <sstream>
#include "graph_compute.h"
#include <filesystem>
#include "tensors/simple_tensor.h"
#include <iostream>
#include <cstdio>
#include <cstdint>

#ifndef K_SCALE_SIZE
#define K_SCALE_SIZE 12
#endif

typedef uint16_t ggml_half;

// Mirrors ggml's ggml_compute_fp16_to_fp32 to ensure identical behaviour for subnormals.
static inline float fp32_from_bits(uint32_t w)
{
    union
    {
        uint32_t as_bits;
        float as_value;
    } fp32;
    fp32.as_bits = w;
    return fp32.as_value;
}

static inline uint32_t fp32_to_bits(float f)
{
    union
    {
        float as_value;
        uint32_t as_bits;
    } fp32;
    fp32.as_value = f;
    return fp32.as_bits;
}

static inline float ggml_compute_fp16_to_fp32(uint16_t h)
{
    const uint32_t w = static_cast<uint32_t>(h) << 16;
    const uint32_t sign = w & UINT32_C(0x80000000);
    const uint32_t two_w = w + w;

    const uint32_t exp_offset = UINT32_C(0xE0) << 23;
#if (defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) || defined(__GNUC__) && !defined(__STRICT_ANSI__)) && (!defined(__cplusplus) || __cplusplus >= 201703L)
    const float exp_scale = 0x1.0p-112f;
#else
    const float exp_scale = fp32_from_bits(UINT32_C(0x7800000));
#endif
    const float normalized_value = fp32_from_bits((two_w >> 4) + exp_offset) * exp_scale;

    const uint32_t magic_mask = UINT32_C(126) << 23;
    const float magic_bias = 0.5f;
    const float denormalized_value = fp32_from_bits((two_w >> 17) | magic_mask) - magic_bias;

    const uint32_t denormalized_cutoff = UINT32_C(1) << 27;
    const uint32_t result = sign | (two_w < denormalized_cutoff ? fp32_to_bits(denormalized_value) : fp32_to_bits(normalized_value));
    return fp32_from_bits(result);
}

// Constructor
ModelLoader::ModelLoader() : loaded_(false) {}

size_t ModelLoader::getFileSize() const
{
    // tellg/seekg are non-const; create a copy of underlying fd position using const_cast
    auto &fs = const_cast<std::ifstream &>(file_stream_);
    if (!fs.good())
        return 0;
    std::ifstream::pos_type current = fs.tellg();
    fs.seekg(0, std::ios::end);
    std::ifstream::pos_type end = fs.tellg();
    fs.seekg(current, std::ios::beg);
    if (end < 0)
        return 0;
    return static_cast<size_t>(end);
}

void ModelLoader::logDequantStats(const std::string &tensor_name, GGUFTensorType type, const std::vector<float> &values, size_t max_samples) const
{
    const auto &snap = llaminar::debugEnv();
    if (!snap.dequant.stats)
        return;
    if (values.empty())
    {
        LOG_INFO("[DEQUANT STATS] tensor='" << tensor_name << "' (empty)");
        return;
    }
    double min_v = std::numeric_limits<double>::infinity();
    double max_v = -std::numeric_limits<double>::infinity();
    long double sum = 0.0L;
    for (size_t i = 0; i < values.size(); ++i)
    {
        float v = values[i];
        if (v < min_v)
            min_v = v;
        if (v > max_v)
            max_v = v;
        sum += v;
    }
    double mean = static_cast<double>(sum / values.size());
    long double var_acc = 0.0L;
    for (float v : values)
    {
        long double d = v - mean;
        var_acc += d * d;
    }
    double var = static_cast<double>(var_acc / values.size());
    double stddev = std::sqrt(var);
    std::ostringstream ss;
    ss.setf(std::ios::fixed);
    ss.precision(6);
    ss << "[DEQUANT STATS] tensor='" << tensor_name << "' type=" << static_cast<int>(type)
       << " n=" << values.size() << " min=" << min_v << " max=" << max_v << " mean=" << mean << " stddev=" << stddev;
    size_t samples = std::min(max_samples, values.size());
    ss << " samples=";
    for (size_t i = 0; i < samples; ++i)
    {
        if (i)
            ss << ",";
        ss << values[i];
    }
    LOG_INFO(ss.str());
}

// Remove legacy duplicate fp16 helpers later in file (original block with fp32_from_bits etc. has been deleted)
void ModelLoader::printModelInfo() const
{
    if (!loaded_)
    {
        std::cout << "Model not loaded" << std::endl;
        return;
    }

    std::cout << "\n=== Model Information ===" << std::endl;
    std::cout << "File: " << file_path_ << std::endl;
    std::cout << "GGUF Version: " << model_.version << std::endl;
    std::cout << "Architecture: " << model_.architecture << std::endl;
    std::cout << "Context Length: " << model_.context_length << std::endl;
    std::cout << "Embedding Length: " << model_.embedding_length << std::endl;
    std::cout << "Block Count: " << model_.block_count << std::endl;
    std::cout << "Head Count: " << model_.head_count << std::endl;
    std::cout << "Head Count KV: " << model_.head_count_kv << std::endl;
    std::cout << "Tensors: " << model_.tensor_count << std::endl;
    std::cout << "Metadata entries: " << model_.metadata_kv_count << std::endl;
    std::cout << "=========================" << std::endl;
}

// ---------------- GGUF helper method definitions (previously only declared) -----------------
bool GGUFModel::hasMetadata(const std::string &key) const { return metadata.find(key) != metadata.end(); }

GGUFTensorInfo *GGUFModel::findTensor(const std::string &name)
{
    for (auto &t : tensors)
        if (t.name == name)
            return &t;
    return nullptr;
}
const GGUFTensorInfo *GGUFModel::findTensor(const std::string &name) const
{
    for (auto const &t : tensors)
        if (t.name == name)
            return &t;
    return nullptr;
}

bool GGUFTensorInfo::isQuantized() const
{
    switch (type)
    {
    case GGUFTensorType::Q4_0:
    case GGUFTensorType::Q4_1:
    case GGUFTensorType::Q5_0:
    case GGUFTensorType::Q5_1:
    case GGUFTensorType::Q8_0:
    case GGUFTensorType::Q2_K:
    case GGUFTensorType::Q3_K:
    case GGUFTensorType::Q4_K:
    case GGUFTensorType::Q5_K:
    case GGUFTensorType::Q6_K:
    case GGUFTensorType::Q8_K:
    case GGUFTensorType::IQ2_XXS:
    case GGUFTensorType::IQ2_XS:
    case GGUFTensorType::IQ3_XXS:
    case GGUFTensorType::IQ1_S:
    case GGUFTensorType::IQ4_NL:
    case GGUFTensorType::IQ3_S:
    case GGUFTensorType::IQ2_S:
    case GGUFTensorType::IQ4_XS:
    case GGUFTensorType::IQ1_M:
        return true;
    default:
        return false;
    }
}

size_t GGUFTensorInfo::getTypeSize() const
{
    switch (type)
    {
    case GGUFTensorType::F32:
        return 4;
    case GGUFTensorType::F16:
        return 2;
    case GGUFTensorType::Q4_0:
        return 18; // 2 + 16
    case GGUFTensorType::Q4_1:
        return 2 * sizeof(ggml_half) + 16; // match ggml block_q4_1 (scale + min + 16 packed)
    case GGUFTensorType::Q5_0:
        return 22; // 2 + 4 + 16
    case GGUFTensorType::Q5_1:
        return 2 * sizeof(ggml_half) + 4 + 16; // scale + min + qh + qs per ggml block_q5_1
    case GGUFTensorType::Q8_0:
        return 34; // 2 + 32
    case GGUFTensorType::Q2_K:
        return 84;
    case GGUFTensorType::Q3_K:
        return 110;
    case GGUFTensorType::Q4_K:
        return 144;
    case GGUFTensorType::Q5_K:
        return 176;
    case GGUFTensorType::Q6_K:
        return 210;
    case GGUFTensorType::Q8_K:
        return 288; // placeholder, verify if used
    case GGUFTensorType::IQ2_XXS:
        return sizeof(block_iq2_xxs);
    case GGUFTensorType::IQ2_XS:
        return sizeof(block_iq2_xs);
    case GGUFTensorType::IQ3_XXS:
        return sizeof(block_iq3_xxs);
    case GGUFTensorType::IQ1_S:
        return sizeof(block_iq1_s);
    case GGUFTensorType::IQ4_NL:
        return sizeof(block_iq4_nl);
    case GGUFTensorType::IQ3_S:
        return sizeof(block_iq3_s);
    case GGUFTensorType::IQ2_S:
        return sizeof(block_iq2_s);
    case GGUFTensorType::IQ4_XS:
        return sizeof(block_iq4_xs);
    case GGUFTensorType::IQ1_M:
        return sizeof(block_iq1_m);
    default:
        return 0;
    }
}

size_t GGUFTensorInfo::getBlockSize() const
{
    switch (type)
    {
    case GGUFTensorType::Q4_0:
    case GGUFTensorType::Q4_1:
    case GGUFTensorType::Q5_0:
    case GGUFTensorType::Q5_1:
    case GGUFTensorType::Q8_0:
        return 32;
    case GGUFTensorType::Q2_K:
    case GGUFTensorType::Q3_K:
    case GGUFTensorType::Q4_K:
    case GGUFTensorType::Q5_K:
    case GGUFTensorType::Q6_K:
    case GGUFTensorType::Q8_K:
        return 256;
    case GGUFTensorType::IQ2_XXS:
    case GGUFTensorType::IQ2_XS:
    case GGUFTensorType::IQ3_XXS:
    case GGUFTensorType::IQ1_S:
    case GGUFTensorType::IQ3_S:
    case GGUFTensorType::IQ2_S:
    case GGUFTensorType::IQ4_XS:
    case GGUFTensorType::IQ1_M:
        return 256; // these IQ variants use QK_K block size upstream
    case GGUFTensorType::IQ4_NL:
        return 32; // QK4_NL
    default:
        return 0;
    }
}

std::string GGUFValue::asString() const
{
    if (type != GGUFValueType::STRING || data.size() < 8)
        return {};
    uint64_t len;
    std::memcpy(&len, data.data(), 8);
    if (8 + len > data.size())
        return {};
    return std::string(reinterpret_cast<const char *>(data.data() + 8), len);
}

std::vector<std::string> GGUFValue::asStringArray() const
{
    // Layout in data: [array_type(4)][array_length(8)] then repeated [len(8)][bytes]
    std::vector<std::string> out;
    if (data.size() < 12)
        return out;
    uint32_t atype;
    std::memcpy(&atype, data.data(), 4);
    if (atype != static_cast<uint32_t>(GGUFValueType::STRING))
        return out;
    uint64_t count;
    std::memcpy(&count, data.data() + 4, 8);
    size_t cursor = 12;
    for (uint64_t i = 0; i < count && cursor + 8 <= data.size(); ++i)
    {
        uint64_t len;
        std::memcpy(&len, data.data() + cursor, 8);
        cursor += 8;
        if (cursor + len > data.size())
            break;
        out.emplace_back(reinterpret_cast<const char *>(data.data() + cursor), len);
        cursor += len;
    }
    return out;
}

// -------------- Model loading high-level API --------------
bool ModelLoader::loadModel(const std::string &file_path)
{
    const auto &loader_env0 = llaminar::debugEnv();
    if (loader_env0.loader.model_load_debug)
    {
        LOG_DEBUG("[MODEL_LOAD_FLOW] Enter loadModel path=" << file_path);
    }
    file_stream_.close();
    file_stream_.clear();
    file_stream_.open(file_path, std::ios::binary);
    if (!file_stream_)
    {
        LOG_ERROR("Failed to open GGUF file: " << file_path);
        loaded_ = false;
        return false;
    }
    file_path_ = file_path;
    if (!parseHeader())
    {
        LOG_ERROR("parseHeader failed");
        return false;
    }
    if (!parseMetadata())
    {
        LOG_ERROR("parseMetadata failed");
        return false;
    }
    if (!parseTensorInfo())
    {
        LOG_ERROR("parseTensorInfo failed");
        return false;
    }
    {
        const auto &loader_env1 = llaminar::debugEnv();
        bool dbg = loader_env1.loader.model_load_debug;
        if (dbg)
        {
            LOG_DEBUG("[MODEL_LOAD_DEBUG] tensor summary count=" << model_.tensors.size());
            size_t dumpN = std::min<size_t>(10, model_.tensors.size());
            for (size_t i = 0; i < dumpN; ++i)
            {
                const auto &t = model_.tensors[i];
                LOG_WARN("[MODEL_LOAD_DEBUG] i=" << i << " name=" << t.name
                                                 << " off=" << t.offset << " size=" << t.size_bytes
                                                 << " dims=" << (t.dimensions.size())
                                                 << " type=" << (int)t.type << (t.isQuantized() ? " Q" : " F"));
            }
        }
    }
    extractModelMetadata();
    if (!validateModel())
    {
        LOG_ERROR("validateModel failed");
        return false;
    }
    // Compute data offset (current position) with required alignment padding.
    // GGUF writers pad the header/tensor info section to 32-byte alignment before
    // the tensor data region. Our previous code recorded the raw stream position
    // which could be in the middle of the padding, causing all subsequent tensor
    // reads to start earlier than intended (observed as denormal ~1e-38 float values
    // for F32 tensors like output_norm.weight due to reading quantized byte regions).
    {
        const auto &loader_env2 = llaminar::debugEnv();
        bool dbg = loader_env2.loader.model_load_debug;
        std::streampos pos = file_stream_.tellg();
        uint64_t cur = static_cast<uint64_t>(pos);
        uint64_t align = model_.alignment ? model_.alignment : 32;
        uint64_t aligned = (cur + align - 1) / align * align;
        if (aligned != cur)
        {
            file_stream_.seekg(static_cast<std::streamoff>(aligned), std::ios::beg);
            if (!file_stream_)
            {
                LOG_ERROR("Failed to seek to aligned data offset");
                return false;
            }
        }
        uint64_t base = aligned;
        // Recompute base using first tensor's offset if it is non-zero (some exporters
        // have been observed to emit a non-zero first offset, making the aligned position
        // insufficient). If first offset is zero, aligned is the correct base.
        if (!model_.tensors.empty())
        {
            uint64_t first_off = model_.tensors[0].offset;
            if (first_off != 0)
            {
                // Current file position corresponds to (base + first_off). Solve for base.
                uint64_t recomputed = aligned - first_off;
                if (dbg)
                {
                    LOG_WARN("[MODEL_LOAD_DEBUG] first tensor offset !=0 (" << first_off
                                                                            << ") recomputing data_offset base from aligned=" << aligned
                                                                            << " -> base=" << recomputed);
                }
                base = recomputed;
            }
            else if (dbg)
            {
                LOG_INFO("[MODEL_LOAD_DEBUG] first tensor offset=0 (expected), using aligned base=" << aligned);
            }
        }
        model_.data_offset = base;
        if (dbg)
        {
            LOG_INFO("[MODEL_LOAD_DEBUG] final data_offset=" << model_.data_offset << " (raw_pos=" << cur << ", aligned_pos=" << aligned << ")");
        }
    }

    // Optional: compare our parsed tensor offsets/types with upstream gguf loader
    // Add diagnostic logging to verify this block executes and env var is visible.
    const auto &loader_env3 = llaminar::debugEnv();
    if (loader_env3.loader.model_compare_gguf)
    {
        LOG_DEBUG("[GGUF_COMPARE_DIAG] entry cmp_env=<snapshot>"
                  << " alignment=" << model_.alignment
                  << " data_offset=" << model_.data_offset
                  << " tensor_count=" << model_.tensors.size());
        LOG_INFO("[GGUF_COMPARE] starting upstream comparison phase");
        const char *cpath = file_path.c_str();
        struct gguf_init_params params{/* no_alloc */ true, /* ctx */ nullptr};
        struct gguf_context *ctx = gguf_init_from_file(cpath, params);
        if (!ctx)
        {
            LOG_WARN("[GGUF_COMPARE] Failed to init gguf context for comparison");
        }
        else
        {
            size_t n_up = gguf_get_n_tensors(ctx);
            if (n_up != model_.tensors.size())
            {
                LOG_WARN("[GGUF_COMPARE] tensor count mismatch ours=" << model_.tensors.size() << " upstream=" << n_up);
            }
            size_t n = std::min(n_up, model_.tensors.size());
            uint64_t running_expected = 0;
            uint32_t align = model_.alignment ? model_.alignment : 32;
            size_t mismatch_count = 0;
            for (size_t i = 0; i < n; ++i)
            {
                const GGUFTensorInfo &ours = model_.tensors[i];
                const char *up_name = gguf_get_tensor_name(ctx, i);
                auto up_type = gguf_get_tensor_type(ctx, i);
                uint64_t up_off = gguf_get_tensor_offset(ctx, i);
                bool name_diff = (ours.name != up_name);
                bool off_diff = (ours.offset != up_off);
                bool type_diff = (static_cast<int>(ours.type) != static_cast<int>(up_type));
                // compute our padded size for previous tensor to validate chain
                if (i == 0)
                    running_expected = 0;
                else
                {
                    const GGUFTensorInfo &prev = model_.tensors[i - 1];
                    size_t prev_size = prev.size_bytes;
                    size_t padded = ((prev_size + align - 1) / align) * align;
                    running_expected += padded;
                }
                bool chain_diff = (ours.offset != running_expected);
                if (name_diff || off_diff || type_diff || chain_diff)
                {
                    if (mismatch_count < 50)
                    {
                        LOG_WARN("[GGUF_COMPARE] i=" << i
                                                     << " name='" << ours.name << "' up_name='" << up_name << "'"
                                                     << " ours_off=" << ours.offset << " up_off=" << up_off
                                                     << " chain_expected=" << running_expected
                                                     << " ours_type=" << (int)ours.type << " up_type=" << (int)up_type
                                                     << (name_diff ? " nameDiff" : "")
                                                     << (off_diff ? " offDiff" : "")
                                                     << (type_diff ? " typeDiff" : "")
                                                     << (chain_diff ? " chainDiff" : ""));
                    }
                    ++mismatch_count;
                }
            }
            LOG_INFO("[GGUF_COMPARE] Completed comparison n=" << n << " mismatches=" << mismatch_count);
            gguf_free(ctx);
        }
    }
    else if (loader_env3.loader.model_load_debug)
    {
        LOG_INFO("[GGUF_COMPARE] comparison skipped (disabled)");
    }
    // Post-parse invariants: verify offset chain alignment (debug only)
    if (loader_env3.loader.model_load_debug)
    {
        uint32_t align = model_.alignment ? model_.alignment : 32;
        uint64_t expected = 0;
        size_t mismatches = 0;
        for (size_t i = 0; i < model_.tensors.size(); ++i)
        {
            const auto &t = model_.tensors[i];
            if (t.offset != expected)
            {
                if (mismatches < 8)
                {
                    LOG_ERROR("[MODEL_LOAD_INVARIANT] chain mismatch i=" << i << " name='" << t.name << "' got_off=" << t.offset << " expected_off=" << expected);
                }
                ++mismatches;
            }
            size_t padded = ((t.size_bytes + align - 1) / align) * align;
            expected += padded;
        }
        if (mismatches == 0)
        {
            LOG_DEBUG("[MODEL_LOAD_INVARIANT] offset chain OK tensors=" << model_.tensors.size() << " align=" << align);
        }
        else
        {
            LOG_ERROR("[MODEL_LOAD_INVARIANT] offset chain FAILED mismatches=" << mismatches);
        }
    }
    loaded_ = true;
    return true;
}

std::shared_ptr<llaminar::TensorBase> ModelLoader::loadTensor(const std::string &tensor_name)
{
    if (!loaded_)
    {
        LOG_ERROR("Model not loaded");
        return nullptr;
    }
    const GGUFTensorInfo *info = model_.findTensor(tensor_name);
    if (!info)
    {
        LOG_ERROR("Tensor not found: " << tensor_name);
        return nullptr;
    }
    // Seek to tensor offset
    file_stream_.seekg(model_.data_offset + info->offset, std::ios::beg);
    std::vector<uint8_t> raw(info->size_bytes);
    if (!file_stream_.read(reinterpret_cast<char *>(raw.data()), raw.size()))
    {
        LOG_ERROR("Failed to read tensor bytes: " << tensor_name);
        return nullptr;
    }
    const auto &loader_env_load = llaminar::debugEnv();
    bool dbg = loader_env_load.loader.model_load_debug;
    size_t n_elems = 1;
    for (auto d : info->dimensions)
        n_elems *= d;
    std::vector<float> data_f32;
    if (info->isQuantized())
    {
        if (dbg)
        {
            std::ostringstream oss;
            oss << "[MODEL_LOAD_DEBUG] tensor='" << tensor_name << "' type_enum=" << (int)info->type
                << " quantized=1 size_bytes=" << info->size_bytes
                << " dims=";
            for (size_t i = 0; i < info->dimensions.size(); ++i)
            {
                if (i)
                    oss << "x";
                oss << info->dimensions[i];
            }
            size_t sample_bytes = std::min<size_t>(raw.size(), 32);
            oss << " raw_first_bytes=";
            for (size_t i = 0; i < sample_bytes; ++i)
            {
                if (i)
                    oss << ' ';
                oss << std::hex << std::uppercase << (int)raw[i] << std::dec;
            }
            LOG_INFO(oss.str());
        }
        data_f32 = dequantizeTensor(*info, raw, tensor_name);
    }
    else
    {
        // F16/F32 decode
        if (info->type == GGUFTensorType::F32)
        {
            data_f32.resize(n_elems);
            std::memcpy(data_f32.data(), raw.data(), n_elems * sizeof(float));
            if (dbg)
            {
                float fv = data_f32.empty() ? 0.0f : data_f32[0];
                uint32_t bits;
                std::memcpy(&bits, &fv, 4);
                std::ostringstream bytes_oss;
                bytes_oss << std::hex << std::uppercase;
                size_t byte_count = std::min<size_t>(8, raw.size());
                for (size_t i = 0; i < byte_count; ++i)
                {
                    if (i)
                        bytes_oss << ' ';
                    bytes_oss << (int)raw[i];
                }
                std::ostringstream oss;
                oss << "[MODEL_LOAD_DEBUG] tensor='" << tensor_name << "' type=F32 first_f32=" << fv << " bits=0x" << std::hex << bits;
                oss << " raw_first_bytes=" << bytes_oss.str();
                LOG_INFO(oss.str());
            }
        }
        else if (info->type == GGUFTensorType::F16)
        {
            data_f32.resize(n_elems);
            for (size_t i = 0; i < n_elems; ++i)
            {
                uint16_t h;
                std::memcpy(&h, raw.data() + 2 * i, 2);
                data_f32[i] = ggml_compute_fp16_to_fp32(h);
            }
            if (dbg)
            {
                uint16_t h0 = 0;
                if (!raw.empty())
                    std::memcpy(&h0, raw.data(), 2);
                std::ostringstream oss;
                oss << "[MODEL_LOAD_DEBUG] tensor='" << tensor_name << "' type=F16 first_half_bits=0x" << std::hex << h0 << std::dec << " first_f32=" << (data_f32.empty() ? 0.0f : data_f32[0]);
                LOG_INFO(oss.str());
            }
        }
        else
        {
            LOG_ERROR("Unsupported non-quantized type enum=" << (int)info->type);
            return nullptr;
        }
    }
    // Allocate simple tensor (row-major) -- placeholder until hybrid tensors integrated
    std::vector<int> dims;
    dims.reserve(info->dimensions.size());
    for (auto d : info->dimensions)
        dims.push_back(static_cast<int>(d));
    auto simple = std::make_shared<llaminar::SimpleTensor>(dims, data_f32);
    LOG_INFO("Loaded tensor '" << tensor_name << "' elements=" << n_elems << " first=" << (data_f32.empty() ? 0 : data_f32[0]));
    return simple;
}

std::vector<std::shared_ptr<llaminar::TensorBase>> ModelLoader::loadAllTensors()
{
    std::vector<std::shared_ptr<llaminar::TensorBase>> tensors;
    if (!loaded_)
        return tensors;
    tensors.reserve(model_.tensors.size());
    for (auto const &ti : model_.tensors)
    {
        tensors.push_back(loadTensor(ti.name));
    }
    return tensors;
}

bool ModelLoader::parseHeader()
{
    // Read GGUF magic number
    char magic[4];
    if (!file_stream_.read(magic, 4) || std::string(magic, 4) != "GGUF")
    {
        return false;
    }

    // Read version
    if (!readValue(model_.version))
        return false;

    // Read tensor count
    if (!readValue(model_.tensor_count))
        return false;

    // Read metadata count
    if (!readValue(model_.metadata_kv_count))
        return false;

    std::cout << "GGUF Header: version=" << model_.version
              << ", tensors=" << model_.tensor_count
              << ", metadata=" << model_.metadata_kv_count << std::endl;

    return true;
}

bool ModelLoader::parseMetadata()
{
    LOG_TRACE("parseMetadata: Starting, metadata_kv_count=" << model_.metadata_kv_count);
    for (uint64_t i = 0; i < model_.metadata_kv_count; ++i)
    {
        LOG_TRACE("parseMetadata: Processing metadata entry " << i << "/" << model_.metadata_kv_count);
        std::string key;
        if (!readString(key))
            return false;
        LOG_TRACE("parseMetadata: Read key: " << key);

        uint32_t value_type;
        if (!readValue(value_type))
            return false;
        LOG_TRACE("parseMetadata: Read value_type=" << value_type);

        GGUFValue value;
        value.type = static_cast<GGUFValueType>(value_type);
        LOG_TRACE("parseMetadata: Cast to GGUFValueType: " << static_cast<int>(value.type));

        if (value.type == GGUFValueType::ARRAY)
        {
            if (!readArray(value))
                return false;
        }
        else
        {
            // Read simple value based on type
            size_t value_size = 0;
            switch (value.type)
            {
            case GGUFValueType::UINT8:
                value_size = 1;
                break;
            case GGUFValueType::INT8:
                value_size = 1;
                break;
            case GGUFValueType::UINT16:
                value_size = 2;
                break;
            case GGUFValueType::INT16:
                value_size = 2;
                break;
            case GGUFValueType::UINT32:
                value_size = 4;
                break;
            case GGUFValueType::INT32:
                value_size = 4;
                break;
            case GGUFValueType::FLOAT32:
                value_size = 4;
                break;
            case GGUFValueType::UINT64:
                value_size = 8;
                break;
            case GGUFValueType::INT64:
                value_size = 8;
                break;
            case GGUFValueType::FLOAT64:
                value_size = 8;
                break;
            case GGUFValueType::BOOL:
                value_size = 1;
                break;
            case GGUFValueType::STRING:
            {
                uint64_t str_len;
                if (!readValue(str_len))
                    return false;
                LOG_TRACE("parseMetadata: STRING type, str_len=" << str_len);
                if (str_len > 1000000)
                { // 1MB sanity check
                    LOG_ERROR("parseMetadata: STRING length too large: " << str_len);
                    return false;
                }
                LOG_TRACE("parseMetadata: Attempting to resize data to " << (8 + str_len));
                value.data.resize(8 + str_len);
                LOG_TRACE("parseMetadata: Resize successful, copying data");
                std::memcpy(value.data.data(), &str_len, 8);
                if (!file_stream_.read(reinterpret_cast<char *>(value.data.data() + 8), str_len))
                {
                    return false;
                }
                break;
            }
            default:
                return false;
            }

            if (value.type != GGUFValueType::STRING)
            {
                LOG_TRACE("parseMetadata: Non-string type, value_size=" << value_size);
                if (value_size > 1000000)
                { // 1MB sanity check
                    LOG_ERROR("parseMetadata: Value size too large: " << value_size);
                    return false;
                }
                LOG_TRACE("parseMetadata: Attempting to resize data to " << value_size);
                value.data.resize(value_size);
                LOG_TRACE("parseMetadata: Resize successful, reading data");
                if (!file_stream_.read(reinterpret_cast<char *>(value.data.data()), value_size))
                {
                    return false;
                }
            }
        }

        model_.metadata[key] = std::move(value);
        LOG_TRACE("parseMetadata: Completed entry " << i << " for key: " << key);
    }
    LOG_TRACE("parseMetadata: All metadata processed successfully");

    // Extract and validate architecture
    auto arch_it = model_.metadata.find("general.architecture");
    if (arch_it != model_.metadata.end() && arch_it->second.type == GGUFValueType::STRING)
    {
        const auto &arch_data = arch_it->second.data;
        if (arch_data.size() >= 8) // String has 8-byte length prefix
        {
            uint64_t str_len;
            std::memcpy(&str_len, arch_data.data(), 8);
            if (arch_data.size() >= 8 + str_len)
            {
                model_.architecture = std::string(reinterpret_cast<const char *>(arch_data.data() + 8), str_len);
                LOG_INFO("parseMetadata: Extracted architecture: " << model_.architecture);
            }
        }
    }

    if (model_.architecture.empty())
    {
        LOG_WARN("parseMetadata: Architecture not found or could not be extracted from metadata");
    }

    return true;
}

bool ModelLoader::parseTensorInfo()
{
    LOG_TRACE("parseTensorInfo: Resizing tensors vector to " << model_.tensor_count);
    model_.tensors.resize(model_.tensor_count);
    LOG_TRACE("parseTensorInfo: Vector resized successfully");

    for (uint64_t i = 0; i < model_.tensor_count; ++i)
    {
        LOG_TRACE("parseTensorInfo: Processing tensor " << i << "/" << model_.tensor_count);
        auto &tensor = model_.tensors[i];

        // Read tensor name
        if (!readString(tensor.name))
            return false;
        LOG_TRACE("parseTensorInfo: Read tensor name: " << tensor.name);

        // Read number of dimensions
        uint32_t n_dims;
        if (!readValue(n_dims))
            return false;

        // Read dimensions
        tensor.dimensions.resize(n_dims);
        for (uint32_t j = 0; j < n_dims; ++j)
        {
            if (!readValue(tensor.dimensions[j]))
                return false;
        }

        // Read tensor type
        uint32_t type_val;
        if (!readValue(type_val))
            return false;
        // Map raw type id -> enum. Only Q8_1 (9) remains unsupported; allow Q4_1(3)/Q5_1(7) again.
        if (type_val == 9)
        {
            LOG_ERROR("parseTensorInfo: Tensor '" << tensor.name << "' uses unsupported quantization type id=9 (Q8_1). Re-export model without Q8_1.");
            return false;
        }
        if (type_val >= 16 && type_val <= 23)
        {
            LOG_TRACE("parseTensorInfo: IQ-family tensor type id=" << type_val << " name='" << tensor.name << "'");
        }
        tensor.type = static_cast<GGUFTensorType>(type_val);
        LOG_TRACE("parseTensorInfo: Tensor '" << tensor.name << "' raw type_val=" << type_val << ", cast enum value=" << static_cast<int>(tensor.type));

        // Read tensor offset
        if (!readValue(tensor.offset))
            return false;

        // Calculate tensor size
        size_t total_elements = 1;
        for (uint64_t dim : tensor.dimensions)
        {
            total_elements *= dim;
        }

        if (tensor.isQuantized())
        {
            size_t block_size = tensor.getBlockSize();
            size_t type_size = tensor.getTypeSize();
            size_t num_blocks = (total_elements + block_size - 1) / block_size;
            LOG_TRACE("parseTensorInfo: Quantized tensor '" << tensor.name << "' total_elements=" << total_elements
                                                            << ", block_elems=" << block_size << ", type_block_bytes=" << type_size
                                                            << ", num_blocks=" << num_blocks << ", computed_size=" << (num_blocks * type_size));
            tensor.size_bytes = num_blocks * type_size;
            const auto &snap_dbg = llaminar::debugEnv();
            if (snap_dbg.loader.model_load_debug)
            {
                if (tensor.size_bytes == 0 || type_size == 0)
                {
                    LOG_WARN("[MODEL_LOAD_DEBUG] tensor='" << tensor.name << "' computed zero size (type_size=" << type_size << ")");
                }
            }
        }
        else
        {
            tensor.size_bytes = total_elements * tensor.getTypeSize();
        }
        LOG_TRACE("parseTensorInfo: Final size_bytes for '" << tensor.name << "' = " << tensor.size_bytes);
    }

    return true;
}

template <typename T>
bool ModelLoader::readValue(T &value)
{
    return static_cast<bool>(file_stream_.read(reinterpret_cast<char *>(&value), sizeof(T)));
}

bool ModelLoader::readString(std::string &str)
{
    uint64_t length;
    if (!readValue(length))
        return false;

    LOG_TRACE("readString: Read length=" << length);
    if (length > 10000000)
    { // 10MB sanity check for string length
        LOG_ERROR("readString: String length too large: " << length);
        return false;
    }

    LOG_TRACE("readString: Attempting to resize string to " << length);
    str.resize(length);
    LOG_TRACE("readString: Resize successful, reading string data");
    return static_cast<bool>(file_stream_.read(str.data(), length));
}

void ModelLoader::extractModelMetadata()
{
    model_.architecture = model_.hasMetadata("general.architecture") ? model_.metadata.at("general.architecture").asString() : "unknown";

    model_.context_length = model_.getMetadata<uint32_t>("qwen2.context_length", 0);
    model_.embedding_length = model_.getMetadata<uint32_t>("qwen2.embedding_length", 0);
    model_.block_count = model_.getMetadata<uint32_t>("qwen2.block_count", 0);
    // Feed-forward hidden dimension (sometimes named feed_forward_length). Use 0 default if absent.
    model_.feed_forward_length = model_.getMetadata<uint32_t>("qwen2.feed_forward_length", 0);
    model_.head_count = model_.getMetadata<uint32_t>("qwen2.attention.head_count", 0);
    model_.head_count_kv = model_.getMetadata<uint32_t>("qwen2.attention.head_count_kv", 0);

    if (model_.hasMetadata("tokenizer.ggml.tokens"))
    {
        model_.token_list = model_.metadata.at("tokenizer.ggml.tokens").asStringArray();
    }
}

// Exact copy of llama.cpp's ggml_compute_fp16_to_fp32 function

std::vector<float> ModelLoader::dequantizeQ8_0(const uint8_t *data, size_t n_elements, const std::string &tensor_name)
{
    if (!data || n_elements == 0)
        return {};
    const size_t QK = 32;
    size_t full = (n_elements / QK) * QK;
    size_t tail = n_elements - full;
    std::vector<float> out(n_elements, 0.0f);
    if (full)
        dequantize_row_q8_0(reinterpret_cast<const block_q8_0 *>(data), out.data(), (int64_t)full);
    if (tail)
    {
        float tmp[QK];
        dequantize_row_q8_0(reinterpret_cast<const block_q8_0 *>(data) + (full / QK), tmp, QK);
        std::memcpy(out.data() + full, tmp, tail * sizeof(float));
    }
    bool found_nan = false;
    for (float &v : out)
    {
        if (std::isnan(v) || std::isinf(v))
        {
            found_nan = true;
            v = 0.0f;
        }
    }
    if (found_nan)
    {
        LOG_WARN("dequantizeQ8_0: NaN/Inf detected in tensor '" << tensor_name << "' replaced with 0");
    }
    else if (!out.empty() && std::isnan(out[0]))
    {
        LOG_ERROR("dequantizeQ8_0: first element NaN prior to scrub for tensor '" << tensor_name << "'");
    }
    return out;
}

bool ModelLoader::validateModel()
{
    if (model_.architecture.empty() || model_.architecture == "unknown")
    {
        LOG_ERROR("validateModel: No architecture specified");
        return false;
    }
    if (model_.architecture != "qwen2")
    {
        LOG_ERROR("validateModel: Unsupported architecture '" << model_.architecture << "' (expected 'qwen2')");
        return false;
    }
    if (model_.tensors.empty())
    {
        LOG_ERROR("validateModel: No tensors found");
        return false;
    }
    LOG_INFO("validateModel: Architecture '" << model_.architecture << "' is supported");
    return true;
}

TransformerLayerConfig ModelLoader::createLayerConfig() const
{
    TransformerLayerConfig cfg{};
    cfg.n_head = static_cast<int>(model_.head_count);
    cfg.n_head_kv = static_cast<int>(model_.head_count_kv ? model_.head_count_kv : model_.head_count);
    cfg.head_dim = (model_.embedding_length && model_.head_count) ? static_cast<int>(model_.embedding_length / model_.head_count) : 0;
    cfg.d_model = static_cast<int>(model_.embedding_length);
    cfg.d_ff = static_cast<int>(model_.feed_forward_length);
    cfg.vocab_size = static_cast<int>(model_.token_list.size());
    cfg.max_seq_len = static_cast<int>(model_.context_length);
    cfg.n_layers = static_cast<int>(model_.block_count);
    cfg.eps = 1e-5f; // default RMSNorm epsilon; TODO: pull from metadata if present
    return cfg;
}

bool ModelLoader::supportsQuantization(GGUFTensorType type) const
{
    switch (type)
    {
    case GGUFTensorType::F32:
    case GGUFTensorType::F16:
    case GGUFTensorType::Q4_0:
    case GGUFTensorType::Q4_1:
    case GGUFTensorType::Q5_0:
    case GGUFTensorType::Q5_1:
    case GGUFTensorType::Q8_0:
    case GGUFTensorType::Q2_K:
    case GGUFTensorType::Q3_K:
    case GGUFTensorType::Q4_K:
    case GGUFTensorType::Q5_K:
    case GGUFTensorType::Q6_K:
    case GGUFTensorType::Q8_K:
    case GGUFTensorType::IQ2_XXS:
    case GGUFTensorType::IQ2_XS:
    case GGUFTensorType::IQ3_XXS:
    case GGUFTensorType::IQ1_S:
    case GGUFTensorType::IQ4_NL:
    case GGUFTensorType::IQ3_S:
    case GGUFTensorType::IQ2_S:
    case GGUFTensorType::IQ4_XS:
    case GGUFTensorType::IQ1_M:
        return true;
    default:
        return false;
    }
}

std::vector<float> ModelLoader::dequantizeTensor(const GGUFTensorInfo &tensor_info,
                                                 const std::vector<uint8_t> &quantized_data,
                                                 const std::string &tensor_name)
{
    static bool printed_enum_map = false;
    if (!printed_enum_map)
    {
        const auto &snap_enum = llaminar::debugEnv();
        bool dump_enum = snap_enum.loader.enum_map_debug;
        printed_enum_map = true;
        if (dump_enum)
        {
            std::cerr << "[ENUM MAP] GGUFTensorType values: F32=" << static_cast<int>(GGUFTensorType::F32)
                      << " F16=" << static_cast<int>(GGUFTensorType::F16)
                      << " Q4_0=" << static_cast<int>(GGUFTensorType::Q4_0)
                      << " Q4_1=" << static_cast<int>(GGUFTensorType::Q4_1)
                      << " Q5_0=" << static_cast<int>(GGUFTensorType::Q5_0)
                      << " Q5_1=" << static_cast<int>(GGUFTensorType::Q5_1)
                      << " Q8_0=" << static_cast<int>(GGUFTensorType::Q8_0)
                      << " Q2_K=" << static_cast<int>(GGUFTensorType::Q2_K)
                      << " Q3_K=" << static_cast<int>(GGUFTensorType::Q3_K)
                      << " Q4_K=" << static_cast<int>(GGUFTensorType::Q4_K)
                      << " Q5_K=" << static_cast<int>(GGUFTensorType::Q5_K)
                      << " Q6_K=" << static_cast<int>(GGUFTensorType::Q6_K)
                      << " Q8_K=" << static_cast<int>(GGUFTensorType::Q8_K)
                      << " Q4_K_M=" << static_cast<int>(GGUFTensorType::Q4_K_M)
                      << std::endl;
            std::cerr << "[ENUM MAP NOTE] Unsupported id: Q8_1=9 (Q4_1 & Q5_1 restored)" << std::endl;
        }
    }
    std::cerr << "[DEQ ENTRY] tensor='" << tensor_name << "' enum=" << static_cast<int>(tensor_info.type)
              << " bytes=" << quantized_data.size() << std::endl;
    LOG_TRACE("dequantizeTensor: Enter name='" << tensor_name << "' type=" << static_cast<int>(tensor_info.type)
                                               << " quantized_bytes=" << quantized_data.size());
    const IDequantizer *dq = selectDequantizer(tensor_info.type);
    if (!dq)
    {
        std::cerr << "Error: Unsupported quantization type for tensor " << tensor_info.name
                  << " (enum value=" << static_cast<int>(tensor_info.type) << ")" << std::endl;
        LOG_ERROR("dequantizeTensor: No dequantizer for type enum=" << static_cast<int>(tensor_info.type)
                                                                    << " tensor='" << tensor_name << "'");
        return {};
    }
    LOG_TRACE("dequantizeTensor: Found dequantizer for type enum=" << static_cast<int>(tensor_info.type)
                                                                   << " tensor='" << tensor_name << "'");
    return dq->run(tensor_info, quantized_data, tensor_name, *this);
}
// ---- Polymorphic Dequantizers ----
const ModelLoader::IDequantizer *ModelLoader::selectDequantizer(GGUFTensorType type) const
{
    static Q8_0Dequantizer q8_0;
    static Q4_0Dequantizer q4_0;
    static Q4KDequantizer q4k;
    struct Q4_1Dequantizer : IDequantizer
    {
        std::vector<float> run(const GGUFTensorInfo &i, const std::vector<uint8_t> &d, const std::string &n, ModelLoader &L) const override { return L.dequantizeQ4_1(d.data(), i); }
    };
    // New format dequantizers (implemented below)
    struct Q5_0Dequantizer : IDequantizer
    {
        std::vector<float> run(const GGUFTensorInfo &i, const std::vector<uint8_t> &d, const std::string &n, ModelLoader &L) const override { return L.dequantizeQ5_0(d.data(), i); }
    };
    struct Q5_1Dequantizer : IDequantizer
    {
        std::vector<float> run(const GGUFTensorInfo &i, const std::vector<uint8_t> &d, const std::string &n, ModelLoader &L) const override { return L.dequantizeQ5_1(d.data(), i); }
    };
    struct Q5KDequantizer : IDequantizer
    {
        std::vector<float> run(const GGUFTensorInfo &i, const std::vector<uint8_t> &d, const std::string &n, ModelLoader &L) const override { return L.dequantizeQ5_K(d.data(), i.type, n, i); }
    };
    struct Q2KDequantizer : IDequantizer
    {
        std::vector<float> run(const GGUFTensorInfo &i, const std::vector<uint8_t> &d, const std::string &n, ModelLoader &L) const override { return L.dequantizeQ2_K(d.data(), i.type, n, i); }
    };
    struct Q3KDequantizer : IDequantizer
    {
        std::vector<float> run(const GGUFTensorInfo &i, const std::vector<uint8_t> &d, const std::string &n, ModelLoader &L) const override { return L.dequantizeQ3_K(d.data(), i.type, n, i); }
    };
    struct Q6KDequantizer : IDequantizer
    {
        std::vector<float> run(const GGUFTensorInfo &i, const std::vector<uint8_t> &d, const std::string &n, ModelLoader &L) const override { return L.dequantizeQ6_K(d.data(), i.type, n, i); }
    };
    static Q5_0Dequantizer q5_0;
    static Q5_1Dequantizer q5_1;
    static Q5KDequantizer q5k;
    static Q2KDequantizer q2k;
    static Q3KDequantizer q3k;
    static Q6KDequantizer q6k;
    static Q4_1Dequantizer q4_1;
    // IQ family thin wrappers
    struct IQ2XXSDequantizer : IDequantizer
    {
        std::vector<float> run(const GGUFTensorInfo &i, const std::vector<uint8_t> &d, const std::string &n, ModelLoader &L) const override
        {
            size_t n_el = std::accumulate(i.dimensions.begin(), i.dimensions.end(), 1ULL, std::multiplies<uint64_t>());
            std::vector<float> out(n_el);
            dequantize_row_iq2_xxs(reinterpret_cast<const block_iq2_xxs *>(d.data()), out.data(), (int64_t)n_el);
            L.logDequantStats(n, GGUFTensorType::IQ2_XXS, out, 8);
            return out;
        }
    };
    struct IQ2XSDequantizer : IDequantizer
    {
        std::vector<float> run(const GGUFTensorInfo &i, const std::vector<uint8_t> &d, const std::string &n, ModelLoader &L) const override
        {
            size_t n_el = std::accumulate(i.dimensions.begin(), i.dimensions.end(), 1ULL, std::multiplies<uint64_t>());
            std::vector<float> out(n_el);
            dequantize_row_iq2_xs(reinterpret_cast<const block_iq2_xs *>(d.data()), out.data(), (int64_t)n_el);
            L.logDequantStats(n, GGUFTensorType::IQ2_XS, out, 8);
            return out;
        }
    };
    struct IQ3XXSDequantizer : IDequantizer
    {
        std::vector<float> run(const GGUFTensorInfo &i, const std::vector<uint8_t> &d, const std::string &n, ModelLoader &L) const override
        {
            size_t n_el = std::accumulate(i.dimensions.begin(), i.dimensions.end(), 1ULL, std::multiplies<uint64_t>());
            std::vector<float> out(n_el);
            dequantize_row_iq3_xxs(reinterpret_cast<const block_iq3_xxs *>(d.data()), out.data(), (int64_t)n_el);
            L.logDequantStats(n, GGUFTensorType::IQ3_XXS, out, 8);
            return out;
        }
    };
    struct IQ2SDequantizer : IDequantizer
    {
        std::vector<float> run(const GGUFTensorInfo &i, const std::vector<uint8_t> &d, const std::string &n, ModelLoader &L) const override
        {
            size_t n_el = std::accumulate(i.dimensions.begin(), i.dimensions.end(), 1ULL, std::multiplies<uint64_t>());
            std::vector<float> out(n_el);
            dequantize_row_iq2_s(reinterpret_cast<const block_iq2_s *>(d.data()), out.data(), (int64_t)n_el);
            L.logDequantStats(n, GGUFTensorType::IQ2_S, out, 8);
            return out;
        }
    };
    struct IQ3SDequantizer : IDequantizer
    {
        std::vector<float> run(const GGUFTensorInfo &i, const std::vector<uint8_t> &d, const std::string &n, ModelLoader &L) const override
        {
            size_t n_el = std::accumulate(i.dimensions.begin(), i.dimensions.end(), 1ULL, std::multiplies<uint64_t>());
            std::vector<float> out(n_el);
            dequantize_row_iq3_s(reinterpret_cast<const block_iq3_s *>(d.data()), out.data(), (int64_t)n_el);
            L.logDequantStats(n, GGUFTensorType::IQ3_S, out, 8);
            return out;
        }
    };
    struct IQ1SDequantizer : IDequantizer
    {
        std::vector<float> run(const GGUFTensorInfo &i, const std::vector<uint8_t> &d, const std::string &n, ModelLoader &L) const override
        {
            size_t n_el = std::accumulate(i.dimensions.begin(), i.dimensions.end(), 1ULL, std::multiplies<uint64_t>());
            std::vector<float> out(n_el);
            dequantize_row_iq1_s(reinterpret_cast<const block_iq1_s *>(d.data()), out.data(), (int64_t)n_el);
            L.logDequantStats(n, GGUFTensorType::IQ1_S, out, 8);
            return out;
        }
    };
    struct IQ1MDequantizer : IDequantizer
    {
        std::vector<float> run(const GGUFTensorInfo &i, const std::vector<uint8_t> &d, const std::string &n, ModelLoader &L) const override
        {
            size_t n_el = std::accumulate(i.dimensions.begin(), i.dimensions.end(), 1ULL, std::multiplies<uint64_t>());
            std::vector<float> out(n_el);
            dequantize_row_iq1_m(reinterpret_cast<const block_iq1_m *>(d.data()), out.data(), (int64_t)n_el);
            L.logDequantStats(n, GGUFTensorType::IQ1_M, out, 8);
            return out;
        }
    };
    struct IQ4NLDequantizer : IDequantizer
    {
        std::vector<float> run(const GGUFTensorInfo &i, const std::vector<uint8_t> &d, const std::string &n, ModelLoader &L) const override
        {
            size_t n_el = std::accumulate(i.dimensions.begin(), i.dimensions.end(), 1ULL, std::multiplies<uint64_t>());
            std::vector<float> out(n_el);
            dequantize_row_iq4_nl(reinterpret_cast<const block_iq4_nl *>(d.data()), out.data(), (int64_t)n_el);
            L.logDequantStats(n, GGUFTensorType::IQ4_NL, out, 8);
            return out;
        }
    };
    struct IQ4XSDequantizer : IDequantizer
    {
        std::vector<float> run(const GGUFTensorInfo &i, const std::vector<uint8_t> &d, const std::string &n, ModelLoader &L) const override
        {
            size_t n_el = std::accumulate(i.dimensions.begin(), i.dimensions.end(), 1ULL, std::multiplies<uint64_t>());
            std::vector<float> out(n_el);
            dequantize_row_iq4_xs(reinterpret_cast<const block_iq4_xs *>(d.data()), out.data(), (int64_t)n_el);
            L.logDequantStats(n, GGUFTensorType::IQ4_XS, out, 8);
            return out;
        }
    };
    static IQ2XXSDequantizer iq2xxs;
    static IQ2XSDequantizer iq2xs;
    static IQ3XXSDequantizer iq3xxs;
    static IQ2SDequantizer iq2s;
    static IQ3SDequantizer iq3s;
    static IQ1SDequantizer iq1s;
    static IQ1MDequantizer iq1m;
    static IQ4NLDequantizer iq4nl;
    static IQ4XSDequantizer iq4xs;
    switch (type)
    {
    case GGUFTensorType::Q8_0:
        LOG_TRACE("selectDequantizer: Returning Q8_0Dequantizer");
        return &q8_0;
    case GGUFTensorType::Q4_0:
        LOG_TRACE("selectDequantizer: Returning Q4_0Dequantizer");
        return &q4_0;
    case GGUFTensorType::Q4_1:
        return &q4_1;
    case GGUFTensorType::Q4_K: // fallthrough intentional for grouped variants later / alias covers Q4_K_M
        LOG_TRACE("selectDequantizer: Returning Q4KDequantizer (Q4_K / alias)");
        return &q4k;
    case GGUFTensorType::Q5_0:
        return &q5_0;
    case GGUFTensorType::Q5_1:
        return &q5_1;
    case GGUFTensorType::Q5_K:
        return &q5k;
    case GGUFTensorType::Q2_K:
        return &q2k;
    case GGUFTensorType::Q3_K:
        return &q3k;
    case GGUFTensorType::Q6_K:
        return &q6k;
    case GGUFTensorType::IQ2_XXS:
        return &iq2xxs;
    case GGUFTensorType::IQ2_XS:
        return &iq2xs;
    case GGUFTensorType::IQ3_XXS:
        return &iq3xxs;
    case GGUFTensorType::IQ1_S:
        return &iq1s;
    case GGUFTensorType::IQ4_NL:
        return &iq4nl;
    case GGUFTensorType::IQ3_S:
        return &iq3s;
    case GGUFTensorType::IQ2_S:
        return &iq2s;
    case GGUFTensorType::IQ4_XS:
        return &iq4xs;
    case GGUFTensorType::IQ1_M:
        return &iq1m;
    default:
        LOG_TRACE("selectDequantizer: No dequantizer for enum=" << static_cast<int>(type));
        return nullptr;
    }
}

std::vector<float> ModelLoader::Q8_0Dequantizer::run(const GGUFTensorInfo &info, const std::vector<uint8_t> &data, const std::string &name, ModelLoader &loader) const
{
    size_t n = std::accumulate(info.dimensions.begin(), info.dimensions.end(), 1ULL, std::multiplies<uint64_t>());
    return loader.dequantizeQ8_0(data.data(), n, name);
}

// Added legacy *_1 format wrappers using upstream ggml row dequant routines
std::vector<float> ModelLoader::dequantizeQ4_1(const uint8_t *data, const GGUFTensorInfo &info)
{
    size_t n_elements = std::accumulate(info.dimensions.begin(), info.dimensions.end(), 1ULL, std::multiplies<uint64_t>());
    if (!data || n_elements == 0)
        return {};
    std::vector<float> out(n_elements);
    // Use upstream ggml dequantization (handles blocks of 32 internally)
    dequantize_row_q4_1(reinterpret_cast<const block_q4_1 *>(data), out.data(), static_cast<int64_t>(n_elements));
    logDequantStats(info.name, GGUFTensorType::Q4_1, out, 8);
    return out;
}

std::vector<float> ModelLoader::dequantizeQ5_1(const uint8_t *data, const GGUFTensorInfo &info)
{
    size_t n_elements = std::accumulate(info.dimensions.begin(), info.dimensions.end(), 1ULL, std::multiplies<uint64_t>());
    if (!data || n_elements == 0)
        return {};
    std::vector<float> out(n_elements);
    dequantize_row_q5_1(reinterpret_cast<const block_q5_1 *>(data), out.data(), static_cast<int64_t>(n_elements));
    logDequantStats(info.name, GGUFTensorType::Q5_1, out, 8);
    return out;
}

std::vector<float> ModelLoader::Q4_0Dequantizer::run(const GGUFTensorInfo &info, const std::vector<uint8_t> &data, const std::string &name, ModelLoader &loader) const
{
    size_t n_elements = std::accumulate(info.dimensions.begin(), info.dimensions.end(), 1ULL, std::multiplies<uint64_t>());
    if (n_elements == 0 || data.empty())
        return {};
    std::vector<float> out(n_elements);
    dequantize_row_q4_0(reinterpret_cast<const block_q4_0 *>(data.data()), out.data(), (int64_t)n_elements);
    loader.logDequantStats(name, GGUFTensorType::Q4_0, out, 8);
    return out;
}

std::vector<float> ModelLoader::Q4KDequantizer::run(const GGUFTensorInfo &info, const std::vector<uint8_t> &data, const std::string &name, ModelLoader &loader) const
{
    size_t n = std::accumulate(info.dimensions.begin(), info.dimensions.end(), 1ULL, std::multiplies<uint64_t>());
    return loader.dequantizeQ4_K(data.data(), n, info.type, name);
}

// Thin wrapper retained for compatibility – now delegates to upstream ggml row dequant.
std::vector<float> ModelLoader::dequantizeQ4_0(const uint8_t *data, size_t n_elements)
{
    if (!data || n_elements == 0)
        return {};
    std::vector<float> out(n_elements);
    dequantize_row_q4_0(reinterpret_cast<const block_q4_0 *>(data), out.data(), (int64_t)n_elements);
    return out;
}

// ---- Q5_0 Dequant (mirrors ggml dequantize_row_q5_0 logic) ----
// Block layout (22 bytes, 32 values):
//   uint16_t d (fp16 scale)
//   uint8_t  qh[4]  (packed high bits for both halves)
//   uint8_t  qs[16] (low/high nibbles -> base 4-bit values for 2 values each)
// Reconstruction for j in [0,15]:
//   xh_0 = ((qh >> (j + 0)) & 1) << 4  (high bit for value j)
//   xh_1 = ((qh >> (j + 12)) & 1) << 4 (high bit for value j+16)
//   q = qs[j]
//   v0 = ((q & 0x0F) | xh_0) - 16   (signed 5-bit)
//   v1 = ((q >> 4)   | xh_1) - 16
//   out[j] = v0 * d; out[j+16] = v1 * d
std::vector<float> ModelLoader::dequantizeQ5_0(const uint8_t *data, const GGUFTensorInfo &info)
{
    size_t n_elements = std::accumulate(info.dimensions.begin(), info.dimensions.end(), 1ULL, std::multiplies<uint64_t>());
    if (!data || n_elements == 0)
        return {};
    std::vector<float> out(n_elements);
    dequantize_row_q5_0(reinterpret_cast<const block_q5_0 *>(data), out.data(), (int64_t)n_elements);
    logDequantStats(info.name, GGUFTensorType::Q5_0, out, 8);
    return out;
}

// ---- Q4_K Dequant (spec implementation) ----
// Follows llama.cpp's ggml-quants.c::dequantize_row_q4_K logic for CPU path.
// block_q4_K layout per 256 (QK_K) elements:
//   uint16_t d; uint16_t dmin; uint8_t scales[12]; uint8_t qs[128]; total 144 bytes.
// Value reconstruction per 64-value segment uses two scale/min pairs.
std::vector<float> ModelLoader::dequantizeQ4_K(const uint8_t *data, size_t n_elements, GGUFTensorType type, const std::string &tensor_name)
{
    if (!data || n_elements == 0)
        return {};
    std::vector<float> out(n_elements);
    dequantize_row_q4_K(reinterpret_cast<const block_q4_K *>(data), out.data(), (int64_t)n_elements);
    logDequantStats(tensor_name, type, out, 8);
    return out;
}

// (Removed obsolete Q5_0 duplicate & Q5_1 support region)

// ---------------- Quant shard cache & partial load helpers (minimal implementation) ----------------
const ModelLoader::CachedFullTensor *ModelLoader::getOrCacheFullQuantTensor(const std::string &tensor_name, const GGUFTensorInfo &info)
{
    // Fast path: lookup
    {
        std::lock_guard<std::mutex> lock(quant_cache_mutex_);
        auto it = quant_full_cache_.find(tensor_name);
        if (it != quant_full_cache_.end())
        {
            it->second.last_access = ++quant_cache_clock_;
            quant_cache_hits_++;
            quant_cache_loads_++; // count hit as a load for stats consistency
            return &it->second;
        }
    }
    quant_cache_misses_++;
    // Simplified: just read declared size; deprecated Q5_1 variant detection removed.
    size_t logical_read_size = info.size_bytes;
    const GGUFTensorInfo &adjusted_info = info; // retained for interface uniformity

    // Re-seek and read selected number of bytes
    file_stream_.seekg(model_.data_offset + info.offset, std::ios::beg);
    std::vector<uint8_t> raw(logical_read_size);
    if (!file_stream_.read(reinterpret_cast<char *>(raw.data()), raw.size()))
    {
        LOG_ERROR("getOrCacheFullQuantTensor: read failed for tensor '" << tensor_name << "' (requested=" << logical_read_size << ")");
        return nullptr;
    }
    size_t n_elems = 1;
    for (auto d : info.dimensions)
        n_elems *= d;
    std::vector<float> decoded;
    if (info.isQuantized())
        decoded = dequantizeTensor(adjusted_info, raw, tensor_name);
    else
    {
        if (info.type == GGUFTensorType::F32)
        {
            decoded.resize(n_elems);
            std::memcpy(decoded.data(), raw.data(), n_elems * sizeof(float));
        }
        else if (info.type == GGUFTensorType::F16)
        {
            decoded.resize(n_elems);
            for (size_t i = 0; i < n_elems; ++i)
            {
                uint16_t h;
                std::memcpy(&h, raw.data() + 2 * i, 2);
                decoded[i] = ggml_compute_fp16_to_fp32(h);
            }
        }
        else
        {
            return nullptr;
        }
    }
    // Global safety: scrub unexpected NaN/Inf from decoded tensor to keep parity ops stable
    bool any_bad = false;
    for (float &v : decoded)
    {
        if (std::isnan(v) || std::isinf(v))
        {
            v = 0.0f;
            any_bad = true;
        }
    }
    if (any_bad)
    {
        LOG_WARN("getOrCacheFullQuantTensor: scrubbed NaN/Inf values in tensor '" << tensor_name << "' (format enum=" << (int)info.type << ")");
    }
    auto cache_bytes_limit = quantShardCacheMaxBytes();
    const size_t new_bytes = decoded.size() * sizeof(float);
    {
        std::lock_guard<std::mutex> lock(quant_cache_mutex_);
        // Simple LRU eviction until space
        while (cache_bytes_limit > 0 && quant_cache_bytes_ + new_bytes > cache_bytes_limit && !quant_full_cache_.empty())
        {
            auto victim_it = std::min_element(quant_full_cache_.begin(), quant_full_cache_.end(), [](auto &a, auto &b)
                                              { return a.second.last_access < b.second.last_access; });
            if (victim_it == quant_full_cache_.end())
                break;
            quant_cache_bytes_ -= victim_it->second.bytes;
            quant_full_cache_.erase(victim_it);
            quant_cache_evictions_++;
        }
        auto &entry = quant_full_cache_[tensor_name];
        entry.data.swap(decoded);
        entry.shape.clear();
        for (auto d : info.dimensions)
            entry.shape.push_back(static_cast<int>(d));
        entry.bytes = new_bytes;
        entry.last_access = ++quant_cache_clock_;
        quant_cache_bytes_ += new_bytes;
        quant_cache_loads_++;
        return &entry;
    }
}

size_t ModelLoader::quantShardCacheMaxBytes() const
{
    const auto &snap = llaminar::debugEnv();
    long long mb = snap.loader.shard_cache_max_mb; // >=0
    if (mb == 0)
        return 0; // disabled
    return static_cast<size_t>(mb) * 1024ULL * 1024ULL;
}

void ModelLoader::clearQuantShardCache()
{
    std::lock_guard<std::mutex> lock(quant_cache_mutex_);
    quant_full_cache_.clear();
    quant_cache_bytes_ = 0;
    quant_cache_loads_ = 0;
    quant_cache_hits_ = 0;
    quant_cache_misses_ = 0;
    quant_cache_evictions_ = 0;
}

ModelLoader::QuantShardCacheStats ModelLoader::getQuantShardCacheStats() const
{
    QuantShardCacheStats s;
    s.loads = quant_cache_loads_;
    s.cache_hits = quant_cache_hits_;
    s.cache_misses = quant_cache_misses_;
    s.evictions = quant_cache_evictions_;
    s.bytes_resident = quant_cache_bytes_;
    return s;
}

std::vector<std::string> ModelLoader::getTensorNames() const
{
    std::vector<std::string> names;
    names.reserve(model_.tensors.size());
    for (auto const &t : model_.tensors)
        names.push_back(t.name);
    return names;
}

bool ModelLoader::loadTensorColumnShards(const std::string &tensor_name,
                                         const std::vector<int> &col_offsets,
                                         const std::vector<int> &col_counts,
                                         const std::vector<float *> &dests)
{
    if (col_offsets.size() != col_counts.size() || col_offsets.size() != dests.size())
    {
        LOG_ERROR("loadTensorColumnShards: mismatched shard vector sizes");
        return false;
    }
    const GGUFTensorInfo *info = model_.findTensor(tensor_name);
    if (!info)
    {
        LOG_ERROR("loadTensorColumnShards: tensor not found '" << tensor_name << "'");
        return false;
    }
    if (info->dimensions.size() != 2)
    {
        LOG_WARN("loadTensorColumnShards: only 2D tensors supported; falling back to full");
    }
    size_t rows = info->dimensions[0];
    size_t cols = (info->dimensions.size() > 1) ? info->dimensions[1] : 1;
    // Fallback: use (cached) full tensor and slice
    const CachedFullTensor *full = getOrCacheFullQuantTensor(tensor_name, *info);
    if (!full)
        return false;
    const float *src = full->data.data();
    for (size_t s = 0; s < col_offsets.size(); ++s)
    {
        int off = col_offsets[s];
        int count = col_counts[s];
        float *dst = dests[s];
        if (!dst || off < 0 || count <= 0 || static_cast<size_t>(off + count) > cols)
        {
            // Downgraded to WARN because higher layer logic intentionally probes a speculative
            // shard layout and falls back to full-load scatter when unsupported for a quant type.
            LOG_WARN("loadTensorColumnShards: invalid shard spec index=" << s << ", falling back to full tensor path");
            return false;
        }
        for (size_t r = 0; r < rows; ++r)
        {
            const float *row_src = src + r * cols + off;
            std::memcpy(dst + r * count, row_src, sizeof(float) * count);
        }
    }
    return true;
}

bool ModelLoader::loadTensorRowShard(const std::string &tensor_name,
                                     int row_offset,
                                     int row_count,
                                     float *dest)
{
    if (row_count <= 0 || row_offset < 0)
        return false;
    const GGUFTensorInfo *info = model_.findTensor(tensor_name);
    if (!info)
        return false;
    if (info->dimensions.size() != 2)
        return false;
    size_t rows = info->dimensions[0];
    size_t cols = info->dimensions[1];
    if (static_cast<size_t>(row_offset + row_count) > rows)
        return false;
    const CachedFullTensor *full = getOrCacheFullQuantTensor(tensor_name, *info);
    if (!full)
        return false;
    const float *src = full->data.data() + static_cast<size_t>(row_offset) * cols;
    std::memcpy(dest, src, sizeof(float) * static_cast<size_t>(row_count) * cols);
    return true;
}

// ---- Simplified K-family dequants now directly call upstream ggml row functions ----
std::vector<float> ModelLoader::dequantizeQ2_K(const uint8_t *data, GGUFTensorType type, const std::string &tensor_name, const GGUFTensorInfo &info)
{
    if (!data)
        return {};
    size_t n = std::accumulate(info.dimensions.begin(), info.dimensions.end(), 1ULL, std::multiplies<uint64_t>());
    std::vector<float> out(n);
    dequantize_row_q2_K(reinterpret_cast<const block_q2_K *>(data), out.data(), (int64_t)n);
    logDequantStats(tensor_name, type, out, 8);
    return out;
}
std::vector<float> ModelLoader::dequantizeQ3_K(const uint8_t *data, GGUFTensorType type, const std::string &tensor_name, const GGUFTensorInfo &info)
{
    if (!data)
        return {};
    size_t n = std::accumulate(info.dimensions.begin(), info.dimensions.end(), 1ULL, std::multiplies<uint64_t>());
    std::vector<float> out(n);
    dequantize_row_q3_K(reinterpret_cast<const block_q3_K *>(data), out.data(), (int64_t)n);
    logDequantStats(tensor_name, type, out, 8);
    return out;
}
std::vector<float> ModelLoader::dequantizeQ5_K(const uint8_t *data, GGUFTensorType type, const std::string &tensor_name, const GGUFTensorInfo &info)
{
    if (!data)
        return {};
    size_t n = std::accumulate(info.dimensions.begin(), info.dimensions.end(), 1ULL, std::multiplies<uint64_t>());
    std::vector<float> out(n);
    dequantize_row_q5_K(reinterpret_cast<const block_q5_K *>(data), out.data(), (int64_t)n);
    logDequantStats(tensor_name, type, out, 8);
    return out;
}
std::vector<float> ModelLoader::dequantizeQ6_K(const uint8_t *data, GGUFTensorType type, const std::string &tensor_name, const GGUFTensorInfo &info)
{
    if (!data)
        return {};
    size_t n = std::accumulate(info.dimensions.begin(), info.dimensions.end(), 1ULL, std::multiplies<uint64_t>());
    std::vector<float> out(n);
    dequantize_row_q6_K(reinterpret_cast<const block_q6_K *>(data), out.data(), (int64_t)n);
    logDequantStats(tensor_name, type, out, 8);
    return out;
}

bool ModelLoader::readArray(GGUFValue &value)
{
    // Read array type and length
    uint32_t array_type;
    uint64_t array_length;
    if (!readValue(array_type) || !readValue(array_length))
    {
        return false;
    }

    LOG_TRACE("readArray: type=" << array_type << ", length=" << array_length);

    // Reserve space for the array data
    value.data.clear();
    value.data.reserve(12 + array_length * 20); // Estimate

    // Store array type and length at the beginning
    value.data.resize(12);
    std::memcpy(value.data.data(), &array_type, 4);
    std::memcpy(value.data.data() + 4, &array_length, 8);

    // Read array elements based on type
    GGUFValueType element_type = static_cast<GGUFValueType>(array_type);

    switch (element_type)
    {
    case GGUFValueType::STRING:
    {
        // Read string array - this is what we need for vocabulary
        for (uint64_t i = 0; i < array_length; ++i)
        {
            uint64_t str_len;
            if (!readValue(str_len))
            {
                return false;
            }

            // Store string length
            size_t len_offset = value.data.size();
            value.data.resize(value.data.size() + 8);
            std::memcpy(value.data.data() + len_offset, &str_len, 8);

            // Store string data
            size_t str_offset = value.data.size();
            value.data.resize(value.data.size() + str_len);
            if (!file_stream_.read(reinterpret_cast<char *>(value.data.data() + str_offset), str_len))
            {
                LOG_ERROR("readArray: Failed to read string data");
                return false;
            }
        }
        LOG_INFO("readArray: Successfully read string array with " << array_length << " elements");
        break;
    }

    case GGUFValueType::UINT32:
    case GGUFValueType::INT32:
    {
        // Read 32-bit integer array
        for (uint64_t i = 0; i < array_length; ++i)
        {
            uint32_t val;
            if (!readValue(val))
            {
                return false;
            }
            size_t offset = value.data.size();
            value.data.resize(value.data.size() + 4);
            std::memcpy(value.data.data() + offset, &val, 4);
        }
        LOG_INFO("readArray: Successfully read int32 array with " << array_length << " elements");
        break;
    }

    default:
    {
        // For other types, skip the data for now
        LOG_WARN("readArray: Skipping unsupported array type " << array_type);
        size_t element_size = 0;
        switch (element_type)
        {
        case GGUFValueType::UINT8:
        case GGUFValueType::INT8:
        case GGUFValueType::BOOL:
            element_size = 1;
            break;
        case GGUFValueType::UINT16:
        case GGUFValueType::INT16:
            element_size = 2;
            break;
        case GGUFValueType::FLOAT32:
            element_size = 4;
            break;
        case GGUFValueType::UINT64:
        case GGUFValueType::INT64:
        case GGUFValueType::FLOAT64:
            element_size = 8;
            break;
        default:
            LOG_ERROR("readArray: Unknown array element type: " << array_type);
            return false;
        }

        if (element_size > 0)
        {
            size_t bytes_to_skip = array_length * element_size;
            if (!file_stream_.seekg(bytes_to_skip, std::ios::cur))
            {
                LOG_ERROR("readArray: Failed to skip array data");
                return false;
            }
        }
        value.data.clear(); // Clear data since we didn't read it
        break;
    }
    }

    return true;
}

// Template method definitions and explicit instantiations
// These were moved here to ensure linkage after introducing new code sections above.
template <typename T>
T GGUFValue::as() const
{
    if constexpr (std::is_same_v<T, int32_t>)
    {
        int32_t v;
        std::memcpy(&v, data.data(), sizeof(v));
        return v;
    }
    else if constexpr (std::is_same_v<T, uint32_t>)
    {
        uint32_t v;
        std::memcpy(&v, data.data(), sizeof(v));
        return v;
    }
    else if constexpr (std::is_same_v<T, float>)
    {
        float v;
        std::memcpy(&v, data.data(), sizeof(v));
        return v;
    }
    else if constexpr (std::is_same_v<T, std::string>)
    {
        return asString();
    }
    else
    {
        static_assert(!sizeof(T *), "Unsupported GGUFValue::as<T>() instantiation");
    }
}

template <typename T>
T GGUFModel::getMetadata(const std::string &key, T default_value) const
{
    auto it = metadata.find(key);
    if (it == metadata.end())
        return default_value;
    return it->second.as<T>();
}

template std::string GGUFModel::getMetadata<std::string>(const std::string &key, std::string default_value) const;
template uint32_t GGUFModel::getMetadata<uint32_t>(const std::string &key, uint32_t default_value) const;
template int32_t GGUFModel::getMetadata<int32_t>(const std::string &key, int32_t default_value) const;
template float GGUFModel::getMetadata<float>(const std::string &key, float default_value) const;

template int32_t GGUFValue::as<int32_t>() const;
template uint32_t GGUFValue::as<uint32_t>() const;
template float GGUFValue::as<float>() const;
template std::string GGUFValue::as<std::string>() const;