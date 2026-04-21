#pragma once
#include <string>
#include <vector>
#include <cstddef>

namespace llaminar2
{

// Forward declarations
struct GGUFModel;
class ModelLoader;

struct TensorSizeInfo
{
    std::string name;
    size_t native_bytes = 0;
    std::string quant_type;     // "F32", "Q8_0", "Q4_0", "IQ4_NL", etc.
    size_t elements = 0;
    size_t K = 0;               // Inner dimension (last dim) for packed size estimation
    int layer_index = -1;       // -1 for non-layer tensors (embedding, lm_head, norms)
};

struct ModelMemoryProfile
{
    // Architecture
    std::string architecture;
    int n_layers = 0;
    int d_model = 0;
    int d_ff = 0;
    int n_heads = 0;
    int n_kv_heads = 0;
    int head_dim = 0;
    int vocab_size = 0;
    int max_seq_len = 0;

    // Weight sizing
    size_t total_native_bytes = 0;
    std::vector<TensorSizeInfo> tensors;

    // Factory methods
    static ModelMemoryProfile fromGGUF(const GGUFModel& model);

    // Query helpers
    size_t weightBytesForLayers(int first_layer, int last_layer) const;
    size_t embeddingBytes() const;
    size_t lmHeadBytes() const;
    size_t normBytes() const;
    size_t layerWeightBytes(int layer) const;
};

} // namespace llaminar2
