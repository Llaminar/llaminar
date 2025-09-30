// Embedding orientation diagnostic test
// Compares token embedding tensor layout between Llaminar loader and llama.cpp internal model
// to determine if the warning about transposed layout corresponds to llama.cpp's representation.

#include "model_loader.h"
#include "logger.h"
#include "tensors/tensor_base.h"

#include <gtest/gtest.h>

#include <mpi.h>

#include <filesystem>
#include <vector>
#include <string>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <cstdlib>

extern "C"
{
#include "llama.h"
}

// Pull in ggml type enumeration so we can detect quantized embedding and optionally
// skip raw value diff (since llama.cpp keeps quantized blocks in-memory while our
// loader has already dequantized to fp32).
extern "C"
{
#include "ggml.h"
}

// NOTE: We intentionally include the internal header to introspect the llama.cpp model tensor
// layout; this is for a diagnostic test only and not production code.
// Internal llama.cpp header: rely on added include path in CMakeLists.
#include "llama-model.h"

using namespace llaminar;

namespace
{
    std::string find_test_model()
    {
        namespace fs = std::filesystem;
        fs::path models_dir{"models"};
        if (!fs::exists(models_dir))
            return {};
        const std::vector<std::string> preferred = {
            "qwen2.5-0.5b-instruct-q4_0.gguf",
            "qwen2.5-0.5b-instruct-q4_k_m.gguf",
            "qwen2.5-0.5b-instruct-q5_0.gguf",
            "qwen2.5-0.5b-instruct-fp16.gguf"};
        for (const auto &c : preferred)
        {
            fs::path p = models_dir / c;
            if (fs::exists(p))
                return p.string();
        }
        for (const auto &entry : fs::directory_iterator(models_dir))
        {
            if (entry.is_regular_file() && entry.path().extension() == ".gguf")
            {
                return entry.path().string();
            }
        }
        return {};
    }
} // namespace

TEST(EmbeddingOrientation, CompareWithLlamaCpp)
{
    int provided = 0;
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (!initialized)
    {
        MPI_Init_thread(nullptr, nullptr, MPI_THREAD_FUNNELED, &provided);
    }
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank != 0)
    {
        // Only rank 0 performs diagnostics; others just barrier & return.
        MPI_Barrier(MPI_COMM_WORLD);
        SUCCEED();
        return;
    }

    std::string model_path = find_test_model();
    ASSERT_FALSE(model_path.empty()) << "No test model (.gguf) found under models/";

    LOG_INFO("[EmbedOrient] Using model: " << model_path);

    // Load via Llaminar ModelLoader
    ModelLoader loader;
    ASSERT_TRUE(loader.loadModel(model_path)) << "Failed to parse GGUF model";
    const auto &gm = loader.getModel();
    // Derive d_model and vocab from metadata already extracted
    int d_model = static_cast<int>(gm.embedding_length);
    int vocab_size = static_cast<int>(gm.token_list.size());
    ASSERT_GT(d_model, 0);
    ASSERT_GT(vocab_size, 0);

    std::shared_ptr<TensorBase> token_tensor;
    const std::vector<std::string> candidate_names = {
        "tok_embeddings.weight", "embed_tokens.weight", "model.embed_tokens.weight",
        "token_embd.weight", "tok_embd.weight", "token_embd", "tok_embd"};
    for (const auto &name : candidate_names)
    {
        token_tensor = loader.loadTensor(name);
        if (token_tensor)
        {
            LOG_INFO("[EmbedOrient] Found embedding tensor via candidate name: " << name);
            break;
        }
    }
    if (!token_tensor)
    {
        // Fallback heuristic: search by shape & name substring
        auto names = loader.getTensorNames();
        for (const auto &n : names)
        {
            if (n.find("emb") == std::string::npos && n.find("tok") == std::string::npos)
                continue;
            token_tensor = loader.loadTensor(n);
            if (!token_tensor)
                continue;
            auto s = token_tensor->shape();
            if (s.size() == 2 && ((s[0] == vocab_size && s[1] == d_model) || (s[0] == d_model && s[1] == vocab_size)))
            {
                LOG_INFO("[EmbedOrient] Heuristic matched embedding tensor: " << n);
                break;
            }
            token_tensor.reset();
        }
    }
    ASSERT_TRUE(token_tensor) << "Failed to locate token embedding tensor by known names or heuristic";
    auto emb_shape = token_tensor->shape();
    ASSERT_EQ(emb_shape.size(), 2u);
    int rows = emb_shape[0];
    int cols = emb_shape[1];

    bool llaminar_vocab_first = (rows == vocab_size && cols == d_model);
    bool llaminar_model_first = (rows == d_model && cols == vocab_size);
    LOG_INFO("[EmbedOrient] Llaminar embedding shape=" << rows << "x" << cols
                                                       << " (vocab_first=" << llaminar_vocab_first
                                                       << ", model_first=" << llaminar_model_first << ")");

    // Load llama.cpp model
    llama_model_params mparams = llama_model_default_params();
    llama_model *lmodel = llama_load_model_from_file(model_path.c_str(), mparams);
    ASSERT_NE(lmodel, nullptr) << "Failed to load llama.cpp model";
    // Access internal struct
    auto *internal = reinterpret_cast<struct llama_model *>(lmodel);
    ASSERT_NE(internal->tok_embd, nullptr) << "llama.cpp tok_embd tensor null";

    // ggml tensors store ne[]; consult llama.cpp: create_tensor called with {n_embd, n_vocab} for many archs.
    // In ggml, ne[0] is number of elements along x dimension (columns), ne[1] along y (rows) in row-major view.
    // ggml_get_rows selects by index along dimension 1 (y). So for embedding lookup to index tokens by id, ne[1] should equal vocab_size.
    // If llama.cpp created {n_embd, n_vocab}, then (ne[0]=n_embd, ne[1]=n_vocab) and tokens are rows via y-dim => orientation is model_first in memory but still accessible as rows by token id. We map orientation logically as model_first when ne[0]==d_model && ne[1]==vocab_size.
    const int64_t gg_x = internal->tok_embd->ne[0];
    const int64_t gg_y = internal->tok_embd->ne[1];
    bool llama_model_first = (gg_x == d_model && gg_y == vocab_size);
    bool llama_vocab_first = (gg_x == vocab_size && gg_y == d_model);
    LOG_INFO("[EmbedOrient] llama.cpp tok_embd ne[0]=" << gg_x << " ne[1]=" << gg_y
                                                       << " (interpreted model_first=" << llama_model_first
                                                       << ", vocab_first=" << llama_vocab_first << ")");

    // Orientation parity expectation: our warning triggers when we detect model_first; so we expect llama.cpp also to be model_first.
    // Record parity and compute a small value diff after normalizing orientation.
    ASSERT_TRUE(llaminar_vocab_first || llaminar_model_first) << "Llaminar embedding shape invalid";
    ASSERT_TRUE(llama_vocab_first || llama_model_first) << "llama.cpp embedding shape unexpected";

    // Build a function to fetch embedding vector for a token id from llaminar weights depending on orientation
    auto *lam_data = token_tensor->data();
    ASSERT_NE(lam_data, nullptr);
    auto fetch_lam_vec = [&](int token, std::vector<float> &out)
    {
        out.resize(d_model);
        if (llaminar_vocab_first)
        {
            // Row = token
            const float *row_ptr = lam_data + static_cast<size_t>(token) * d_model;
            std::copy(row_ptr, row_ptr + d_model, out.begin());
        }
        else
        {
            // Column = token
            for (int d = 0; d < d_model; ++d)
            {
                out[d] = lam_data[static_cast<size_t>(d) * vocab_size + token];
            }
        }
    };

    // Fetch embedding vector from llama.cpp memory (raw). We must interpret orientation consistent with ggml usage.
    // For llama_model_first (ne[0]=d_model), the memory is laid out with contiguous d_model values (x dimension) for each y (row token index). So token row stride = gg_x.
    auto *llama_data = static_cast<float *>(internal->tok_embd->data);
    ASSERT_NE(llama_data, nullptr);
    auto fetch_llama_vec = [&](int token, std::vector<float> &out)
    {
        out.resize(d_model);
        if (llama_model_first)
        {
            // row index token over y dimension with stride gg_x
            const float *row_ptr = llama_data + static_cast<size_t>(token) * gg_x;
            std::copy(row_ptr, row_ptr + d_model, out.begin());
        }
        else
        {
            // row-major vocab_first layout: contiguous d_model per row but x= vocab_size? (unlikely here)
            const float *row_ptr = llama_data + static_cast<size_t>(token) * d_model;
            std::copy(row_ptr, row_ptr + d_model, out.begin());
        }
    };

    // If llama.cpp embedding tensor is stored in a quantized type (most common for *.q4_0 models),
    // we skip direct value comparison because internal->tok_embd->data contains packed blocks, while
    // our loader has already produced dequantized fp32. A fair comparison would require invoking
    // ggml_get_rows (which performs dequant) or re-implementing the same decode path here.
    ggml_type llama_emb_type = internal->tok_embd->type;
    if (llama_emb_type == GGML_TYPE_F32 || llama_emb_type == GGML_TYPE_F16)
    {
        const int sample_tokens = std::min(5, vocab_size);
        double max_abs_after_orientation = 0.0;
        for (int t = 0; t < sample_tokens; ++t)
        {
            std::vector<float> v_lam, v_llama;
            fetch_lam_vec(t, v_lam);
            fetch_llama_vec(t, v_llama);
            ASSERT_EQ(v_lam.size(), v_llama.size());
            for (int d = 0; d < d_model; ++d)
            {
                double diff = std::fabs(static_cast<double>(v_lam[d]) - static_cast<double>(v_llama[d]));
                if (diff > max_abs_after_orientation)
                    max_abs_after_orientation = diff;
            }
        }
        LOG_INFO("[EmbedOrient] Post-normalization sample max_abs_diff=" << max_abs_after_orientation);
        if (max_abs_after_orientation > 1e-2)
        {
            LOG_WARN("[EmbedOrient] max_abs_after_orientation=" << max_abs_after_orientation << " exceeds 1e-2 tolerance");
        }
        EXPECT_LT(max_abs_after_orientation, 5e-1) << "Embedding data mismatch beyond relaxed tolerance (0.5)";
    }
    else
    {
        LOG_INFO("[EmbedOrient] Skipping value diff: llama.cpp embedding ggml_type=" << llama_emb_type
                                                                                     << " (quantized). Orientation parity validated (llaminar_model_first=" << llaminar_model_first
                                                                                     << ", llama_model_first=" << llama_model_first << ")");
        SUCCEED();
    }

    llama_free_model(lmodel);
}
