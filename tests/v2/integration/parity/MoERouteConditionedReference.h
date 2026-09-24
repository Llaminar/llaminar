/**
 * @file MoERouteConditionedReference.h
 * @brief Bounded bridge to the independent CPU PyTorch expert reference.
 *
 * The canonical fixture authenticates the HF checkpoint before calling this
 * bridge. Only the model descriptor, checkpoint name, and selected expert IDs
 * cross into Python: native activation/output tensors are deliberately absent
 * from the API. ReferenceGenerationLease owns cache validation/publication;
 * inference, graph capture, and runtime scheduling never call this helper.
 */
#pragma once

#include "../../utils/ReferenceGenerationLease.h"
#include "../../utils/MoERouteConditionedProof.h"

#include <cnpy.h>
#include <array>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace llaminar2::test::parity
{
    /** @brief Typed identity of a separately stored GGUF SwiGLU expert family. */
    struct MoERouteReferenceIdentity
    {
        std::filesystem::path model;
        std::filesystem::path snapshot_directory;
        std::filesystem::path artifacts;
        std::string checkpoint_prefix;
        int gguf_block = -1;
    };

    /** @brief Quote one literal shell argument without evaluating its contents. */
    inline std::string moeReferenceShellQuote(const std::string &value)
    {
        std::string result = "'";
        for (char c : value)
            result += c == '\'' ? "'\\''" : std::string(1, c);
        return result + "'";
    }

    /**
     * @brief Load/generate authenticated unweighted [expert,row,hidden] equations.
     * @param identity Exact real-model/checkpoint identity, never a native tensor.
     * @param expert_ids Sorted unique union of native and canonical selected IDs.
     * @param rows HF input row count.
     * @param hidden HF input/output width.
     * @return Immutable FP32 bank and independently authenticated model RMS parameters.
     * @throws std::runtime_error For unsupported, corrupt, or incomplete evidence.
     *
     * A cache hit imports only NumPy and checks small-input identity. A miss
     * reads only the requested expert slices, then proves their canonical sum
     * reproduces the existing HF checkpoint before committing the cache.
     */
    inline MoEIndependentReference loadMoERouteConditionedReference(
        const MoERouteReferenceIdentity &identity,
        const std::vector<int> &expert_ids, size_t rows, size_t hidden)
    {
        if (identity.gguf_block < 0 || expert_ids.empty() || !rows || !hidden ||
            identity.checkpoint_prefix.empty() ||
            identity.checkpoint_prefix.find_first_not_of(
                "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_") != std::string::npos)
            throw std::runtime_error("invalid route-conditioned HF identity");
        std::ostringstream ids;
        std::string basename;
        for (size_t index = 0; index < expert_ids.size(); ++index)
        {
            if (expert_ids[index] < 0 || (index && expert_ids[index - 1] >= expert_ids[index]))
                throw std::runtime_error("route-conditioned HF IDs must be sorted and unique");
            if (index) ids << ',';
            ids << expert_ids[index];
            basename += std::to_string(expert_ids[index]) + "_";
        }
        const auto output = identity.snapshot_directory / "route_conditioned" /
            identity.checkpoint_prefix / (basename + ".npy");
        ReferenceGenerationLease lease;
        std::ostringstream script;
        script << "unset OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS "
                  "OMP_PROC_BIND OMP_PLACES KMP_AFFINITY; "
                  "if [ -f /workspaces/llaminar/.venv/bin/activate ]; then "
                  "source /workspaces/llaminar/.venv/bin/activate; fi; "
                  "python3 python/reference/generate_moe_route_conditioned_reference.py"
               << " --model " << moeReferenceShellQuote(identity.model.string())
               << " --reference-dir " << moeReferenceShellQuote(identity.snapshot_directory.string())
               << " --reference-prefix " << moeReferenceShellQuote(identity.checkpoint_prefix)
               << " --tensor-prefix blk." << identity.gguf_block
               << " --expert-ids " << moeReferenceShellQuote(ids.str())
               << " --output " << moeReferenceShellQuote(output.string());
        const std::string command = "bash -c " + moeReferenceShellQuote(script.str()) + " 2>&1";
        FILE *pipe = popen(command.c_str(), "r");
        if (!pipe)
            throw std::runtime_error("cannot launch independent route-conditioned HF generator");
        std::string diagnostics;
        std::array<char, 512> buffer{};
        while (fgets(buffer.data(), buffer.size(), pipe))
            diagnostics += buffer.data();
        const int status = pclose(pipe);
        std::filesystem::create_directories(identity.artifacts);
        std::ofstream log(identity.artifacts / "route_conditioned_hf_generation.log", std::ios::app);
        log << diagnostics;
        if (status != 0)
            throw std::runtime_error("independent route-conditioned HF generation failed: " + diagnostics);
        const auto array = cnpy::npy_load(output.string());
        if (array.word_size != sizeof(float) || array.fortran_order ||
            array.shape != std::vector<size_t>{expert_ids.size(), rows, hidden})
            throw std::runtime_error("route-conditioned HF bank has invalid geometry/dtype");
        const float *data = array.data<float>();
        auto norm_path = output;
        norm_path.replace_extension(".norm.npy");
        const auto norm = cnpy::npy_load(norm_path.string());
        if (norm.word_size != sizeof(float) || norm.fortran_order ||
            norm.shape != std::vector<size_t>{hidden + 1})
            throw std::runtime_error("route-conditioned HF normalization has invalid geometry/dtype");
        const float *parameters = norm.data<float>();
        return {.experts = {data, data + array.num_vals},
            .normalization = {.scale = {parameters, parameters + hidden}, .epsilon = parameters[hidden]}};
    }
}
