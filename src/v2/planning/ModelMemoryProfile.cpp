/**
 * @file ModelMemoryProfile.cpp
 * @brief Builds and serializes compact model weight inventories for placement planning.
 *
 * The memory profile keeps the planning layer independent of loaded Tensor objects.
 * It records GGUF tensor names, element counts, native byte sizes, quantization
 * type strings, and inferred layer ownership so higher-level placement code can
 * estimate per-device weight pressure before constructing the execution graph.
 * Wire decoding bounds all extents before allocating and consumes the complete
 * payload, so a stale or malformed directory cannot become an admitted model.
 */
#include "planning/ModelMemoryProfile.h"
#include "kernels/common/SparseQueryExecutionProgram.h"
#include "../models/qwen4exp/Qwen4ExpConfig.h"
#include "../models/qwen4exp/Qwen4ExpSidecarManifest.h"
#include "../models/qwen4exp/Qwen4ExpPublishedGeometry.h"
#include <array>
#include <algorithm>

#include "loaders/ModelLoader.h"
#include "execution/mtp/MTPLearnedBlockCount.h"
#include <regex>
#include <numeric>
#include <cstring>
#include <stdexcept>
#include <string_view>
#include <limits>

namespace llaminar2
{
    std::size_t ModelMemoryProfile::sparseQueryProgramCount() const
    {
        if (!flash_next) return 1u;
        const int main_layers = mainLayerCount();
        return std::any_of(flash_next->sparse_decoder_layers.begin(),
            flash_next->sparse_decoder_layers.end(), [&](auto layer) {
                return layer < static_cast<std::uint32_t>(main_layers);
            }) ? kSparseQueryExecutionProgramCount : 1u;
    }

    int ModelMemoryProfile::mainLayerCount() const
    {
        if (n_layers < 0 || mtp_layer_count < 0 || (!flash_next && mtp_layer_count > n_layers))
            throw std::invalid_argument("Model layer counts are inconsistent");
        // Flash Next publishes main block_count separately from nextn; ordinary
        // Qwen exports include trailing predictors in their block_count.
        return flash_next ? n_layers : n_layers - mtp_layer_count;
    }

    int ModelMemoryProfile::totalLayerCount() const
    {
        const int main = mainLayerCount();
        if (mtp_layer_count > std::numeric_limits<int>::max() - main)
            throw std::overflow_error("Model total layer count exceeds integer geometry");
        return main + mtp_layer_count;
    }

    namespace
    {

        /** @brief Stable marker preceding every serialized memory profile. */
        constexpr uint32_t kProfileWireMagic = 0x4C4D5032U; // "LMP2"
        /** @brief Current profile layout, including exact MoE geometry. */
        constexpr uint32_t kProfileWireVersion = 6U;

        /** Exact stable order of fixed-width Flash Next geometry scalars. */
        constexpr std::array<uint32_t Qwen4ExpMemoryGeometry::*, 18> kFlashNextFields{
            &Qwen4ExpMemoryGeometry::residual_width,
            &Qwen4ExpMemoryGeometry::terminal_head_width,
            &Qwen4ExpMemoryGeometry::residual_branches,
            &Qwen4ExpMemoryGeometry::residual_gate_rank,
            &Qwen4ExpMemoryGeometry::indexer_query_heads,
            &Qwen4ExpMemoryGeometry::indexer_key_heads,
            &Qwen4ExpMemoryGeometry::indexer_head_dimension,
            &Qwen4ExpMemoryGeometry::compression_ratio,
            &Qwen4ExpMemoryGeometry::selected_token_budget,
            &Qwen4ExpMemoryGeometry::maximum_selected_tokens,
            &Qwen4ExpMemoryGeometry::ple_decoder_layer,
            &Qwen4ExpMemoryGeometry::ple_heads,
            &Qwen4ExpMemoryGeometry::ple_row_elements,
            &Qwen4ExpMemoryGeometry::ple_convolution_kernel,
            &Qwen4ExpMemoryGeometry::ple_convolution_dilation,
            &Qwen4ExpMemoryGeometry::ple_convolution_history,
            &Qwen4ExpMemoryGeometry::ple_hash_history_tokens,
            &Qwen4ExpMemoryGeometry::ple_table_gguf_type
        };

        /** @brief Validate relationships before a profile can influence admission. */
        void validateFlashNextGeometry(const ModelMemoryProfile &profile)
        {
            if ((profile.architecture == "qwen4exp") != profile.flash_next.has_value())
                throw std::runtime_error("ModelMemoryProfile qwen4exp geometry identity is missing or foreign");
            if (!profile.flash_next)
            {
                if (profile.flash_next_predictor) throw std::runtime_error("Foreign qwen4exp predictor metadata");
                return;
            }
            if (profile.flash_next_predictor.has_value() != (profile.mtp_layer_count > 0))
                throw std::runtime_error("Qwen4Exp predictor policy is missing or unexpected");
            if (profile.flash_next_predictor)
            {
                const auto &predictor = *profile.flash_next_predictor;
                if (!predictor.validForMainLayers(static_cast<uint32_t>(profile.n_layers)))
                    throw std::runtime_error("Qwen4Exp predictor policy differs from the authenticated source convention");
            }
            const auto &g = *profile.flash_next;
            namespace published = qwen4exp::published;
            if (profile.d_model != published::ComputeWidth || profile.n_layers != published::MainLayers ||
                profile.mtp_layer_count < 0 || profile.mtp_layer_count > 1 ||
                profile.n_heads != published::QueryHeads || profile.n_kv_heads != published::KVHeads ||
                profile.head_dim != published::HeadDimension ||
                g.residual_branches != published::Branches || g.residual_gate_rank != published::GateRank ||
                g.indexer_query_heads != published::IndexerQueryHeads ||
                g.indexer_key_heads != published::IndexerKeyHeads ||
                g.indexer_head_dimension != published::IndexerDimension ||
                g.compression_ratio != published::CompressionRatio ||
                g.selected_token_budget != published::SelectedTokenBudget ||
                g.ple_decoder_layer != published::PLELayer ||
                g.ple_heads != (published::NgramOrder - 1) * published::HeadsPerOrder ||
                g.ple_row_elements != published::RowElements ||
                g.ple_convolution_kernel != published::PLEKernel ||
                g.ple_convolution_dilation != published::NgramOrder ||
                g.ple_hash_history_tokens != published::NgramOrder - 1)
                throw std::runtime_error("ModelMemoryProfile qwen4exp geometry differs from published source contract");
            std::vector<uint32_t> expected_sparse;
            for (uint32_t layer = 0; layer < published::MainLayers + profile.mtp_layer_count; ++layer)
                if ((layer < published::MainLayers && (layer + 1) % published::AttentionInterval == 0) ||
                    (layer >= published::MainLayers && profile.flash_next_predictor &&
                     profile.flash_next_predictor->attention == Qwen4ExpPredictorAttention::Sparse))
                    expected_sparse.push_back(layer);
            if (g.sparse_decoder_layers != expected_sparse)
                throw std::runtime_error("ModelMemoryProfile qwen4exp sparse layer set differs from source contract");
            if (profile.d_model <= 0 || profile.n_layers <= 0 || profile.mtp_layer_count < 0 ||
                g.residual_branches < 2 ||
                uint64_t(g.residual_width) != uint64_t(profile.d_model) * g.residual_branches ||
                g.terminal_head_width != uint32_t(profile.d_model) || g.residual_gate_rank == 0 ||
                g.indexer_query_heads == 0 || g.indexer_key_heads == 0 || g.indexer_head_dimension == 0 ||
                g.compression_ratio < 2 || g.selected_token_budget == 0 ||
                g.selected_token_budget % g.compression_ratio != 0 ||
                uint64_t(g.maximum_selected_tokens) != uint64_t(g.selected_token_budget) + g.compression_ratio - 1 ||
                g.ple_heads == 0 || g.ple_row_elements == 0 ||
                uint64_t(g.ple_heads) * g.ple_row_elements != uint32_t(profile.d_model) ||
                g.ple_convolution_kernel < 2 || g.ple_convolution_dilation == 0 ||
                uint64_t(g.ple_convolution_history) != uint64_t(g.ple_convolution_kernel - 1) * g.ple_convolution_dilation ||
                g.ple_hash_history_tokens == 0 || g.ple_table_rows == 0 || g.ple_table_row_bytes == 0 ||
                g.ple_table_rows > UINT64_MAX / g.ple_table_row_bytes ||
                g.ple_table_native_bytes != g.ple_table_rows * g.ple_table_row_bytes ||
                g.ple_decoder_layer >= uint32_t(profile.n_layers) || g.sparse_decoder_layers.empty())
                throw std::runtime_error("ModelMemoryProfile qwen4exp geometry relationships are invalid");
            GGUFTensorInfo table{.type = static_cast<GGUFTensorType>(g.ple_table_gguf_type)};
            (void)ggufToTensorType(table.type);
            const auto block = table.isQuantized() ? table.getBlockSize() : 1u;
            if (block == 0 || g.ple_row_elements % block != 0 ||
                g.ple_table_row_bytes != g.ple_row_elements / block * table.getTypeSize())
                throw std::runtime_error("ModelMemoryProfile qwen4exp native row format is invalid");
            // Whole-tier metadata fixtures may omit both arrays. Actual source
            // projection always supplies them; explicit sharding requires16.
            if(!g.ple_head_offsets.empty() || !g.ple_head_vocab_sizes.empty())
            {
                if(g.ple_head_offsets.size()!=g.ple_heads || g.ple_head_vocab_sizes.size()!=g.ple_heads)
                    throw std::runtime_error("QFN original head metadata is incomplete");
                std::uint64_t end=0;
                for(std::size_t h=0;h<g.ple_heads;++h)
                {
                    const auto begin=g.ple_head_offsets[h],size=g.ple_head_vocab_sizes[h];
                    if(!size || begin<end || begin>g.ple_table_rows || size>g.ple_table_rows-begin)
                        throw std::runtime_error("QFN original head source interval is invalid");
                    end=begin+size;
                }
            }
            uint32_t previous = 0;
            for (size_t i = 0; i < g.sparse_decoder_layers.size(); ++i)
            {
                const auto layer = g.sparse_decoder_layers[i];
                if (layer >= uint32_t(profile.n_layers + profile.mtp_layer_count) ||
                    (i != 0 && layer <= previous))
                    throw std::runtime_error("ModelMemoryProfile qwen4exp sparse layer identity is invalid");
                previous = layer;
            }
        }


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

        /**
         * @brief Add one native extent without allowing metadata admission to wrap.
         * @param total Bytes already owned by this inventory or selected layer set.
         * @param bytes Exact next original source extent.
         * @return Representable sum; an unrepresentable BOM is rejected.
         */
        size_t addNativeBytes(size_t total, size_t bytes)
        {
            if (bytes > std::numeric_limits<size_t>::max() - total)
                throw std::overflow_error("Model native source byte total overflows");
            return total + bytes;
        }

        /** @brief Append one original native source extent to the consumed weight inventory. */
        void appendNativeTensor(ModelMemoryProfile &profile, const GGUFTensorInfo &tensor)
        {
            TensorSizeInfo info;
            info.name = tensor.name;
            if (tensor.size_bytes > std::numeric_limits<size_t>::max())
                throw std::overflow_error("Model native source extent exceeds host byte geometry");
            info.native_bytes = static_cast<size_t>(tensor.size_bytes);
            info.quant_type = ggufTypeToString(tensor.type);
            info.elements = computeElements(tensor.dimensions);
            info.K = logicalMatrixK(tensor, profile.expert_count);
            info.layer_index = parseLayerIndex(tensor.name);
            profile.total_native_bytes = addNativeBytes(profile.total_native_bytes, info.native_bytes);
            profile.tensors.push_back(std::move(info));
        }

    } // anonymous namespace

    Qwen4ExpMemoryGeometry Qwen4ExpMemoryGeometry::fromConfig(const Qwen4ExpConfig &config)
    {
        const auto &qsa = config.sparseAttention();
        const auto &ple = config.ngrams();
        Qwen4ExpMemoryGeometry geometry{
            .residual_width = config.activations().trunk,
            .terminal_head_width = config.activations().terminal_head,
            .residual_branches = config.hyperConnections().branches,
            .residual_gate_rank = config.hyperConnections().gate_rank,
            .indexer_query_heads = qsa.indexer_query_heads,
            .indexer_key_heads = qsa.indexer_key_heads,
            .indexer_head_dimension = qsa.indexer_dimension,
            .compression_ratio = qsa.compression_ratio,
            .selected_token_budget = qsa.selected_token_budget,
            .maximum_selected_tokens = qsa.maximum_selected_tokens,
            .ple_decoder_layer = ple.decoder_layer,
            .ple_heads = ple.heads,
            .ple_row_elements = ple.row_elements,
            .ple_convolution_kernel = ple.convolution_kernel,
            .ple_convolution_dilation = ple.convolution_dilation,
            .ple_convolution_history = ple.convolution_history,
            .ple_hash_history_tokens = ple.maximum_order - 1,
            .ple_table_gguf_type = static_cast<uint32_t>(ple.table.type),
            .ple_table_rows = ple.table.dimensions[0],
            .ple_table_row_bytes = ple.row_bytes,
            .ple_table_native_bytes = ple.table.size_bytes,
            .ple_head_offsets = ple.head_offsets,
            .ple_head_vocab_sizes = ple.head_vocab_sizes};
        for (uint32_t layer = 0; layer < config.mixers().size(); ++layer)
            if (config.mixers()[layer] == Qwen4ExpMixer::SparseAttention)
                geometry.sparse_decoder_layers.push_back(layer);
        return geometry;
    }

    Qwen4ExpLayerKind ModelMemoryProfile::flashNextLayerKind(int layer) const
    {
        if (!flash_next || architecture != "qwen4exp")
            throw std::invalid_argument("Qwen4Exp layer kind requires authenticated geometry");
        if (layer < 0 || layer >= totalLayerCount())
            throw std::invalid_argument("Qwen4Exp layer kind exceeds the original source interval");
        if (layer >= mainLayerCount())
        {
            if (!flash_next_predictor || !flash_next_predictor->validForMainLayers(n_layers))
                throw std::invalid_argument("Qwen4Exp predictor lacks an exact attention policy");
            return flash_next_predictor->attention == Qwen4ExpPredictorAttention::Dense
                ? Qwen4ExpLayerKind::Dense : Qwen4ExpLayerKind::QSA;
        }
        return std::find(flash_next->sparse_decoder_layers.begin(), flash_next->sparse_decoder_layers.end(),
            static_cast<uint32_t>(layer)) != flash_next->sparse_decoder_layers.end()
                ? Qwen4ExpLayerKind::QSA : Qwen4ExpLayerKind::GDN;
    }

    size_t ModelMemoryProfile::flashNextSparseLayerCount(int first_layer, int last_layer) const
    {
        if (!flash_next) throw std::invalid_argument("Sparse layer query requires qwen4exp geometry");
        if (first_layer < 0 || last_layer < first_layer || last_layer >= totalLayerCount())
            throw std::invalid_argument("Qwen4Exp sparse layer query exceeds the declared source interval");
        return std::count_if(flash_next->sparse_decoder_layers.begin(), flash_next->sparse_decoder_layers.end(),
            [&](uint32_t layer) { return layer >= static_cast<uint32_t>(first_layer) && layer <= static_cast<uint32_t>(last_layer); });
    }

    ModelMemoryProfile ModelMemoryProfile::fromGGUF(const GGUFModel &model)
    {
        ModelMemoryProfile profile;

        profile.architecture = model.architecture;
        if (model.architecture == "qwen4exp")
        {
            const Qwen4ExpConfig config(model);
            profile.n_layers = config.mainLayers();
            profile.d_model = config.activations().compute;
            profile.d_ff = config.experts().shared_intermediate;
            profile.n_heads = config.sparseAttention().query_heads;
            profile.n_kv_heads = config.sparseAttention().kv_heads;
            profile.head_dim = config.sparseAttention().head_dimension;
            profile.vocab_size = config.vocabularySize();
            profile.max_seq_len = config.contextLength();
            profile.expert_count = config.experts().count;
            profile.expert_used_count = config.experts().selected;
            profile.expert_feed_forward_length = config.experts().intermediate;
            profile.expert_shared_feed_forward_length = config.experts().shared_intermediate;
            profile.mtp_layer_count = config.predictorLayers();
            profile.full_attention_interval = static_cast<int>(
                model.metadata.at("qwen4exp.full_attention_interval").asUInt64Checked());
            profile.gdn_conv_kernel_size = config.deltaNet().convolution_kernel;
            profile.gdn_state_size = config.deltaNet().head_dimension;
            profile.gdn_inner_size = config.deltaNet().value_width;
            profile.gdn_group_count = config.deltaNet().key_heads;
            profile.gdn_time_step_rank = config.deltaNet().value_heads;
            profile.flash_next = Qwen4ExpMemoryGeometry::fromConfig(config);
            if (config.predictorLayers())
                profile.flash_next_predictor = Qwen4ExpPredictorMetadata{config.mainLayers(),
                    Qwen4ExpPredictorAttention::Sparse, Qwen4ExpPredictorHiddenNorm::PerBranch,
                    Qwen4ExpPredictorSource::Inline};
        }
        else
        {
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
            // Count real directory-inferred predictors exactly as graph discovery
            // does. Optional exporter metadata must not erase their shifted KV BOM.
            profile.mtp_layer_count = mtpLearnedBlockCount(model);
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

        }

        // Build tensor inventory
        profile.total_native_bytes = 0;
        profile.tensors.reserve(model.tensors.size());

        for (const auto &tensor : model.tensors) appendNativeTensor(profile, tensor);

        return profile;
    }

    ModelMemoryProfile ModelMemoryProfile::fromGGUF(const GGUFModel &main, const GGUFModel &predictor,
        SharedPredictorMaterialization materialization)
    {
        if (materialization != SharedPredictorMaterialization::Active && materialization != SharedPredictorMaterialization::Inactive)
            throw std::invalid_argument("Unknown shared predictor materialization policy");
        const Qwen4ExpSidecarManifest manifest(main, predictor);
        auto profile = fromGGUF(main);
        profile.mtp_layer_count = 1;
        profile.flash_next_predictor = manifest.predictorMetadata();
        if (manifest.attention() == Qwen4ExpPredictorAttention::Sparse)
            profile.flash_next->sparse_decoder_layers.push_back(manifest.sourceLayer());
        if (materialization == SharedPredictorMaterialization::Active)
            for (const auto &source : manifest.sources())
                if (source.source == Qwen4ExpTensorSource::Predictor && source.consumed)
                    appendNativeTensor(profile, source.tensor);
        validateFlashNextGeometry(profile);
        return profile;
    }

    ModelMemoryProfile ModelMemoryProfile::fromLoader(const ModelLoader &loader)
    {
        if (const auto *predictor = loader.ggufPredictorDirectory())
        {
            if (!loader.predictorManifest())
                throw std::invalid_argument("External predictor requires its authenticated source manifest");
            return fromGGUF(loader.getModel(), *predictor, loader.predictorMaterialization());
        }
        if (loader.predictorManifest())
            throw std::invalid_argument("Predictor manifest has no original directory");
        return fromGGUF(loader.getModel());
    }

    /**
     * @brief Sum only native extents in the inclusive admitted source interval.
     * @param first_layer First selected source layer, including -1 for globals.
     * @param last_layer Last selected source layer.
     * @return Exact selected bytes.
     * @throws std::overflow_error If selected extents cannot fit the planning byte type.
     */
    size_t ModelMemoryProfile::weightBytesForLayers(int first_layer, int last_layer) const
    {
        size_t total = 0;
        for (const auto &t : tensors)
        {
            if (t.layer_index >= first_layer && t.layer_index <= last_layer)
            {
                total = addNativeBytes(total, t.native_bytes);
            }
        }
        return total;
    }

    size_t ModelMemoryProfile::embeddingBytes() const
    {
        for (const auto &t : tensors)
        {
            if (t.name == "token_embd.weight" || t.name == "embed_tokens.weight" ||
                t.name == "model.embed_tokens.weight")
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

    /** @brief Return exact global norm bytes, rejecting an unrepresentable source sum. */
    size_t ModelMemoryProfile::normBytes() const
    {
        size_t total = 0;
        for (const auto &t : tensors)
        {
            if (t.layer_index == -1 &&
                (t.name.find("norm") != std::string::npos ||
                 t.name.find("ln_") != std::string::npos))
            {
                total = addNativeBytes(total, t.native_bytes);
            }
        }
        return total;
    }

    /**
     * @brief Return exact native extents owned by one source layer.
     * @param layer Selected source layer, including -1 for global tensors.
     * @return Exact selected bytes.
     * @throws std::overflow_error If the source byte total is unrepresentable.
     */
    size_t ModelMemoryProfile::layerWeightBytes(int layer) const
    {
        size_t total = 0;
        for (const auto &t : tensors)
        {
            if (t.layer_index == layer)
            {
                total = addNativeBytes(total, t.native_bytes);
            }
        }
        return total;
    }

    // ================================================================
    // MPI serialization helpers (local to this TU)
    // ================================================================

    namespace
    {

        /** @brief Append one native profile scalar without introducing alignment padding. */
        template <typename T>
        void writeVal(std::vector<uint8_t> &buf, T value)
        {
            const auto *p = reinterpret_cast<const uint8_t *>(&value);
            buf.insert(buf.end(), p, p + sizeof(T));
        }

        /** @brief Append a length-prefixed string, rejecting unrepresentable lengths. */
        void writeStr(std::vector<uint8_t> &buf, const std::string &s)
        {
            if (s.size() > std::numeric_limits<uint32_t>::max())
                throw std::length_error("ModelMemoryProfile string exceeds wire capacity");
            writeVal<uint32_t>(buf, static_cast<uint32_t>(s.size()));
            buf.insert(buf.end(), s.begin(), s.end());
        }

        /** @brief Read a scalar without forming a pointer outside the supplied extent. */
        template <typename T>
        T readVal(const uint8_t *&ptr, const uint8_t *end)
        {
            if (static_cast<size_t>(end - ptr) < sizeof(T))
                throw std::runtime_error("ModelMemoryProfile deserialization: buffer underflow");
            T v;
            std::memcpy(&v, ptr, sizeof(T));
            ptr += sizeof(T);
            return v;
        }

        /** @brief Validate the complete string extent before constructing its storage. */
        std::string readStr(const uint8_t *&ptr, const uint8_t *end)
        {
            uint32_t len = readVal<uint32_t>(ptr, end);
            if (static_cast<size_t>(end - ptr) < len)
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
        validateFlashNextGeometry(*this);
        if (tensors.size() > std::numeric_limits<uint32_t>::max())
            throw std::length_error("ModelMemoryProfile tensor inventory exceeds wire capacity");
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
        writeVal<uint8_t>(buf, flash_next.has_value() ? 1u : 0u);
        if (flash_next)
        {
            const auto &g = *flash_next;
            for (const auto field : kFlashNextFields) writeVal<uint32_t>(buf, g.*field);
            writeVal<uint64_t>(buf, g.ple_table_rows);
            writeVal<uint64_t>(buf, g.ple_table_row_bytes);
            writeVal<uint64_t>(buf, g.ple_table_native_bytes);
            for(const auto *values:{&g.ple_head_offsets,&g.ple_head_vocab_sizes})
            {
                writeVal<uint32_t>(buf,static_cast<std::uint32_t>(values->size()));
                for(const auto value:*values)writeVal<uint64_t>(buf,value);
            }
            writeVal<uint32_t>(buf, static_cast<uint32_t>(g.sparse_decoder_layers.size()));
            for (const auto layer : g.sparse_decoder_layers) writeVal<uint32_t>(buf, layer);
        }

        writeVal<uint8_t>(buf, flash_next_predictor.has_value() ? 1u : 0u);
        if (flash_next_predictor)
        {
            const auto &predictor = *flash_next_predictor;
            writeVal<uint32_t>(buf, predictor.source_layer);
            writeVal<uint8_t>(buf, static_cast<uint8_t>(predictor.attention));
            writeVal<uint8_t>(buf, static_cast<uint8_t>(predictor.hidden_norm));
            writeVal<uint8_t>(buf, static_cast<uint8_t>(predictor.source));
            writeVal<uint8_t>(buf, static_cast<uint8_t>(predictor.cacheOrigin()));
        }

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
        if (!data || size < 2 * sizeof(uint32_t))
            throw std::runtime_error("ModelMemoryProfile deserialization: missing header");
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
        const auto has_flash_next = readVal<uint8_t>(ptr, end);
        if (has_flash_next > 1u)
            throw std::runtime_error("ModelMemoryProfile deserialization: invalid qwen4exp presence tag");
        if (has_flash_next)
        {
            auto &g = p.flash_next.emplace();
            for (const auto field : kFlashNextFields) g.*field = readVal<uint32_t>(ptr, end);
            g.ple_table_rows = readVal<uint64_t>(ptr, end);
            g.ple_table_row_bytes = readVal<uint64_t>(ptr, end);
            g.ple_table_native_bytes = readVal<uint64_t>(ptr, end);
            for(auto *values:{&g.ple_head_offsets,&g.ple_head_vocab_sizes})
            {
                const auto n=readVal<uint32_t>(ptr,end);
                if(n>16 || n>static_cast<std::size_t>(end-ptr)/sizeof(uint64_t))
                    throw std::runtime_error("QFN original head metadata count exceeds its source/frame");
                for(uint32_t h=0;h<n;++h)values->push_back(readVal<uint64_t>(ptr,end));
            }
            const uint32_t layers = readVal<uint32_t>(ptr, end);
            if (layers > static_cast<size_t>(end - ptr) / sizeof(uint32_t))
                throw std::runtime_error("ModelMemoryProfile deserialization: impossible qwen4exp sparse layer count");
            g.sparse_decoder_layers.reserve(layers);
            for (uint32_t i = 0; i < layers; ++i)
                g.sparse_decoder_layers.push_back(readVal<uint32_t>(ptr, end));
        }
        const auto has_predictor = readVal<uint8_t>(ptr, end);
        if (has_predictor > 1) throw std::runtime_error("Invalid qwen4exp predictor presence tag");
        if (has_predictor) {
            p.flash_next_predictor = Qwen4ExpPredictorMetadata{
                readVal<uint32_t>(ptr, end), static_cast<Qwen4ExpPredictorAttention>(readVal<uint8_t>(ptr, end)),
                static_cast<Qwen4ExpPredictorHiddenNorm>(readVal<uint8_t>(ptr, end)),
                static_cast<Qwen4ExpPredictorSource>(readVal<uint8_t>(ptr, end))};
            if(static_cast<MTPShiftedCacheOrigin>(readVal<uint8_t>(ptr,end))!=p.flash_next_predictor->cacheOrigin())
                throw std::runtime_error("QFN profile must authenticate its mandatory visible token-zero bootstrap");
        }
        validateFlashNextGeometry(p);

        uint32_t n_tensors = readVal<uint32_t>(ptr, end);
        // Even an empty-name tensor needs both string lengths, three 64-bit
        // extents and a layer index. Check this lower bound before reserve:
        // a corrupt count must not trigger a multi-gigabyte allocation first.
        constexpr size_t minimum_tensor_record_bytes =
            2 * sizeof(uint32_t) + 3 * sizeof(uint64_t) + sizeof(int32_t);
        if (n_tensors > static_cast<size_t>(end - ptr) / minimum_tensor_record_bytes)
            throw std::runtime_error("ModelMemoryProfile deserialization: impossible tensor count");
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

        if (ptr != end)
            throw std::runtime_error("ModelMemoryProfile deserialization: trailing payload bytes");
        return p;
    }

} // namespace llaminar2
