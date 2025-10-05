/**
 * @file model_integration_test_base.cpp
 * @brief Implementation of base model integration test framework
 * @author David Sanftenberg
 */

#include "model_integration_test_base.h"
#include "abstract_pipeline.h"
#include "model_loader.h"
#include "qwen_pipeline_adapter.h"
#include "logger.h"
#include <fstream>
#include <cmath>
#include <mpi.h>
#include <filesystem>
#include <cstdlib>
#include <iostream>

namespace llaminar
{
    namespace testing
    {

        using namespace llaminar::parity; // For SnapshotRegistry, loaders, etc

        void ModelIntegrationTestBase::TearDown()
        {
            // Clear registry
            SnapshotRegistry::instance().clear();
            LOG_INFO("ModelIntegrationTest teardown complete");
        }

        bool ModelIntegrationTestBase::run_integration_test(const TestConfig &config)
        {
            LOG_INFO("=== Running Integration Test ===");
            LOG_INFO("Model: " << config.model_name << "-" << config.precision);
            LOG_INFO("Stage: " << config.stage);
            LOG_INFO("Tokens: " << config.tokens.size() << " tokens");

            // 1. Check if golden reference exists
            std::string golden_path = config.get_golden_path();
            if (!golden_reference_exists(golden_path))
            {
                LOG_WARN("Golden reference not found: " << golden_path);
                LOG_WARN("Skipping test - generate with: python3 tests/integration/generate_golden_references.py");
                // Return true to indicate "skip" (not a failure) - tests should check and call GTEST_SKIP themselves
                return true;
            }

            // 2. Load PyTorch golden reference
            LOG_INFO("Loading PyTorch golden reference: " << golden_path);
            size_t pytorch_count = load_golden_reference(golden_path, "pytorch");
            if (pytorch_count == 0)
            {
                LOG_ERROR("Failed to load PyTorch snapshots from: " << golden_path);
                return false;
            }
            LOG_INFO("Loaded " << pytorch_count << " PyTorch snapshots");

            // 3. Run Llaminar pipeline with capture
            std::string model_path = config.get_model_path();
            bool is_prefill = (config.stage == "prefill");

            LOG_INFO("Running Llaminar pipeline: " << model_path);
            if (!run_llaminar_with_capture(model_path, config.tokens, is_prefill))
            {
                LOG_ERROR("Llaminar pipeline execution failed");
                return false;
            }

            // 4. Compare snapshots layer-by-layer
            std::string first_diverging_layer;
            bool all_match = compare_snapshots(config.tolerances, first_diverging_layer);

            if (all_match)
            {
                LOG_INFO("✓ All layers match within tolerance!");
                LOG_INFO("  max_abs=" << config.tolerances.max_abs
                                      << ", rel_l2=" << config.tolerances.rel_l2);
            }
            else
            {
                LOG_ERROR("✗ Layer divergence detected!");
                LOG_ERROR("  First diverging layer: " << first_diverging_layer);
            }

            return all_match;
        }

        size_t ModelIntegrationTestBase::load_golden_reference(const std::string &npz_path,
                                                               const std::string &source_name)
        {
            // Use static loader method
            return PytorchSnapshotLoader::load_from_npz(npz_path, source_name);
        }

        bool ModelIntegrationTestBase::run_llaminar_with_capture(const std::string &model_path,
                                                                 const std::vector<int> &tokens,
                                                                 bool is_prefill)
        {
            try
            {
                int rank;
                MPI_Comm_rank(MPI_COMM_WORLD, &rank);

                if (rank == 0)
                {
                    LOG_INFO("Loading model for pipeline execution: " << model_path);
                }

                // 1. Load model to get layer configuration
                auto model_loader = std::make_unique<ModelLoader>();
                if (!model_loader->loadModel(model_path))
                {
                    LOG_ERROR("Failed to load model from: " << model_path);
                    return false;
                }

                auto layer_config = model_loader->createLayerConfig();
                if (rank == 0)
                {
                    LOG_DEBUG("Model config: " << layer_config.n_layers << " layers, "
                                               << layer_config.n_head << " heads, " << layer_config.d_model << " dimensions");
                }

                // 2. Create ModelConfig with architecture (assuming Qwen for now)
                ModelConfig model_config(layer_config, "qwen");
                model_config.has_gqa = (layer_config.n_head_kv < layer_config.n_head);

                // 3. Register architecture and create pipeline
                registerQwenPipeline();
                auto pipeline = PipelineFactory::instance().create(model_config);
                if (!pipeline)
                {
                    LOG_ERROR("Failed to create pipeline for architecture: qwen");
                    return false;
                }

                if (rank == 0)
                {
                    LOG_DEBUG("Pipeline created: " << pipeline->name());
                }

                // 4. Load weights through pipeline interface
                auto loaded_weights = pipeline->loadWeights(model_path);
                if (!loaded_weights)
                {
                    LOG_ERROR("Failed to load model weights");
                    return false;
                }

                if (rank == 0)
                {
                    LOG_DEBUG("Model weights loaded successfully");
                }

                // 5. Enable snapshot capture
                setenv("LLAMINAR_PARITY_CAPTURE", "1", 1);
                parity::LlaminarSnapshotHook::set_enabled(true);

                // 6. Create StageContext for pipeline execution
                StageContext ctx;

                // 7. Execute appropriate stage
                bool success = false;
                if (is_prefill)
                {
                    if (rank == 0)
                    {
                        LOG_INFO("Running prefill with " << tokens.size() << " tokens");
                    }
                    success = pipeline->prefill(tokens, *loaded_weights, ctx);
                    if (!success)
                    {
                        LOG_ERROR("Prefill execution failed");
                        return false;
                    }
                }
                else
                {
                    // For decode, we need to run prefill first to establish context,
                    // then decode the final token
                    if (tokens.empty())
                    {
                        LOG_ERROR("Cannot run decode with empty token list");
                        return false;
                    }

                    if (rank == 0)
                    {
                        LOG_INFO("Running decode: prefill " << (tokens.size() - 1)
                                                            << " tokens, then decode last token");
                    }

                    // Prefill all but last token
                    std::vector<int> prefill_tokens(tokens.begin(), tokens.end() - 1);
                    if (!prefill_tokens.empty())
                    {
                        success = pipeline->prefill(prefill_tokens, *loaded_weights, ctx);
                        if (!success)
                        {
                            LOG_ERROR("Prefill phase before decode failed");
                            return false;
                        }
                    }

                    // Decode the last token
                    int decode_token = tokens.back();
                    success = pipeline->decode(decode_token, *loaded_weights, ctx);
                    if (!success)
                    {
                        LOG_ERROR("Decode execution failed");
                        return false;
                    }
                }

                // 8. Disable capture
                parity::LlaminarSnapshotHook::set_enabled(false);
                unsetenv("LLAMINAR_PARITY_CAPTURE");

                // 9. Verify snapshots were captured
                auto &registry = SnapshotRegistry::instance();
                auto all_keys = registry.list_keys();

                size_t llaminar_count = 0;
                for (const auto &key : all_keys)
                {
                    if (key.find("llaminar") != std::string::npos)
                    {
                        llaminar_count++;
                    }
                }

                if (rank == 0)
                {
                    LOG_INFO("Captured " << llaminar_count << " Llaminar snapshots");
                }

                return llaminar_count > 0;
            }
            catch (const std::exception &e)
            {
                LOG_ERROR("Exception in run_llaminar_with_capture: " << e.what());
                return false;
            }
        }

        bool ModelIntegrationTestBase::compare_snapshots(const ToleranceConfig &tolerances,
                                                         std::string &first_diverging_layer)
        {
            LOG_INFO("Comparing snapshots layer-by-layer...");

            // Get all snapshot keys and filter for PyTorch ones
            auto &registry = SnapshotRegistry::instance();
            auto all_keys = registry.list_keys();

            std::vector<std::string> pytorch_keys;
            for (const auto &key : all_keys)
            {
                if (key.find("pytorch") != std::string::npos)
                {
                    pytorch_keys.push_back(key);
                }
            }

            if (pytorch_keys.empty())
            {
                LOG_ERROR("No PyTorch snapshots to compare");
                return false;
            }

            LOG_INFO("Comparing " << pytorch_keys.size() << " layers");

            bool all_match = true;
            first_diverging_layer.clear();

            for (const auto &pytorch_key : pytorch_keys)
            {
                // Get PyTorch snapshot
                TensorSnapshot pytorch_snapshot;
                if (!registry.get_snapshot(pytorch_key, pytorch_snapshot))
                {
                    LOG_WARN("PyTorch snapshot not found: " << pytorch_key);
                    continue;
                }

                // Find corresponding Llaminar snapshot
                // PyTorch key format: "pytorch:layer_N_stage"
                // Llaminar key format: "llaminar:stage:layer_N"
                // Need to convert between formats

                std::string pytorch_stage_name;
                int layer_index;
                if (!parse_pytorch_key(pytorch_key, layer_index, pytorch_stage_name))
                {
                    LOG_WARN("Failed to parse PyTorch key: " << pytorch_key);
                    continue;
                }

                // Map PyTorch stage name to Llaminar stage name
                std::string llaminar_stage_name = map_pytorch_stage_to_llaminar(pytorch_stage_name);

                // Construct Llaminar key using same format as make_key()
                // Format: "llaminar_layer_{N}_{stage}" or "llaminar_{stage}"
                std::string llaminar_key;
                if (layer_index < 0)
                {
                    llaminar_key = "llaminar_" + llaminar_stage_name;
                }
                else
                {
                    llaminar_key = "llaminar_layer_" + std::to_string(layer_index) + "_" + llaminar_stage_name;
                }

                TensorSnapshot llaminar_snapshot;
                if (!registry.get_snapshot(llaminar_key, llaminar_snapshot))
                {
                    LOG_WARN("Llaminar snapshot not found for: " << llaminar_key);
                    LOG_WARN("  (corresponding to PyTorch: " << pytorch_key << ")");
                    if (first_diverging_layer.empty())
                    {
                        first_diverging_layer = pytorch_key + " (missing in Llaminar)";
                        all_match = false;
                    }
                    continue;
                }

                // Compare snapshots - manual comparison since we don't have compare_with method
                // Calculate max_abs and rel_l2
                size_t size = std::min(pytorch_snapshot.data.size(), llaminar_snapshot.data.size());
                float max_abs = 0.0f;
                float sum_sq_diff = 0.0f;
                float sum_sq_pytorch = 0.0f;

                for (size_t i = 0; i < size; ++i)
                {
                    float diff = std::abs(pytorch_snapshot.data[i] - llaminar_snapshot.data[i]);
                    max_abs = std::max(max_abs, diff);
                    sum_sq_diff += diff * diff;
                    sum_sq_pytorch += pytorch_snapshot.data[i] * pytorch_snapshot.data[i];
                }

                double rel_l2 = (sum_sq_pytorch > 0) ? std::sqrt(sum_sq_diff / sum_sq_pytorch) : 0.0;

                bool match = (max_abs <= tolerances.max_abs) && (rel_l2 <= tolerances.rel_l2);

                if (match)
                {
                    LOG_DEBUG("✓ " << pytorch_key << " matches (max_abs=" << max_abs
                                   << ", rel_l2=" << rel_l2 << ")");
                }
                else
                {
                    LOG_ERROR("✗ " << pytorch_key << " diverges!");
                    LOG_ERROR("  max_abs=" << max_abs << " (threshold=" << tolerances.max_abs << ")");
                    LOG_ERROR("  rel_l2=" << rel_l2 << " (threshold=" << tolerances.rel_l2 << ")");

                    print_divergence_report(pytorch_key, pytorch_snapshot,
                                            llaminar_snapshot, tolerances);

                    if (first_diverging_layer.empty())
                    {
                        first_diverging_layer = pytorch_key;
                        all_match = false;
                    }
                }
            }

            return all_match;
        }

        bool ModelIntegrationTestBase::golden_reference_exists(const std::string &path) const
        {
            std::filesystem::path fs_path(path);
            return std::filesystem::exists(fs_path) && std::filesystem::is_regular_file(fs_path);
        }

        void ModelIntegrationTestBase::print_divergence_report(const std::string &diverging_layer,
                                                               const TensorSnapshot &pytorch_snapshot,
                                                               const TensorSnapshot &llaminar_snapshot,
                                                               const ToleranceConfig &tolerances)
        {
            LOG_ERROR("=== DIVERGENCE REPORT ===");
            LOG_ERROR("Layer: " << diverging_layer);
            LOG_ERROR("");
            LOG_ERROR("PyTorch snapshot:");
            LOG_ERROR("  Shape: " << pytorch_snapshot.metadata.stage_name);
            LOG_ERROR("  Size: " << pytorch_snapshot.data.size() << " elements");
            LOG_ERROR("");
            LOG_ERROR("Llaminar snapshot:");
            LOG_ERROR("  Shape: " << llaminar_snapshot.metadata.stage_name);
            LOG_ERROR("  Size: " << llaminar_snapshot.data.size() << " elements");
            LOG_ERROR("");

            // Sample divergent values
            const float *pytorch_data = pytorch_snapshot.data.data();
            const float *llaminar_data = llaminar_snapshot.data.data();
            size_t size = std::min(pytorch_snapshot.data.size(), llaminar_snapshot.data.size());

            LOG_ERROR("Sample divergent values (first 10):");
            int samples = 0;
            for (size_t i = 0; i < size && samples < 10; ++i)
            {
                float diff = std::abs(pytorch_data[i] - llaminar_data[i]);
                if (diff > tolerances.max_abs)
                {
                    LOG_ERROR("  [" << i << "] PyTorch=" << pytorch_data[i]
                                    << ", Llaminar=" << llaminar_data[i]
                                    << ", diff=" << diff);
                    samples++;
                }
            }
            LOG_ERROR("=== END DIVERGENCE REPORT ===");
        }

        // Map PyTorch stage names to Llaminar PipelineStage enum names
        std::string map_pytorch_stage_to_llaminar(const std::string &pytorch_stage)
        {
            // Direct mappings
            if (pytorch_stage == "embeddings")
                return "EMBEDDING";
            if (pytorch_stage == "input_norm_out")
                return "ATTENTION_NORM";
            if (pytorch_stage == "attn_out")
                return "ATTENTION_OUTPUT";
            if (pytorch_stage == "post_attn_norm_out")
                return "FFN_NORM";
            if (pytorch_stage == "ffn_out")
                return "FFN_DOWN";
            if (pytorch_stage == "out")
                return "FFN_RESIDUAL";
            if (pytorch_stage == "final_norm_out")
                return "FINAL_NORM";
            if (pytorch_stage == "logits")
                return "LM_HEAD";

            // If no mapping found, return original
            return pytorch_stage;
        }

        bool parse_pytorch_key(const std::string &key, int &layer_index, std::string &stage_name)
        {
            // Format: "pytorch_layer_N_stage" or "pytorch_embeddings" or "pytorch_final_norm_out"
            size_t prefix_pos = key.find("pytorch_");
            if (prefix_pos == std::string::npos)
            {
                return false;
            }

            std::string content = key.substr(prefix_pos + 8); // Skip "pytorch_"

            // Special keys
            if (content == "embeddings")
            {
                layer_index = -1;
                stage_name = "embeddings";
                return true;
            }
            if (content == "final_norm_out")
            {
                layer_index = -1;
                stage_name = "final_norm_out";
                return true;
            }
            if (content == "logits")
            {
                layer_index = -1;
                stage_name = "logits";
                return true;
            }

            // layer_N_stage format
            if (content.substr(0, 6) == "layer_")
            {
                size_t underscore_pos = content.find('_', 6);
                if (underscore_pos == std::string::npos)
                {
                    return false;
                }

                std::string layer_str = content.substr(6, underscore_pos - 6);
                layer_index = std::stoi(layer_str);
                stage_name = content.substr(underscore_pos + 1);
                return true;
            }

            return false;
        }

    } // namespace testing
} // namespace llaminar
