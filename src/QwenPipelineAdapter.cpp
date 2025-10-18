/**
 * @file QwenPipelineAdapter.cpp
 */
#include "QwenPipelineAdapter.h"
#include "AbstractPipeline.h"
#include "QwenPipeline.h"
#include "Logger.h"
#include "utils/DebugEnv.h"

namespace llaminar
{
    QwenPipelineAdapter::QwenPipelineAdapter(const ModelConfig &cfg) : cfg_(cfg)
    {
        legacy_ = std::make_unique<QwenPipeline>(cfg);
    }

    std::unique_ptr<IModelWeights> QwenPipelineAdapter::loadWeights(const std::string &path)
    {
        // Use ModelLoader directly instead of deprecated free function
        ModelLoader loader;
        if (!loader.loadModel(path))
        {
            throw std::runtime_error("ModelLoader loadModel failed: " + path);
        }
        auto loaded = llaminar::loadModelWeights_impl_bridge(loader, cfg_.getLayerConfig());
        auto weights = std::make_unique<QwenModelWeights>();
        weights->inner = std::move(loaded);

        // CRITICAL: Validate all weights against canonical GGUF format contracts
        // This ensures kernels can trust the weight dimensions/orientations without runtime detection
        // NOW supports MPI-aware validation (checks sliced dimensions when mpi_size > 1)
        int mpi_rank = 0, mpi_size = 1;
#ifdef LLAMINAR_HAVE_MPI
        int mpi_initialized = 0;
        MPI_Initialized(&mpi_initialized);
        if (mpi_initialized)
        {
            MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
            MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
        }
#endif

        try
        {
            weights->validate_with_mpi(cfg_.getLayerConfig(), mpi_rank, mpi_size);
            if (mpi_size > 1)
            {
                LOG_INFO("✓ All weights validated with MPI slicing (rank " << mpi_rank << "/" << mpi_size << ")");
            }
            else
            {
                LOG_INFO("✓ All weights validated against canonical GGUF format");
            }
        }
        catch (const std::exception &e)
        {
            LOG_ERROR("Weight validation failed: " << e.what());
            throw; // Re-throw to fail fast with clear error
        }

        return weights;
    }
    bool QwenPipelineAdapter::prefill(const std::vector<int> &tokens, const IModelWeights &weights_base, StageContext &ctx)
    {
        {
            std::ofstream trace("/tmp/adapter_trace.txt", std::ios::app);
            trace << "ENTERING prefill, tokens.size()=" << tokens.size() << std::endl;
            trace.close();
        }

        // Use the new provider-based prefill path (PrefillProviderFactory)
        // This enables proper COSMA vs OpenBLAS selection based on sequence length
        ctx.stage = InferenceStage::Prefill;
        ctx.seq_len = (int)tokens.size();
        current_tokens_ = tokens; // retain for decode continuation

        if (!legacy_)
        {
            LOG_ERROR("QwenPipelineAdapter: legacy pipeline is null");
            return false;
        }

        {
            std::ofstream trace("/tmp/adapter_trace.txt", std::ios::app);
            trace << "ABOUT TO CALL legacy_->prefill" << std::endl;
            trace.close();
        }

        // Call the new provider-based prefill method (NOT legacy execute)
        // This will use PrefillProviderFactory to select OpenBLAS or COSMA based on sequence length
        if (!legacy_->prefill(tokens, weights_base, ctx))
        {
            LOG_ERROR("QwenPipelineAdapter: provider-based prefill failed");
            return false;
        }

        {
            std::ofstream trace("/tmp/adapter_trace.txt", std::ios::app);
            trace << "RETURNED FROM legacy_->prefill" << std::endl;
            trace.close();
        }

        // Get the logits from the pipeline
        if (!legacy_->logits(last_logits_))
        {
            LOG_ERROR("QwenPipelineAdapter: failed to retrieve logits after prefill");
            return false;
        }

        {
            std::ofstream trace("/tmp/adapter_trace.txt", std::ios::app);
            trace << "GOT LOGITS, about to check" << std::endl;
            if (last_logits_ && last_logits_->data())
            {
                size_t total_elements = 1;
                for (auto d : last_logits_->shape())
                    total_elements *= d;
                double sum_sq = 0.0;
                for (size_t i = 0; i < total_elements; ++i)
                    sum_sq += last_logits_->data()[i] * last_logits_->data()[i];
                double l2_norm = std::sqrt(sum_sq / total_elements);
                trace << "FINAL LOGITS: L2_norm=" << l2_norm << std::endl;
            }
            trace.close();
        }

        // Context already updated by legacy_->prefill()
        return true;
    }

    bool QwenPipelineAdapter::decode(int next_token, const IModelWeights &weights_base, StageContext &ctx)
    {
        ctx.stage = InferenceStage::Decode;
        current_tokens_.push_back(next_token);
        ctx.seq_len = (int)current_tokens_.size();
        ctx.generated += 1;
        if (legacy_)
            legacy_->setStageDecode();
        const auto *wm = dynamic_cast<const QwenModelWeights *>(&weights_base);
        if (!wm)
        {
            LOG_ERROR("QwenPipelineAdapter: invalid weights type (decode)");
            return false;
        }
        // Attempt incremental decode first (will return false if disabled / not possible)
        if (legacy_)
        {
            std::shared_ptr<TensorBase> incr_logits;
            if (legacy_->incrementalDecodeToken(next_token, wm->inner, incr_logits))
            {
                // Expect incremental path to return only the new token logits with shape [1, vocab]
                // but the AbstractPipelineParity test expects cumulative logits with shape
                // [current_sequence_length, vocab]. If we already have history from prefill or prior
                // decodes, append the new row to produce the expected shape.
                if (last_logits_ && incr_logits && last_logits_->shape().size() == 2 && incr_logits->shape().size() == 2 && incr_logits->shape()[0] == 1)
                {
                    int vocab = incr_logits->shape()[1];
                    int history_rows = last_logits_->shape()[0];
                    // Sanity: history should be current_seq_len - 1
                    if (history_rows == ctx.seq_len - 1)
                    {
                        auto combined = TensorFactory::create_simple({ctx.seq_len, vocab});
                        // copy history
                        std::memcpy(combined->data(), last_logits_->data(), (size_t)history_rows * vocab * sizeof(float));
                        // append new row
                        std::memcpy(combined->data() + (size_t)history_rows * vocab, incr_logits->data(), (size_t)vocab * sizeof(float));
                        last_logits_ = combined;
                    }
                    else
                    {
                        // Fallback: unexpected history shape; just propagate incremental logits (parity test may fail)
                        last_logits_ = incr_logits;
                    }
                }
                else
                {
                    // No prior history (edge case) or unexpected shapes; forward incremental logits
                    last_logits_ = incr_logits;
                }
                return true; // incremental path succeeded
            }
        }

        // Fallback: replay full sequence
        if (legacy_ && !legacy_->execute(current_tokens_, wm->inner, last_logits_))
        {
            LOG_ERROR("QwenPipelineAdapter: legacy execute failed in decode (replay fallback)");
            return false;
        }
        if (legacy_)
        {
            ctx.kv_capacity = legacy_->getKVCacheCapacity();
            ctx.kv_used = legacy_->getKVCacheUsed();
        }
        return true;
    }

    bool QwenPipelineAdapter::prefillBatch(
        const std::vector<std::vector<int>> &token_batches,
        const IModelWeights &weights_base,
        StageContext &ctx,
        std::shared_ptr<TensorBase> &out_logits)
    {
        if (!legacy_)
        {
            LOG_ERROR("QwenPipelineAdapter: legacy pipeline is null (prefillBatch)");
            return false;
        }

        const auto *wm = dynamic_cast<const QwenModelWeights *>(&weights_base);
        if (!wm)
        {
            LOG_ERROR("QwenPipelineAdapter: invalid weights type (prefillBatch)");
            return false;
        }

        // Forward to QwenPipeline batch implementation
        if (!legacy_->prefillBatch(token_batches, weights_base, ctx, out_logits))
        {
            LOG_ERROR("QwenPipelineAdapter: prefillBatch failed");
            return false;
        }

        // Store logits for logits() accessor
        last_logits_ = out_logits;
        return true;
    }

    bool QwenPipelineAdapter::decodeBatch(
        const std::vector<int> &next_tokens,
        const IModelWeights &weights_base,
        StageContext &ctx,
        std::shared_ptr<TensorBase> &out_logits)
    {
        if (!legacy_)
        {
            LOG_ERROR("QwenPipelineAdapter: legacy pipeline is null (decodeBatch)");
            return false;
        }

        const auto *wm = dynamic_cast<const QwenModelWeights *>(&weights_base);
        if (!wm)
        {
            LOG_ERROR("QwenPipelineAdapter: invalid weights type (decodeBatch)");
            return false;
        }

        // Forward to QwenPipeline batch implementation
        if (!legacy_->decodeBatch(next_tokens, weights_base, ctx, out_logits))
        {
            LOG_ERROR("QwenPipelineAdapter: decodeBatch failed");
            return false;
        }

        // Store logits for logits() accessor
        last_logits_ = out_logits;
        return true;
    }

    bool QwenPipelineAdapter::logits(std::shared_ptr<TensorBase> &out_logits)
    {
        out_logits = last_logits_;
        return (bool)out_logits;
    }

    static std::unique_ptr<AbstractPipeline> createQwen(const ModelConfig &cfg)
    {
        return std::make_unique<QwenPipelineAdapter>(cfg);
    }

    void registerQwenPipeline()
    {
        PipelineFactory::instance().registerCreator("qwen", &createQwen);
    }

    const KVCacheState *QwenPipelineAdapter::kvCacheState() const
    {
        if (!legacy_)
            return nullptr;
        // Populate a static mirror (thread-safe assumption: single-threaded pipeline calls)
        static KVCacheState mirror;
        mirror.capacity_tokens = legacy_->getKVCacheCapacity();
        mirror.used_tokens = legacy_->getKVCacheUsed();
        mirror.growth_events = legacy_->getKVCacheGrowthEvents();
        return &mirror;
    }

    bool QwenPipelineAdapter::ensureKVCapacity(int required_tokens)
    {
        if (!legacy_)
            return false;
        return legacy_->ensureKVCapacityPublic(required_tokens);
    }
} // namespace llaminar
