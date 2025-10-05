/**
 * @file llama_pipeline_adapter.cpp
 * @brief Llama pipeline adapter implementation.
 * @author David Sanftenberg
 */
#include "llama_pipeline_adapter.h"
#include "abstract_pipeline.h"
#include "qwen_pipeline.h"
#include "logger.h"
#include "utils/debug_env.h"

namespace llaminar
{
    LlamaPipelineAdapter::LlamaPipelineAdapter(const ModelConfig &cfg) : cfg_(cfg)
    {
        legacy_ = std::make_unique<QwenPipeline>(cfg);
        LOG_INFO("LlamaPipelineAdapter initialized for arch='" << cfg.architecture << "'");
    }

    std::unique_ptr<IModelWeights> LlamaPipelineAdapter::loadWeights(const std::string &path)
    {
        // Use ModelLoader directly instead of deprecated free function
        ModelLoader loader;
        if (!loader.loadModel(path))
        {
            throw std::runtime_error("ModelLoader loadModel failed: " + path);
        }
        auto loaded = llaminar::loadModelWeights_impl_bridge(loader, cfg_.getLayerConfig());
        auto weights = std::make_unique<LlamaModelWeights>();
        weights->inner = std::move(loaded);
        return weights;
    }

    bool LlamaPipelineAdapter::prefill(const std::vector<int> &tokens, const IModelWeights &weights_base, StageContext &ctx)
    {
        ctx.stage = InferenceStage::Prefill;
        ctx.seq_len = static_cast<int>(tokens.size());
        current_tokens_ = tokens; // retain for decode continuation

        if (legacy_)
            legacy_->setStagePrefill();

        const auto *wm = dynamic_cast<const LlamaModelWeights *>(&weights_base);
        if (!wm)
        {
            LOG_ERROR("LlamaPipelineAdapter: invalid weights type (expected LlamaModelWeights)");
            return false;
        }

        // Legacy pipeline executes entire forward pass and produces logits for last token
        if (!legacy_)
            return false;

        if (!legacy_->execute(tokens, wm->inner, last_logits_))
        {
            LOG_ERROR("LlamaPipelineAdapter: legacy execute failed in prefill");
            return false;
        }

        if (legacy_)
        {
            ctx.kv_capacity = legacy_->getKVCacheCapacity();
            ctx.kv_used = legacy_->getKVCacheUsed();
        }

        return true;
    }

    bool LlamaPipelineAdapter::decode(int next_token, const IModelWeights &weights_base, StageContext &ctx)
    {
        ctx.stage = InferenceStage::Decode;
        current_tokens_.push_back(next_token);
        ctx.seq_len = static_cast<int>(current_tokens_.size());
        ctx.generated += 1;

        if (legacy_)
            legacy_->setStageDecode();

        const auto *wm = dynamic_cast<const LlamaModelWeights *>(&weights_base);
        if (!wm)
        {
            LOG_ERROR("LlamaPipelineAdapter: invalid weights type (decode)");
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
                if (last_logits_ && incr_logits && last_logits_->shape().size() == 2 &&
                    incr_logits->shape().size() == 2 && incr_logits->shape()[0] == 1)
                {
                    int vocab = incr_logits->shape()[1];
                    int history_rows = last_logits_->shape()[0];
                    // Sanity: history should be current_seq_len - 1
                    if (history_rows == ctx.seq_len - 1)
                    {
                        auto combined = TensorFactory::create_simple({ctx.seq_len, vocab});
                        // copy history
                        std::memcpy(combined->data(), last_logits_->data(),
                                    static_cast<size_t>(history_rows) * vocab * sizeof(float));
                        // append new row
                        std::memcpy(combined->data() + static_cast<size_t>(history_rows) * vocab,
                                    incr_logits->data(), static_cast<size_t>(vocab) * sizeof(float));
                        last_logits_ = combined;
                    }
                    else
                    {
                        // Fallback: unexpected history shape; just propagate incremental logits
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
            LOG_ERROR("LlamaPipelineAdapter: legacy execute failed in decode (replay fallback)");
            return false;
        }

        if (legacy_)
        {
            ctx.kv_capacity = legacy_->getKVCacheCapacity();
            ctx.kv_used = legacy_->getKVCacheUsed();
        }

        return true;
    }

    bool LlamaPipelineAdapter::logits(std::shared_ptr<TensorBase> &out_logits)
    {
        out_logits = last_logits_;
        return static_cast<bool>(out_logits);
    }

    const KVCacheState *LlamaPipelineAdapter::kvCacheState() const
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

    bool LlamaPipelineAdapter::ensureKVCapacity(int required_tokens)
    {
        if (!legacy_)
            return false;
        return legacy_->ensureKVCapacityPublic(required_tokens);
    }

    // Factory registration
    static std::unique_ptr<AbstractPipeline> createLlama(const ModelConfig &cfg)
    {
        return std::make_unique<LlamaPipelineAdapter>(cfg);
    }

    void registerLlamaPipeline()
    {
        PipelineFactory::instance().registerCreator("llama", &createLlama);
        LOG_INFO("Registered 'llama' architecture with PipelineFactory");
    }

} // namespace llaminar
