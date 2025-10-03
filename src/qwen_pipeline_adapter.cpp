/**
 * @file qwen_pipeline_adapter.cpp
 */
#include "qwen_pipeline_adapter.h"
#include "abstract_pipeline.h"
#include "logger.h"
#include "utils/debug_env.h"

namespace llaminar
{
    QwenPipelineAdapter::QwenPipelineAdapter(const TransformerLayerConfig &cfg) : cfg_(cfg)
    {
        legacy_ = std::make_unique<MPITransformerPipeline>(cfg_);
    }

    bool QwenPipelineAdapter::prefill(const std::vector<int> &tokens, const IModelWeights &weights_base, StageContext &ctx)
    {
        ctx.stage = InferenceStage::Prefill;
        ctx.seq_len = (int)tokens.size();
        current_tokens_ = tokens; // retain for decode continuation
        if (legacy_)
            legacy_->setStagePrefill();
        const auto *wm = dynamic_cast<const QwenModelWeights *>(&weights_base);
        if (!wm)
        {
            LOG_ERROR("QwenPipelineAdapter: invalid weights type");
            return false;
        }
        // Legacy pipeline executes entire forward pass and produces logits for last token
        if (!legacy_)
            return false;
        if (!legacy_->execute(tokens, wm->inner, last_logits_))
        {
            LOG_ERROR("QwenPipelineAdapter: legacy execute failed in prefill");
            return false;
        }
        if (legacy_)
        {
            ctx.kv_capacity = legacy_->getKVCacheCapacity();
            ctx.kv_used = legacy_->getKVCacheUsed();
        }
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

    bool QwenPipelineAdapter::logits(std::shared_ptr<TensorBase> &out_logits)
    {
        out_logits = last_logits_;
        return (bool)out_logits;
    }

    static std::unique_ptr<AbstractPipeline> createQwen(const TransformerLayerConfig &cfg)
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
