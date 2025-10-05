#pragma once

#include "tokenizer_interface.h"
#include "../argument_parser.h"
#include "../abstract_pipeline.h"
#include "../qwen_pipeline_adapter.h" // for QwenModelWeights definition (inherits IModelWeights)
#include <memory>
#include <vector>
#include <string>
#include <cstdint>
#include <functional>

namespace llaminar
{
    namespace chat
    {
        using ResponseCallback = std::function<void(const std::string &text, bool is_complete)>;

        class ResponseGenerator
        {
        public:
            // Construct with pipeline + load weights internally from params.model_file
            ResponseGenerator(std::shared_ptr<TokenizerInterface> tokenizer,
                              std::shared_ptr<AbstractPipeline> pipeline,
                              const LlaminarParams &params);

            // Construct with pre-loaded weights (wrapped)
            ResponseGenerator(std::shared_ptr<TokenizerInterface> tokenizer,
                              std::shared_ptr<AbstractPipeline> pipeline,
                              const LlaminarParams &params,
                              const QwenModelWeights &weights);

            ~ResponseGenerator() = default;

            std::string generateResponse(const std::vector<int32_t> &prompt_tokens,
                                         ResponseCallback callback = nullptr);

            std::string generateStreamingResponse(const std::vector<int32_t> &prompt_tokens,
                                                  ResponseCallback callback);

            void setGenerationParams(float temperature, int32_t top_k,
                                     float top_p, int32_t max_tokens);

            void setStopConditions(const std::vector<int32_t> &stop_tokens,
                                   const std::vector<std::string> &stop_sequences);

        private:
            std::shared_ptr<TokenizerInterface> tokenizer_;
            std::shared_ptr<AbstractPipeline> pipeline_;
            const LlaminarParams &params_;
            QwenModelWeights weights_; // wrapper holding inner ModelWeights
            bool have_weights_ = false;
            bool prefilled_ = false;
            StageContext stage_ctx_{}; // tracks prefill/decode progress

            float temperature_;
            int32_t top_k_;
            float top_p_;
            int32_t max_tokens_;

            std::vector<int32_t> stop_tokens_;
            int32_t max_new_tokens_;
            std::vector<std::string> stop_sequences_;

            bool shouldStop(const std::vector<int32_t> &tokens,
                            const std::string &text) const;
            int32_t sampleToken(const std::vector<float> &logits);
            void applyTemperature(std::vector<float> &logits, float temperature);
            void applyTopK(std::vector<float> &logits, int32_t k);
            void applyTopP(std::vector<float> &logits, float p);
            std::vector<float> softmax(const std::vector<float> &logits);
            int32_t sampleFromProbs(const std::vector<float> &probs);
            bool ensureWeights();
            std::vector<float> fetchLastLogitsRow(const std::shared_ptr<TensorBase> &logits_tensor) const;
        };

    } // namespace chat
} // namespace llaminar
