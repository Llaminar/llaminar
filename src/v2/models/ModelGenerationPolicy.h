/**
 * @file ModelGenerationPolicy.h
 * @brief Immutable model-revision sampling, reasoning and chat-template policy.
 *
 * GGUF architecture names describe weight layouts, not generation training:
 * Qwen3.8-27B still identifies as qwen35. Resolve the model's generation profile
 * from its embedded identity once at initialization, then select by request
 * mode. The same revision owns its template; a shared tensor layout cannot
 * replace a newer model's reasoning or history-retention contract. Explicit
 * HTTP sampling fields and CLI template selection remain owned by the client.
 */
#pragma once

#include "../utils/Sampler.h"
#include "../utils/ThinkingMode.h"

namespace llaminar2
{
    class IModelLoader;
    class ITokenizer;

    /** @brief Revision-owned generation policy installed once at model startup. */
    class ModelGenerationPolicy
    {
    public:
        /** @brief Construct standard recommendations for an uninitialized model. */
        ModelGenerationPolicy() = default;

        /**
         * @brief Resolve model-owned recommendations without inspecting its filename.
         * @param loader Loaded GGUF metadata authority; no tensor reads are performed.
         * @return Mode-specific policy for the identified model revision.
         */
        static ModelGenerationPolicy fromMetadata(const IModelLoader &loader);

        /**
         * @brief Install this revision's maintained template, if one is selected.
         * @param tokenizer Fresh tokenizer owning its model's embedded template.
         *
         * Called before any explicit CLI override. Models with their own
         * embedded template policy retain that exact template and its defaults.
         * Selection never probes rendering failures or retries another format.
         */
        void applyChatTemplate(ITokenizer &tokenizer) const;

        /** @return The documented default reasoning mode for this loaded revision. */
        ThinkingMode defaultThinkingMode() const { return default_thinking_mode_; }

        /**
         * @brief Select the recommendation for this request's reasoning mode.
         * @param mode The same reasoning choice passed to the chat template.
         * @return Immutable model defaults; request overrides are merged separately.
         */
        const SamplingParams &forMode(ThinkingMode mode) const
        {
            if (mode == ThinkingMode::ModelDefault)
                mode = default_thinking_mode_;
            return mode == ThinkingMode::Enabled ? thinking_ : non_thinking_;
        }

    private:
        /** @brief Template authority selected from the loaded revision's identity. */
        enum class TemplateSource { ModelMetadata, Qwen35Maintained };
        TemplateSource template_source_ = TemplateSource::ModelMetadata;
        SamplingParams thinking_;
        SamplingParams non_thinking_;
        ThinkingMode default_thinking_mode_ = ThinkingMode::Enabled;
    };
}
