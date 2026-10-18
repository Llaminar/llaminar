/**
 * @file ModelGenerationPolicy.cpp
 * @brief Revision-owned generation defaults selected by embedded GGUF identity.
 *
 * Architecture describes tensor layout, not post-training policy. Explicit
 * revision identities select the general text presets; task-specific coding or
 * vision recommendations remain explicit client overrides. No filename, API
 * alias, shape heuristic or network access participates in model admission.
 */
#include "ModelGenerationPolicy.h"
#include "../loaders/IModelLoader.h"
#include "../utils/Tokenizer.h"
#include "qwen/qwen35/Qwen35ChatTemplate.generated.h"
#include <array>
#include <cctype>
#include <string_view>

namespace llaminar2
{
    namespace
    {
        /** @brief Distinct published generation laws, independent of weight layout. */
        enum class QwenGenerationProfile
        {
            Instruct25Small, Instruct25, Qwen3, Qwen35Small, Qwen35,
            Qwen36Dense, Qwen36MoE, Qwen38Dense,
        };

        /** @brief Exact post-trained identity and its required GGUF architecture. */
        struct ProfileIdentity
        {
            std::string_view model;
            std::string_view architecture;
            QwenGenerationProfile profile;
        };

        // Pinned primary-source references and independent numeric expectations
        // live in Test__ModelGenerationPolicy. This inventory deliberately excludes
        // Base, Coder, Math, 2507 and other separately trained revisions.
        constexpr auto identities = std::to_array<ProfileIdentity>({
            {"qwen2.5-0.5b-instruct", "qwen2", QwenGenerationProfile::Instruct25Small},
            {"qwen2.5-1.5b-instruct", "qwen2", QwenGenerationProfile::Instruct25Small},
            {"qwen2.5-3b-instruct", "qwen2", QwenGenerationProfile::Instruct25},
            {"qwen2.5-7b-instruct", "qwen2", QwenGenerationProfile::Instruct25},
            {"qwen2.5-14b-instruct", "qwen2", QwenGenerationProfile::Instruct25},
            {"qwen2.5-32b-instruct", "qwen2", QwenGenerationProfile::Instruct25},
            {"qwen2.5-72b-instruct", "qwen2", QwenGenerationProfile::Instruct25},
            {"qwen3-0.6b", "qwen3", QwenGenerationProfile::Qwen3},
            {"qwen3-1.7b", "qwen3", QwenGenerationProfile::Qwen3},
            {"qwen3-4b", "qwen3", QwenGenerationProfile::Qwen3},
            {"qwen3-8b", "qwen3", QwenGenerationProfile::Qwen3},
            {"qwen3-14b", "qwen3", QwenGenerationProfile::Qwen3},
            {"qwen3-32b", "qwen3", QwenGenerationProfile::Qwen3},
            {"qwen3-30b-a3b", "qwen3moe", QwenGenerationProfile::Qwen3},
            {"qwen3-235b-a22b", "qwen3moe", QwenGenerationProfile::Qwen3},
            {"qwen3.5-0.8b", "qwen35", QwenGenerationProfile::Qwen35Small},
            {"qwen3.5-2b", "qwen35", QwenGenerationProfile::Qwen35Small},
            {"qwen3.5-4b", "qwen35", QwenGenerationProfile::Qwen35},
            {"qwen3.5-9b", "qwen35", QwenGenerationProfile::Qwen35},
            {"qwen3.5-27b", "qwen35", QwenGenerationProfile::Qwen35},
            {"qwen3.5-35b-a3b", "qwen35moe", QwenGenerationProfile::Qwen35},
            {"qwen3.5-122b-a10b", "qwen35moe", QwenGenerationProfile::Qwen35},
            {"qwen3.6-27b", "qwen35", QwenGenerationProfile::Qwen36Dense},
            {"qwen3.6-35b-a3b", "qwen35moe", QwenGenerationProfile::Qwen36MoE},
            {"qwen3.8-27b", "qwen35", QwenGenerationProfile::Qwen38Dense},
        });

        /**
         * @brief Canonicalize display-name whitespace/case without loosening identity boundaries.
         * @param text One embedded name, never a filesystem path.
         * @return Lowercase identifier with single hyphens between display-name words.
         */
        std::string normalizeIdentity(std::string_view text)
        {
            std::string result;
            bool separator = false;
            for (unsigned char c : text)
            {
                if (std::isspace(c))
                {
                    separator = !result.empty();
                    continue;
                }
                if (separator) result += '-';
                separator = false;
                result += static_cast<char>(std::tolower(c));
            }
            return result;
        }

        /**
         * @brief Extract only official Qwen repository identities from GGUF provenance.
         * @param url Embedded repository URL.
         * @return Model identifier, or empty for unrelated provenance.
         */
        std::string officialRepositoryIdentity(std::string_view url)
        {
            constexpr std::string_view prefix = "https://huggingface.co/Qwen/";
            if (!url.starts_with(prefix)) return {};
            url.remove_prefix(prefix.size());
            if (url.ends_with('/')) url.remove_suffix(1);
            return std::string(url);
        }

        /**
         * @brief Construct the complete neutral-plus-card sampling recommendation.
         * @param temperature Card's temperature.
         * @param top_p Card's nucleus threshold; all audited cards use top-k 20.
         * @param presence Card's additive presence penalty.
         * @param repetition Card's multiplicative repetition penalty.
         * @return Complete defaults with no frequency/DRY penalty or min-p filtering.
         */
        SamplingParams recommendation(float temperature, float top_p,
                                      float presence, float repetition = 1.0f)
        {
            SamplingParams params;
            params.temperature = temperature;
            params.top_p = top_p;
            params.top_k = 20;
            params.presence_penalty = presence;
            params.repetition_penalty = repetition;
            return params;
        }
    }

    ModelGenerationPolicy ModelGenerationPolicy::fromMetadata(const IModelLoader &loader)
    {
        const auto basename = loader.getString("general.basename");
        const auto size = loader.getString("general.size_label");
        const auto finetune = loader.getString("general.finetune");
        // Prefer the current artifact's identity over its training provenance:
        // a later named revision can still cite an earlier base model.
        const std::array candidates = {
            loader.getString("general.name"), basename,
            basename + "-" + finetune, basename + "-" + size,
            basename + "-" + size + "-" + finetune,
            loader.getString("general.base_model.0.name"),
            officialRepositoryIdentity(loader.getString("general.base_model.0.repo_url")),
            officialRepositoryIdentity(loader.getString("general.repo_url"))};
        const auto architecture = loader.architecture();
        ModelGenerationPolicy result;
        for (const auto &candidate : candidates)
        {
            const auto identity = normalizeIdentity(candidate);
            for (const auto &entry : identities)
            {
                if (entry.architecture != architecture || entry.model != identity) continue;
                // These exact revisions retain the maintained template's
                // continuation fixes. Qwen3.8 owns new reasoning defaults in
                // its GGUF template despite sharing the qwen35 graph layout.
                if (entry.profile == QwenGenerationProfile::Qwen35Small ||
                    entry.profile == QwenGenerationProfile::Qwen35 ||
                    entry.profile == QwenGenerationProfile::Qwen36Dense ||
                    entry.profile == QwenGenerationProfile::Qwen36MoE)
                    result.template_source_ = TemplateSource::Qwen35Maintained;
                switch (entry.profile)
                {
                case QwenGenerationProfile::Instruct25Small:
                case QwenGenerationProfile::Instruct25:
                    result.default_thinking_mode_ = ThinkingMode::Disabled;
                    result.thinking_ = result.non_thinking_ = recommendation(
                        0.7f, 0.8f, 0.0f,
                        entry.profile == QwenGenerationProfile::Instruct25Small ? 1.1f : 1.05f);
                    break;
                case QwenGenerationProfile::Qwen3:
                    result.thinking_ = recommendation(0.6f, 0.95f, 0.0f);
                    result.non_thinking_ = recommendation(0.7f, 0.8f, 0.0f);
                    break;
                case QwenGenerationProfile::Qwen35Small:
                    result.default_thinking_mode_ = ThinkingMode::Disabled;
                    result.thinking_ = recommendation(1.0f, 0.95f, 1.5f);
                    result.non_thinking_ = recommendation(1.0f, 1.0f, 2.0f);
                    break;
                case QwenGenerationProfile::Qwen35:
                case QwenGenerationProfile::Qwen36MoE:
                    result.thinking_ = recommendation(1.0f, 0.95f, 1.5f);
                    result.non_thinking_ = recommendation(0.7f, 0.8f, 1.5f);
                    break;
                case QwenGenerationProfile::Qwen36Dense:
                case QwenGenerationProfile::Qwen38Dense:
                    result.thinking_ = recommendation(1.0f, 0.95f, 0.0f);
                    result.non_thinking_ = recommendation(0.7f, 0.8f, 1.5f);
                    break;
                }
                return result;
            }
        }
        return result;
    }
    void ModelGenerationPolicy::applyChatTemplate(ITokenizer &tokenizer) const
    {
        if (template_source_ == TemplateSource::Qwen35Maintained)
            tokenizer.setChatTemplate(ChatTemplate::create(
                std::string(qwen35::kCommunityChatTemplate), "", ""));
    }

}
