/**
 * @file Qwen4ExpOrdinaryGenerationWorkload.h
 * @brief Select original sustained movement work separately from short capture controls.
 *
 * Dynamic expert movement is asynchronous physical work. A three-token capture
 * control cannot promise completed migration before its request retires. The
 * original case already owns a sustained input, sampler and response horizon;
 * this projection consumes that declaration without changing production policy.
 */
#pragma once
#include "../ModelParityDefinition.h"
#include "utils/Sampler.h"
#include "utils/Tokenizer.h"
#include <span>

namespace llaminar2::test::parity::qwen4exp {
/** Two existing public requests; neither phase is a retry or a force-drain. */
enum class Qwen4ExpOrdinaryRequestPhase { First, Reuse };
/** Immutable request values retained through original reset/reuse validation. */
struct Qwen4ExpNativeGenerationRequest {
    std::vector<int32_t> tokens;
    SamplingParams sampling;
    int maximum_tokens;
    int minimum_tokens;
};
/** @brief Source-authored functional workload; no runner/topology capability is granted. */
class Qwen4ExpOrdinaryGenerationWorkload final {
public:
    /**
     * @brief Project the original case before inference and encode its declared chat exactly once.
     * @param cell Original typed source case, including movement expectation and sustained input.
     * @param checkpoint_tokens Existing short numerical prompt, unchanged for static controls.
     * @param checkpoint_sampling Original short-control sampler, unchanged for static controls.
     * @param encode_chat Original tokenizer's public four-argument encodeChat operation.
     * @throws std::invalid_argument for non-Off requests or empty original/tokenized input.
     */
    template<class EncodeChat>
    static Qwen4ExpOrdinaryGenerationWorkload fromOriginalCase(const ModelParityCase& cell,
        std::span<const int32_t> checkpoint_tokens, const SamplingParams& checkpoint_sampling,
        EncodeChat&& encode_chat)
    {
        if(cell.mtp!=ModelParityMTP::Off || checkpoint_tokens.empty())
            throw std::invalid_argument("Ordinary workload requires its original Off cell and checkpoint input");
        const bool sustained=cell.requiresPhysicalExpertMovement();
        std::vector<int32_t> tokens(checkpoint_tokens.begin(),checkpoint_tokens.end());
        auto sampling=checkpoint_sampling;
        if(sustained) {
            const std::vector<ChatMessage> messages{{"system",cell.model.generation_prompt.system()},
                {"user",cell.model.generation_prompt.user()}};
            const auto encoded=encode_chat(messages,true,std::string{},false);
            if(encoded.empty())throw std::invalid_argument("Declared sustained generation chat encoded no original tokens");
            tokens.assign(encoded.begin(),encoded.end());
            sampling=SamplingParams{};
            sampling.temperature=static_cast<float>(kModelParityGenerationTemperature);
            sampling.top_k=kModelParityGenerationTopK;
            sampling.top_p=static_cast<float>(kModelParityGenerationTopP);
            sampling.seed=cell.model.generation_seed.value();
        }
        return Qwen4ExpOrdinaryGenerationWorkload(cell.testName(),std::move(tokens),std::move(sampling),
            sustained,cell.generation_workload.maximumTokens(),cell.generation_workload.minimumTokens());
    }
    /** @return Exact original complete case identity, not a backend/name heuristic. */
    const std::string& originalCase()const noexcept{return original_case_;}
    /** @return Whether the original case requires physical movement and its sustained chat. */
    bool sustained()const noexcept{return sustained_;}
    /** @return Same immutable token stream/sampler for both requests and original declared horizons. */
    Qwen4ExpNativeGenerationRequest request(Qwen4ExpOrdinaryRequestPhase phase)const
    {
        if(phase!=Qwen4ExpOrdinaryRequestPhase::First && phase!=Qwen4ExpOrdinaryRequestPhase::Reuse)
            throw std::invalid_argument("Unknown ordinary request phase");
        const int short_budget=phase==Qwen4ExpOrdinaryRequestPhase::First?3:2;
        return {tokens_,sampling_,sustained_?maximum_:short_budget,sustained_?minimum_:short_budget};
    }
private:
    /** @brief Seal one source selection after complete natural input validation. */
    Qwen4ExpOrdinaryGenerationWorkload(std::string original,std::vector<int32_t> tokens,SamplingParams sampling,
        bool sustained,int maximum,int minimum)
        :original_case_(std::move(original)),tokens_(std::move(tokens)),sampling_(std::move(sampling)),
         sustained_(sustained),maximum_(maximum),minimum_(minimum){}
    std::string original_case_;
    std::vector<int32_t> tokens_;
    SamplingParams sampling_;
    bool sustained_;
    int maximum_;
    int minimum_;
};
} // namespace llaminar2::test::parity::qwen4exp
