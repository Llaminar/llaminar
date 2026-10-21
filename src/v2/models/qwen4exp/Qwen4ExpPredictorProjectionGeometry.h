/**
 * @file Qwen4ExpPredictorProjectionGeometry.h
 * @brief One checked Dense0/PerBranch projection-row authority for graphs and workspace BOM.
 *
 * Allocation capacity remains a graph/arena property. This metadata-only helper
 * prices the actual mutually exclusive predictor consumers and returns row
 * geometry, never a physical reservation or an allocation decision.
 */
#pragma once
#include "Qwen4ExpPredictorPolicy.h"
#include "../../execution/config/RuntimeConfig.h"
#include "../../loaders/WeightIdentity.h"
#include "../../kernels/qwen4exp/Qwen4ExpKernels.h"
#include <algorithm>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string_view>
namespace llaminar2 {
/** @brief Checked original-token versus branch-projection row arithmetic. */
class Qwen4ExpPredictorProjectionGeometry final {
public:
    /**
     * @brief Expand one source-authenticated PerBranch FC invocation without multiplying count storage.
     * @param token_rows Positive exact physical token-row envelope, already request-flattened once.
     * @param branches Authenticated checkpoint branch count; must match the native4-branch contract.
     * @return Native INT32 FC row count, shared by graph construction and workspace admission.
     * @throws std::invalid_argument For zero rows or a foreign branch policy.
     * @throws std::overflow_error If the native expanded dimension would overflow.
     */
    static int perBranchRows(int token_rows,uint32_t branches)
    {
        if (token_rows<=0 || branches!=qwen4exp::kResidualBranches)
            throw std::invalid_argument("QFN PerBranch projection requires positive rows and its authenticated native branch count");
        if (token_rows>std::numeric_limits<int>::max()/static_cast<int>(branches))
            throw std::overflow_error("QFN PerBranch projection rows exceed native INT32 geometry");
        return token_rows*static_cast<int>(branches);
    }
    /**
     * @brief Authenticate the one existing QFN predictor shard/replica policy before BOM construction.
     * @param selected Explicit frozen planning policy; absence is invalid for an active QFN predictor.
     * @return Existing native layout tag, without selecting a replacement policy.
     * @throws std::invalid_argument For a missing or unknown policy.
     */
    static MTPShiftedKVHeadLayout requireLayout(const std::optional<MTPShiftedKVHeadLayout> &selected)
    {
        if (!selected || (*selected!=MTPShiftedKVHeadLayout::PrimaryTensorParallelShard &&
                          *selected!=MTPShiftedKVHeadLayout::FullModelPerParticipant))
            throw std::invalid_argument("Active QFN predictor workspace requires its explicit admitted dense/cache layout");
        return *selected;
    }
    /**
     * @brief Price only graph-consumed predictor rows, preserving every ordinary/main projection.
     * @param ordinary_rows Already-flattened retained invocation rows for this matrix.
     * @param shifted_prefill_rows Already-flattened admitted input-mix/KV-only row envelope.
     * @param role Canonical exact semantic matrix role from inferWeightRole.
     * @param name Exact source name, used only for HC attention down/up consumer distinction.
     * @param layer Physical source layer for this matrix.
     * @param main_layers Exact unchanged decoder count.
     * @param branches Null outside QFN; otherwise authenticated exact branch count.
     * @param predictor Exact admitted predictor policy, required for a QFN predictor source.
     * @return Rows for this consumer; FC alone expands each token into4 physical GEMM rows.
     * @throws std::exception For incomplete predictor policy or checked arithmetic overflow.
     */
    static int workspaceRows(int ordinary_rows,int shifted_prefill_rows,WeightRole role,std::string_view name,
        int layer,uint32_t main_layers,std::optional<uint32_t> branches,
        const std::optional<Qwen4ExpPredictorMetadata> &predictor)
    {
        if (ordinary_rows<=0 || shifted_prefill_rows<=0)
            throw std::invalid_argument("Projection workspace requires positive admitted row envelopes");
        if (!branches) {
            if (predictor) throw std::invalid_argument("QFN predictor policy requires its authenticated model geometry");
            return ordinary_rows;
        }
        if (layer<0 || static_cast<uint32_t>(layer)<main_layers) return ordinary_rows;
        if (!predictor || !predictor->validForMainLayers(main_layers) ||
            layer!=static_cast<int>(predictor->source_layer) || *branches!=qwen4exp::kResidualBranches)
            throw std::invalid_argument("QFN predictor workspace requires exact source/PerBranch metadata");
        const int input_rows=std::max(ordinary_rows,shifted_prefill_rows);
        if (role==WeightRole::PredictorProjection) return perBranchRows(input_rows,*branches);
        if (role==WeightRole::AttentionK || role==WeightRole::AttentionV ||
            (role==WeightRole::HyperConnectionProjection &&
                (name.ends_with(".hc_attn_down.weight") || name.ends_with(".hc_attn_up.weight"))))
            return input_rows;
        return ordinary_rows;
    }
};
} // namespace llaminar2
