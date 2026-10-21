/**
 * @file Qwen4ExpProjectionWorkspace.h
 * @brief Source-authenticated QFN activation correction and its shared named BOM.
 *
 * HC-up, GDN-output, main FA-output and main GDN-QKV have independent setup policies. Serial layer
 * invocations share one participant workspace; merging canonical requirement
 * names takes their maximum rather than retaining one bank per projection.
 * The operation, primary quantizer, native weights and fixed K32 folds are the
 * existing TwoTermQ8 implementation; this policy introduces no arithmetic.
 */
#pragma once
#include "Qwen4ExpHCUpWorkspace.h"
#include "Qwen4ExpConfig.h"
#include "planning/ModelMemoryProfile.h"
#include <charconv>
#include <limits>
#include <optional>

namespace llaminar2
{
/** @brief Select exact original projection roles without guessing from matrix dimensions. */
struct Qwen4ExpProjectionWorkspace final
{
    /**
     * @brief Select only an original main GDN mixed-QKV source from frozen model metadata.
     * @param requested Independent parsed policy, never inferred from a node name.
     * @param flash_next Whether the owning profile has authenticated QFN metadata.
     * @param role Canonical original source role; only FusedQKV is eligible.
     * @param name Exact original blk.L.attn_qkv.weight identity.
     * @param geometry Borrowed immutable model-owned configuration, used only during this call.
     * @return Requested policy only for a main GDN mixer; unrelated sources retain SingleQ8.
     * @throws std::invalid_argument For an unknown policy or missing selected-model geometry.
     */
    static QuantizedActivationOperandPrecision gdnQKVPrecision(
        QuantizedActivationOperandPrecision requested, bool flash_next,
        WeightRole role, std::string_view name, const Qwen4ExpConfig *geometry)
    {
        (void)quantizedActivationOperandPrecisionName(requested);
        if (requested == QuantizedActivationOperandPrecision::SingleQ8)
            return QuantizedActivationOperandPrecision::SingleQ8;
        if (flash_next && requested == QuantizedActivationOperandPrecision::TwoTermQ8 && !geometry)
            throw std::invalid_argument("QFN GDN-QKV correction requires its original immutable mixer geometry");
        if (!flash_next || !geometry || role != WeightRole::FusedQKV)
            return QuantizedActivationOperandPrecision::SingleQ8;
        const auto layer = originalQKVLayer(name);
        if (!layer || *layer >= geometry->mainLayers() || *layer >= geometry->mixers().size() ||
            geometry->mixers()[*layer] != Qwen4ExpMixer::GatedDeltaNet)
            return QuantizedActivationOperandPrecision::SingleQ8;
        return requested;
    }

    /**
     * @brief Apply the same original-source selection to the existing metadata-only planning authority.
     * @param requested Frozen independent QKV precision.
     * @param flash_next Whether the owning profile declares QFN geometry.
     * @param role Original canonical weight role.
     * @param name Original source name, never a stage label.
     * @param profile Borrowed existing retained profile; no metadata or pointer is stored here.
     * @return Requested arithmetic only for a main GDN source identified by flashNextLayerKind.
     * @throws std::invalid_argument For unknown policy or absent selected QFN profile geometry.
     */
    static QuantizedActivationOperandPrecision gdnQKVProfilePrecision(
        QuantizedActivationOperandPrecision requested, bool flash_next,
        WeightRole role, std::string_view name, const ModelMemoryProfile *profile)
    {
        (void)quantizedActivationOperandPrecisionName(requested);
        if (requested == QuantizedActivationOperandPrecision::SingleQ8)
            return QuantizedActivationOperandPrecision::SingleQ8;
        if (flash_next && requested == QuantizedActivationOperandPrecision::TwoTermQ8 &&
            (!profile || !profile->flash_next || profile->architecture != "qwen4exp"))
            throw std::invalid_argument("QFN GDN-QKV correction requires its existing immutable planning profile");
        if (!flash_next || !profile || !profile->flash_next || role != WeightRole::FusedQKV)
            return QuantizedActivationOperandPrecision::SingleQ8;
        const auto layer = originalQKVLayer(name);
        if (!layer || *layer >= static_cast<uint32_t>(profile->mainLayerCount()))
            return QuantizedActivationOperandPrecision::SingleQ8;
        return profile->flashNextLayerKind(static_cast<int>(*layer)) == Qwen4ExpLayerKind::GDN
            ? requested : QuantizedActivationOperandPrecision::SingleQ8;
    }

    /**
     * @brief Authenticate the selected complete original source interval and R1 query envelope.
     * @param policy Frozen QKV policy.
     * @param flash_next Whether immutable QFN metadata is present.
     * @param role Original semantic source role.
     * @param name Exact original source name.
     * @param geometry Model-owned immutable mixer geometry; never retained by this helper.
     * @param requests Independent request prefixes, not physical token rows.
     * @param slice Exact prepared source interval.
     * @throws std::invalid_argument For request holes, TP slices or non-dense ownership.
     */
    static void validateGDNQKVInvocation(QuantizedActivationOperandPrecision policy,
        bool flash_next, WeightRole role, std::string_view name, const Qwen4ExpConfig *geometry,
        int requests, const WeightSliceSpec &slice)
    {
        if (gdnQKVPrecision(policy, flash_next, role, name, geometry) !=
            QuantizedActivationOperandPrecision::TwoTermQ8)
            return;
        if (requests != 1)
            throw std::invalid_argument("QFN GDN-QKV TwoTermQ8 requires one request; quantized request holes are unsupported");
        const uint64_t original_rows = gdnQKVOriginalRows(*geometry);
        if (slice.source_rows != original_rows || slice.source_cols != geometry->activations().compute)
            throw std::invalid_argument("QFN GDN-QKV correction requires the complete original mixed-QKV matrix geometry");
        if (!slice.source_rows || !slice.source_cols || slice.row_start || slice.col_start ||
            slice.row_count != slice.source_rows || slice.col_count != slice.source_cols ||
            slice.expert_start || slice.expert_count || !slice.expert_ids.empty())
            throw std::invalid_argument("QFN GDN-QKV TwoTermQ8 requires its complete original dense source interval; TP arithmetic is unqualified");
    }

    /**
     * @brief Authenticate an admitted complete QKV interval using the existing planning profile.
     * @param policy Frozen QKV arithmetic.
     * @param flash_next Whether the profile is the QFN source owner.
     * @param role Original semantic source role.
     * @param name Original source name.
     * @param profile Existing retained model metadata, borrowed only during this call.
     * @param requests Independent request count.
     * @param slice Proposed exact original dense interval.
     * @throws std::invalid_argument For partial source ownership, request holes or conflicting original geometry.
     */
    static void validateGDNQKVProfileInvocation(QuantizedActivationOperandPrecision policy,
        bool flash_next, WeightRole role, std::string_view name, const ModelMemoryProfile *profile,
        int requests, const WeightSliceSpec &slice)
    {
        if (gdnQKVProfilePrecision(policy, flash_next, role, name, profile) !=
            QuantizedActivationOperandPrecision::TwoTermQ8)
            return;
        if (profile->gdn_group_count <= 0 || profile->gdn_state_size <= 0 || profile->gdn_inner_size <= 0 ||
            profile->d_model <= 0)
            throw std::invalid_argument("QFN original mixed-QKV planning geometry is incomplete");
        const auto original_rows = originalQKVRows(profile->gdn_group_count, profile->gdn_state_size, profile->gdn_inner_size);
        if (requests != 1 || slice.source_rows != original_rows ||
            slice.source_cols != static_cast<size_t>(profile->d_model) ||
            slice.row_start || slice.col_start || slice.row_count != slice.source_rows ||
            slice.col_count != slice.source_cols || slice.expert_start || slice.expert_count || !slice.expert_ids.empty())
            throw std::invalid_argument("QFN GDN-QKV planning requires R1 and its whole original dense source; partial TP is unqualified");
    }

    /**
     * @brief Resolve GDN output arithmetic from a frozen policy and original source.
     * @param requested Parsed setup selection, copied into the graph/admission record.
     * @param flash_next Whether the model owns authenticated QFN geometry.
     * @param role Canonical original weight role.
     * @param name Exact original source name, never a compute-node label.
     * @return Requested arithmetic only for a QFN GDN output; SingleQ8 otherwise.
     * @throws std::invalid_argument For an unknown policy, even without matching sources.
     */
    static QuantizedActivationOperandPrecision gdnOutputPrecision(
        QuantizedActivationOperandPrecision requested, bool flash_next,
        WeightRole role, std::string_view name)
    {
        (void)quantizedActivationOperandPrecisionName(requested);
        return flash_next && role == WeightRole::GDNProjection && name.ends_with(".ssm_out.weight")
            ? requested : QuantizedActivationOperandPrecision::SingleQ8;
    }

    /**
     * @brief Select only a main-layer original FA output, excluding the predictor source.
     * @param requested Frozen independent setup policy; unknown values always reject.
     * @param flash_next Whether immutable QFN metadata is present.
     * @param role Original weight semantic role, never a graph node label.
     * @param name Exact original blk.<source>.attn_output.weight name.
     * @param main_layers Authenticated original main layer count, independent of PP interval.
     * @return Requested arithmetic only for original main FA outputs.
     * @throws std::invalid_argument For missing main geometry on an otherwise selected source.
     */
    static QuantizedActivationOperandPrecision faOutputPrecision(
        QuantizedActivationOperandPrecision requested, bool flash_next,
        WeightRole role, std::string_view name, int main_layers)
    {
        (void)quantizedActivationOperandPrecisionName(requested);
        constexpr std::string_view suffix = ".attn_output.weight";
        if (!flash_next || role != WeightRole::AttentionWO ||
            !name.starts_with("blk.") || !name.ends_with(suffix))
            return QuantizedActivationOperandPrecision::SingleQ8;
        const auto digits = name.substr(4, name.size() - 4 - suffix.size());
        if (digits.empty() || (digits.size() > 1 && digits.front() == '0') ||
            digits.find_first_not_of("0123456789") != std::string_view::npos)
            return QuantizedActivationOperandPrecision::SingleQ8;
        int layer = -1;
        const auto parsed = std::from_chars(digits.data(), digits.data() + digits.size(), layer);
        if (digits.empty() || parsed.ec != std::errc{} || parsed.ptr != digits.data() + digits.size() || layer < 0)
            return QuantizedActivationOperandPrecision::SingleQ8;
        if (requested == QuantizedActivationOperandPrecision::TwoTermQ8 && main_layers <= 0)
            throw std::invalid_argument("QFN FA-output correction requires original main-layer geometry");
        return layer < main_layers ? requested : QuantizedActivationOperandPrecision::SingleQ8;
    }

    /**
     * @brief Validate one selected main output's exact original dense interval.
     * @param policy Frozen FA output arithmetic.
     * @param flash_next Whether immutable QFN metadata is present.
     * @param role Original weight role.
     * @param name Exact original source name.
     * @param main_layers Original full main count, never participant-local layer count.
     * @param requests Independent request prefixes; only one is qualified.
     * @param slice Actual prepared source interval, not TP degree or topology labels.
     * @throws std::invalid_argument For request holes or partial dense source ownership.
     */
    static void validateFAOutputInvocation(QuantizedActivationOperandPrecision policy,
        bool flash_next, WeightRole role, std::string_view name, int main_layers,
        int requests, const WeightSliceSpec &slice)
    {
        if (faOutputPrecision(policy, flash_next, role, name, main_layers) !=
            QuantizedActivationOperandPrecision::TwoTermQ8)
            return;
        if (requests != 1)
            throw std::invalid_argument("QFN FA-output TwoTermQ8 requires one request; quantized request holes are unsupported");
        if (!slice.source_rows || !slice.source_cols || slice.row_start || slice.col_start ||
            slice.row_count != slice.source_rows || slice.col_count != slice.source_cols ||
            slice.expert_start || slice.expert_count || !slice.expert_ids.empty())
            throw std::invalid_argument("QFN FA-output TwoTermQ8 requires its complete original dense source interval; TP arithmetic is unqualified");
    }

    /**
     * @brief Select the union of independent, source-specific correction roles.
     * @param hc_up Frozen HC-up selection.
     * @param gdn_output Frozen GDN-output selection.
     * @param flash_next Authenticated QFN model family.
     * @param role Original canonical source role.
     * @param name Exact original tensor name.
     * @param fa_output Frozen main-only FA policy; omitted by existing HC/GDN-only callers.
     * @param main_layers Original main source count required by FA selection.
     * @param gdn_qkv Independent original main GDN-QKV selection.
     * @param qfn_profile Call-only borrow of the existing canonical metadata planning profile.
     * @return Whether this invocation contributes the correction's named banks.
     * @throws std::invalid_argument For an unknown independent policy.
     */
    static bool selected(QuantizedActivationOperandPrecision hc_up,
        QuantizedActivationOperandPrecision gdn_output, bool flash_next,
        WeightRole role, std::string_view name,
        QuantizedActivationOperandPrecision fa_output = QuantizedActivationOperandPrecision::SingleQ8,
        int main_layers = 0,
        QuantizedActivationOperandPrecision gdn_qkv = QuantizedActivationOperandPrecision::SingleQ8,
        const ModelMemoryProfile *qfn_profile = nullptr)
    {
        const auto qkv = gdnQKVProfilePrecision(gdn_qkv, flash_next, role, name, qfn_profile);
        const auto fa = faOutputPrecision(fa_output, flash_next, role, name, main_layers);
        const auto gdn = gdnOutputPrecision(gdn_output, flash_next, role, name);
        return Qwen4ExpHCUpWorkspace::selected(hc_up, flash_next, role, name) ||
            gdn == QuantizedActivationOperandPrecision::TwoTermQ8 ||
            fa == QuantizedActivationOperandPrecision::TwoTermQ8 ||
            qkv == QuantizedActivationOperandPrecision::TwoTermQ8;
    }

    /**
     * @brief Reject unqualified request-hole or dense-TP geometry at setup.
     * @param policy Frozen GDN-output policy.
     * @param flash_next Authenticated QFN model family.
     * @param role Original canonical source role.
     * @param name Exact original tensor name.
     * @param requests Simultaneously admitted request prefixes, distinct from token rows.
     * @param slice Actual prepared source interval, independent of overlay/PP device count.
     *
     * One request supports every positive admitted row bucket and its existing
     * counted prefix, including zero live rows. Multiple independent request
     * holes need a quantized request-row contract, which the floating-only
     * request scope cannot provide. Dense-TP fold/allreduce equivalence has not
     * been qualified for this diagnostic role and is rejected explicitly.
     */
    static void validateGDNOutputInvocation(QuantizedActivationOperandPrecision policy,
        bool flash_next, WeightRole role, std::string_view name,
        int requests, const WeightSliceSpec &slice)
    {
        if (gdnOutputPrecision(policy, flash_next, role, name) !=
            QuantizedActivationOperandPrecision::TwoTermQ8)
            return;
        if (requests != 1)
            throw std::invalid_argument("QFN GDN-output TwoTermQ8 requires one request; quantized counted request holes are unsupported");
        if (!slice.source_rows || !slice.source_cols || slice.row_start || slice.col_start ||
            slice.row_count != slice.source_rows || slice.col_count != slice.source_cols)
            throw std::invalid_argument("QFN GDN-output TwoTermQ8 requires its complete original dense source interval; TP arithmetic is unqualified");
    }

    /**
     * @brief Obtain the existing operation's exact source-owned workspace contract.
     * @param device Exact participant backend; no context is queried.
     * @param hc_up Frozen independent HC-up policy.
     * @param gdn_output Frozen independent GDN-output policy.
     * @param flash_next Authenticated QFN model family.
     * @param role Original canonical source role.
     * @param name Exact source tensor name.
     * @param format Original GGUF format; selected quantized operations retain its native codebook.
     * @param m Admitted physical token rows, independent of live prefix count.
     * @param n Participant-local output width.
     * @param k Participant-local complete K32 reduction width.
     * @param fa_output Frozen main-only FA policy, independent of existing selections.
     * @param main_layers Original main layer count; source48 predictor is never selected.
     * @param gdn_qkv Independent original-main GDN-QKV precision.
     * @param qfn_profile Call-only borrow of the owning immutable planning profile.
     * @return Empty ordinary contribution or canonical full activation banks.
     * @throws std::invalid_argument For unsupported source, device or geometry.
     */
    static WorkspaceRequirements requirements(DeviceId device,
        QuantizedActivationOperandPrecision hc_up,
        QuantizedActivationOperandPrecision gdn_output, bool flash_next,
        WeightRole role, std::string_view name, std::string_view format,
        int m, int n, int k,
        QuantizedActivationOperandPrecision fa_output = QuantizedActivationOperandPrecision::SingleQ8,
        int main_layers = 0,
        QuantizedActivationOperandPrecision gdn_qkv = QuantizedActivationOperandPrecision::SingleQ8,
        const ModelMemoryProfile *qfn_profile = nullptr)
    {
        if (!selected(hc_up, gdn_output, flash_next, role, name, fa_output, main_layers, gdn_qkv, qfn_profile))
            return {};
        if (Qwen4ExpHCUpWorkspace::selected(hc_up,flash_next,role,name))
            return Qwen4ExpHCUpWorkspace::requirements(device,hc_up,flash_next,role,name,format,m,n,k);
        if (gdnQKVProfilePrecision(gdn_qkv, flash_next, role, name, qfn_profile) ==
            QuantizedActivationOperandPrecision::TwoTermQ8)
        {
            if (qfn_profile->gdn_group_count <= 0 || qfn_profile->gdn_state_size <= 0 ||
                qfn_profile->gdn_inner_size <= 0 || qfn_profile->d_model <= 0)
                throw std::invalid_argument("QFN original mixed-QKV workspace geometry is incomplete");
            const uint64_t original_rows = originalQKVRows(qfn_profile->gdn_group_count,
                qfn_profile->gdn_state_size, qfn_profile->gdn_inner_size);
            if (n <= 0 || k <= 0 || static_cast<uint64_t>(n) != original_rows || k != qfn_profile->d_model)
                throw std::invalid_argument("QFN GDN-QKV workspace requires the whole original native matrix");
        }
        (void)requireQ8ResidualSourceFormat(format,m,n,k);
        if (device.is_cpu())
            return cpuQ8ResidualCorrectionWorkspaceRequirements(m, n, k);
        if (device.is_cuda() || device.is_rocm())
            return gpuQ8ResidualCorrectionWorkspaceRequirements(m, n, k);
        throw std::invalid_argument("QFN activation correction requires a supported physical participant");
    }
private:
    /**
     * @brief Parse only canonical original mixed-QKV names; malformed names never identify another source.
     * @param name Original GGUF tensor name.
     * @return Exact unsigned global layer or no canonical source match.
     */
    static std::optional<uint32_t> originalQKVLayer(std::string_view name)
    {
        constexpr std::string_view suffix = ".attn_qkv.weight";
        if (!name.starts_with("blk.") || !name.ends_with(suffix)) return std::nullopt;
        const auto digits = name.substr(4, name.size() - 4 - suffix.size());
        if (digits.empty() || (digits.size() > 1 && digits.front() == '0') ||
            digits.find_first_not_of("0123456789") != std::string_view::npos) return std::nullopt;
        uint32_t layer = 0;
        const auto parsed = std::from_chars(digits.data(), digits.data() + digits.size(), layer);
        return parsed.ec == std::errc{} && parsed.ptr == digits.data() + digits.size()
            ? std::optional<uint32_t>(layer) : std::nullopt;
    }
    /**
     * @brief Check the sole original mixed-QKV extent projection from either existing source owner.
     * @param key_heads Original complete key-head count.
     * @param head_dimension Original key projection width per head.
     * @param value_width Original complete value projection width.
     * @return Two complete key projections followed by the value projection.
     * @throws std::overflow_error If the complete source extent is not representable.
     */
    static uint64_t originalQKVRows(uint32_t key_heads, uint32_t head_dimension, uint32_t value_width)
    {
        const uint64_t doubled_dimension = uint64_t{2} * head_dimension;
        if (doubled_dimension && key_heads >
            (std::numeric_limits<uint64_t>::max() - value_width) / doubled_dimension)
            throw std::overflow_error("QFN original mixed-QKV source geometry overflows");
        return doubled_dimension * key_heads + value_width;
    }
    /**
     * @brief Derive the original mixed-QKV output rows once from source-owned GDN geometry.
     * @param geometry Actual immutable model configuration, not participant-local head counts.
     * @return Two key projections followed by the original complete value projection.
     * @throws std::overflow_error If source multiplication/addition cannot be represented.
     */
    static uint64_t gdnQKVOriginalRows(const Qwen4ExpConfig &geometry)
    {
        const auto &gdn = geometry.deltaNet();
        return originalQKVRows(gdn.key_heads, gdn.head_dimension, gdn.value_width);
    }
};
} // namespace llaminar2
