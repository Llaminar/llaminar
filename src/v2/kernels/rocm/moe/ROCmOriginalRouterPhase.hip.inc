/**
 * @file ROCmOriginalRouterPhase.hip.inc
 * @brief One original FP16-router arithmetic and live-runtime publication authority.
 *
 * Standalone kernels and cooperative callers share indexed logical128 expert
 * dots and the original physical256 selector. Caller-owned shared planes
 * replace only storage ownership. Existing ABI-asserted runtime views remain
 * the authority; templates create no descriptor mirror, allocation or ledger.
 * Ordered MoE wave arithmetic explicitly requires the existing wave64 build
 * contract. Projection still derives its logical team wave count from warpSize.
 */
#pragma once
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cfloat>
#include <type_traits>
#include <utility>
#include "ROCmMoEWaveReduction.h"
#include "../common/ROCmWorkgroupJoin.h"
#include "execution/moe/DeviceMoERuntimeABI.h"
#include "execution/moe/DeviceMoERebalancePolicyShared.h"
#include "execution/moe/DeviceMoEOverlayEpochABI.h"
#include "kernels/common/MoEIntermediateShardRowExecution.h"
#include "kernels/common/NativeQuantisedDescriptorValidation.h"
#include "kernels/common/MoEDecodeDescriptorPublication.h"
#include "execution/config/RoutedExpertPolicy.h"
namespace llaminar2::rocm::grouped_ffn { class AcquiredGroupedRoutePublication; }
namespace llaminar2::rocm::router_native {
inline constexpr int kDeviceMoEMaxExperts=moe_runtime_abi::kMaxExperts;
inline constexpr int kDeviceMoEMaxTopK=16;
inline constexpr uint32_t kInvalidRuntimeExecutionBank=0xffffffffu;
using moe_wave_contract::rocm_moe_wave_reduce_sum_f32;
using moe_wave_contract::rocm_moe_wave_reduce_max_f32;
/** @brief Type of the existing runtime's placement bank, without another layout declaration. */
// This is an unevaluated member type: avoid std::declval's host-only declaration
// when HIP instantiates the shared router body inside another device entry.
template<class RuntimeView>using RouterBankView=std::remove_cvref_t<decltype(static_cast<RuntimeView*>(nullptr)->banks[0])>;
/** @brief Original terminal guard; invalid state never publishes a replacement route. */
__device__ __forceinline__ void fail_fast_incomplete_llep_transfer(){__builtin_trap();}
#define LLAMINAR_ROUTER_PHASE_FAIL(INVARIANT) do{(void)sizeof(INVARIANT);fail_fast_incomplete_llep_transfer();}while(false)
/** @brief Shared original `runtime_durable_placement_banks` policy; arguments retain their source-owned meanings. */
template<class RuntimeView>
__device__ __forceinline__ const RouterBankView<RuntimeView> *
    runtime_durable_placement_banks(const RuntimeView *runtime)
    {
        return runtime && runtime->overlay_placement_banks
                   ? runtime->overlay_placement_banks
                   : (runtime ? runtime->banks : nullptr);
    }
/** @brief Shared original `runtime_placement_banks` policy; arguments retain their source-owned meanings. */
template<class RuntimeView>
__device__ __forceinline__ const RouterBankView<RuntimeView> *
    runtime_placement_banks(const RuntimeView *runtime)
    {
        if (!runtime)
            return nullptr;
        return runtime->current_batch_llep_transient_bank_active != 0u
                   ? runtime->banks
                   : ::llaminar2::rocm::router_native::runtime_durable_placement_banks(runtime);
    }
/** @brief Shared original `runtime_execution_bank` policy; arguments retain their source-owned meanings. */
template<class RuntimeView>
__device__ __forceinline__ uint32_t runtime_execution_bank(
        const RuntimeView *runtime)
    {
        if (!runtime)
            return kInvalidRuntimeExecutionBank;
        if (!runtime->overlay_epoch_ticket)
        {
            return runtime->active_bank <= 1u
                       ? runtime->active_bank
                       : kInvalidRuntimeExecutionBank;
        }

        const uint64_t epoch = runtime->overlay_epoch_ticket->epoch;
        const uint64_t selector = runtime->overlay_epoch_ticket->selector;
        const uint32_t bank = static_cast<uint32_t>(selector & 1u);
        const uint64_t generation = selector >> 1u;
        const RouterBankView<RuntimeView> *placement_banks =
            ::llaminar2::rocm::router_native::runtime_durable_placement_banks(runtime);
        if (epoch == 0u || generation == 0u || bank > 1u ||
            !placement_banks ||
            static_cast<uint64_t>(placement_banks[bank].epoch) != epoch)
        {
            return kInvalidRuntimeExecutionBank;
        }
        if (runtime->current_batch_llep_transient_bank_active != 0u)
        {
            const uint32_t transient_bank = runtime->active_bank;
            if (!runtime->overlay_placement_banks || transient_bank > 1u ||
                static_cast<uint64_t>(runtime->banks[transient_bank].epoch) !=
                    epoch)
            {
                return kInvalidRuntimeExecutionBank;
            }
            return transient_bank;
        }
        return bank;
    }
/** @brief Shared original `runtime_selected_histogram` policy; arguments retain their source-owned meanings. */
template<class RuntimeView>
__device__ __forceinline__ uint64_t *runtime_selected_histogram(
        RuntimeView &runtime,
        uint32_t source)
    {
        if (runtime.runtime_histogram_banks ||
            runtime.runtime_histogram_active_bank)
        {
            if (!runtime.runtime_histogram_banks ||
                !runtime.runtime_histogram_active_bank ||
                source >= llaminar2::moe_runtime_abi::kHistogramSourceCount)
            {
                __builtin_trap();
            }
            const uint32_t state = *runtime.runtime_histogram_active_bank;
            if ((state &
                 ~llaminar2::moe_runtime_abi::kHistogramWriterStateMask) != 0u)
                __builtin_trap();
            const uint32_t bank =
                state &
                llaminar2::moe_runtime_abi::kHistogramWriterBankMask;
            return runtime.runtime_histogram_banks[bank].selected[source];
        }
        switch (source)
        {
        case 0u:
            return runtime.decode_histogram;
        case 1u:
            return runtime.prefill_histogram;
        case 2u:
            return runtime.grouped_verifier_histogram;
        default:
            __builtin_trap();
            return nullptr;
        }
    }
/** @brief Shared original `runtime_local_histogram` policy; arguments retain their source-owned meanings. */
template<class RuntimeView>
__device__ __forceinline__ uint64_t *runtime_local_histogram(
        RuntimeView &runtime,
        uint32_t source)
    {
        if (runtime.runtime_histogram_banks ||
            runtime.runtime_histogram_active_bank)
        {
            if (!runtime.runtime_histogram_banks ||
                !runtime.runtime_histogram_active_bank ||
                source >= llaminar2::moe_runtime_abi::kHistogramSourceCount)
            {
                __builtin_trap();
            }
            const uint32_t state = *runtime.runtime_histogram_active_bank;
            if ((state &
                 ~llaminar2::moe_runtime_abi::kHistogramWriterStateMask) != 0u)
                __builtin_trap();
            const uint32_t bank =
                state &
                llaminar2::moe_runtime_abi::kHistogramWriterBankMask;
            return runtime.runtime_histogram_banks[bank].local[source];
        }
        switch (source)
        {
        case 0u:
            return runtime.decode_local_histogram;
        case 1u:
            return runtime.prefill_local_histogram;
        case 2u:
            return runtime.grouped_verifier_local_histogram;
        default:
            __builtin_trap();
            return nullptr;
        }
    }
/** @brief Shared original `runtime_histogram_admits_rows` policy; arguments retain their source-owned meanings. */
template<class RuntimeView>
__device__ __forceinline__ bool runtime_histogram_admits_rows(
        const RuntimeView &runtime)
    {
        if (!runtime.runtime_histogram_active_bank)
            return true;
        const uint32_t state = *runtime.runtime_histogram_active_bank;
        if ((state &
             ~llaminar2::moe_runtime_abi::kHistogramWriterStateMask) != 0u)
        {
            __builtin_trap();
        }
        return (state &
                llaminar2::moe_runtime_abi::kHistogramWriterQuarantineBit) ==
               0u;
    }
/**
 * @brief The selected and local decode destinations of the same live histogram writer.
 * @details Null destinations indicate the original quarantined state. The
 * registered producer stream prevents a writer rotation during this operation.
 */
struct RuntimeDecodeHistogramWriter {
    uint64_t* selected=nullptr;
    uint64_t* local=nullptr;
};
/**
 * @brief Borrow the original decode histogram generation once for a publisher slot.
 * @param runtime Live original runtime, ordered on its exact producer stream.
 * @return Its admitted selected/local pair, or the original quarantined state.
 */
template<class RuntimeView>
__device__ __forceinline__ RuntimeDecodeHistogramWriter bindRuntimeDecodeHistogramWriter(RuntimeView& runtime)
{
    auto* banks=runtime.runtime_histogram_banks;
    auto* state_ptr=runtime.runtime_histogram_active_bank;
    if(!banks&&!state_ptr)
        return {runtime.decode_histogram,runtime.decode_local_histogram};
    if(!banks||!state_ptr)
        fail_fast_incomplete_llep_transfer();
    const uint32_t state=*state_ptr;
    if(!llaminar2::moe_runtime_abi::validHistogramWriterState(state))
        fail_fast_incomplete_llep_transfer();
    if((state&llaminar2::moe_runtime_abi::kHistogramWriterQuarantineBit)!=0u)
        return {};
    auto& writer=banks[state&llaminar2::moe_runtime_abi::kHistogramWriterBankMask];
    return {writer.selected[0],writer.local[0]};
}
/** @brief Shared original `runtime_require_decode_execution_bank` policy; arguments retain their source-owned meanings. */
template<class RuntimeView>
__device__ __forceinline__ uint32_t
    runtime_require_decode_execution_bank(
        const RuntimeView *runtime)
    {
        const uint32_t execution_bank = ::llaminar2::rocm::router_native::runtime_execution_bank(runtime);
        if (execution_bank <= 1u)
            return execution_bank;

        // No device printf: it would bind a hostcall buffer to every router
        // kernel and fail their launch on hosts without native PCIe atomics.
        // The ticket and bank epochs stay readable in device memory.
        __builtin_verbose_trap("Llaminar MoE", "decode runtime has no valid overlay execution bank");
    }
/** @brief Shared original `runtime_route_assignment_ledger_ok` policy; arguments retain their source-owned meanings. */
template<class RuntimeView>
__device__ __forceinline__ bool runtime_route_assignment_ledger_ok(
        const RuntimeView *runtime,
        int top_k)
    {
        return runtime && top_k > 0 && runtime->route_participant_ids &&
               runtime->prefill_route_capacity >=
                   static_cast<std::uint32_t>(top_k);
    }
/** @brief Shared original `runtime_expert_owner` policy; arguments retain their source-owned meanings. */
template<class BankView>
__device__ __forceinline__ int runtime_expert_owner(
        const BankView &bank,
        int expert_id)
    {
        return bank.experts[expert_id].owner_participant;
    }
/** @brief Shared original `runtime_participant_bit` policy; arguments retain their source-owned meanings. */
__device__ __forceinline__ uint32_t runtime_participant_bit(int participant)
    {
        return llaminar2::moe_rebalance_policy::participantBit(
            static_cast<uint32_t>(participant));
    }
/** @brief Shared original `runtime_multi_resident_expert_count` policy; arguments retain their source-owned meanings. */
template<class BankView>
__device__ __forceinline__ uint32_t runtime_multi_resident_expert_count(
        const BankView &bank)
    {
        return bank.multi_resident_expert_count;
    }
/** @brief Shared original `runtime_intermediate_shard_row` policy; arguments retain their source-owned meanings. */
template<class RuntimeView>
__device__ __forceinline__ llaminar2::MoEIntermediateShardRowSelection
    runtime_intermediate_shard_row(const RuntimeView *runtime,
        const RouterBankView<RuntimeView> &bank, int expert)
    {
        const auto selection = llaminar2::selectMoEIntermediateShardRow(
            runtime->intermediate_shard, bank, expert, static_cast<int>(runtime->participant_id));
        if (selection.disposition == llaminar2::MoEIntermediateShardRowDisposition::Invalid)
        {
            fail_fast_incomplete_llep_transfer();
        }
        return selection;
    }
/** @brief Shared original `runtime_decode_row_active` policy; arguments retain their source-owned meanings. */
__device__ __forceinline__ bool runtime_decode_row_active(const int32_t *active_row_count_device)
    {
        const int32_t count = active_row_count_device ? *active_row_count_device : 1;
        if (count < 0 || count > 1)
        {
            LLAMINAR_ROUTER_PHASE_FAIL("one-row MoE routing received a live count outside [0,1]");
            return false;
        }
        return count == 1;
    }
/** @brief Shared original `runtime_require_decode_row_policy` policy; arguments retain their source-owned meanings. */
template<class RuntimeView>
__device__ __forceinline__ void runtime_require_decode_row_policy(
        const RuntimeView *runtime,
        llaminar2::RoutedExpertRowExecutionPolicy row_execution_policy)
    {
        if ((row_execution_policy != llaminar2::RoutedExpertRowExecutionPolicy::ParticipantAssigned &&
             row_execution_policy != llaminar2::RoutedExpertRowExecutionPolicy::FullyReplicatedLocal &&
             row_execution_policy != llaminar2::RoutedExpertRowExecutionPolicy::IntermediateShardLocal) ||
            runtime->intermediate_shard.enabled() !=
                (row_execution_policy == llaminar2::RoutedExpertRowExecutionPolicy::IntermediateShardLocal) ||
            (runtime->intermediate_shard.enabled() && runtime->current_batch_llep_transient_bank_active != 0u))
        {
            fail_fast_incomplete_llep_transfer();
            return;
        }
    }
/** @brief CTA-local storage for a token issued by the original runtime bank authority. */
template<class RuntimeView>class RuntimeDecodeExecutionBankStorage;
/**
 * @brief Issued view of one immutable request-acquired execution bank.
 * @tparam RuntimeView Original ABI-authenticated device runtime.
 * @details Only acquire() constructs a token. It retains the original runtime
 * and exact validated bank address, not a descriptor mirror, new epoch or placement
 * ledger. The existing graph epoch boundary keeps that source immutable until
 * kernel retirement. Copies preserve the same issued authority.
 */
template<class RuntimeView>class AcquiredRuntimeDecodeBank final
{
    friend class RuntimeDecodeExecutionBankStorage<RuntimeView>;
    friend class ::llaminar2::rocm::grouped_ffn::AcquiredGroupedRoutePublication;

    /**
     * @brief Borrow Down for a slot from the completed original route publisher.
     * @param slot Original live top-k coordinate in this request's publication.
     * @param liveSlots Actual counted live route extent, checked by the consumer.
     * @return Empty for a remote row, or the exact ready native local role.
     * @pre The sole grouped-route borrower has verified the existing published
     * bank/epoch/ticket after the captured producer stream edge.
     */
    __device__ __forceinline__ PublishedNativeProjection localNativeDownFromGroupedSlot(
        int slot,int liveSlots) const
    {
        if(slot<0 || slot>=liveSlots)fail_fast_incomplete_llep_transfer();
        const auto participant=static_cast<int>(runtime_->participant_id);
        if(__builtin_amdgcn_readfirstlane(runtime_->route_participant_ids[slot])!=participant)
            return PublishedNativeProjection(nullptr);
        const int expert=__builtin_amdgcn_readfirstlane(runtime_->route_expert_ids[slot]);
        if(expert<0 || static_cast<unsigned>(expert)>=bank_->expert_count)
            fail_fast_incomplete_llep_transfer();
        const auto& row=bank_->experts[expert];
        if(row.projection_set!=DeviceMoEProjectionSet::IntermediateShard ||
           row.weight_format!=DeviceMoEWeightFormat::NativeQuantised)
            fail_fast_incomplete_llep_transfer();
        return PublishedNativeProjection(&row.down);
    }

    RuntimeView* runtime_;
    const RouterBankView<RuntimeView>* bank_;
    /** @brief Construct only from the original canonical validated placement, or inside typed storage copy. */
    __device__ AcquiredRuntimeDecodeBank(RuntimeView* runtime,const RouterBankView<RuntimeView>* bank)
        :runtime_(runtime),bank_(bank){}
public:
    using Runtime=RuntimeView; ///< Original ABI type for typed runtime consumers.
    /** @brief Copy the same issued immutable view without acquiring another epoch. */
    __device__ AcquiredRuntimeDecodeBank(const AcquiredRuntimeDecodeBank&)=default;
    /**
     * @brief Validate once through the original precise fatal diagnostic path.
     * @param runtime Original device state, including its request-acquired ticket.
     * @return Issued view; invalid epochs print the unchanged diagnostic and trap.
     */
    __device__ __forceinline__ static AcquiredRuntimeDecodeBank acquire(RuntimeView* runtime)
    {
        // Canonical invalid-epoch diagnostics and trap precede token bank-base
        // resolution. The original acquired epoch fixes this placement for
        // the entire invocation; peers borrow the same issued address.
        const uint32_t index=runtime_require_decode_execution_bank(runtime);
        return {runtime,runtime_placement_banks(runtime)+index};
    }
    /**
     * @brief Resolve original membership from this same producer-validated bank.
     * @param expert Actual original selected ID, never a substituted local slot.
     * @return Original Local/Remote selection and participant; broken ownership traps.
     * @details The existing ready-bank producers validate all reachable local
     * descriptors before bank publication. This method does not admit a raw
     * descriptor or a new epoch; it retains the token's original bank/lease.
     */
    __device__ __forceinline__ MoEIntermediateShardRowSelection intermediateShardMembership(int expert) const
    {
        const auto membership=selectMoEIntermediateShardMembership(runtime_->intermediate_shard,
            *bank_,expert,static_cast<int>(runtime_->participant_id));
        if(membership.disposition==MoEIntermediateShardRowDisposition::Invalid)
            fail_fast_incomplete_llep_transfer();
        return membership;
    }
    /**
     * @brief Read canonical locality from an acquired original fragment bank.
     * @param expert Original selected expert, bounded before reading the bank.
     * @return Local with its existing runtime member coordinate, Remote with no
     * local owner, or Invalid for an out-of-range expert.
     * @pre The existing epoch producer validated the complete original rows,
     * including remote target admissibility and equality of local-compute mask
     * and original cohort membership, before publishing this immutable bank.
     * @details This opt-in reads the existing producer-owned locality mask. It
     * does not inspect a raw descriptor or create a second ownership table.
     */
    __device__ __forceinline__ MoEIntermediateShardRowSelection
    publishedOriginalFragmentMembership(int expert) const
    {
        const int participant=static_cast<int>(runtime_->participant_id);
        if(!runtime_->intermediate_shard.enabled()||participant<0||
           runtime_->current_batch_llep_transient_bank_active!=0u)
            fail_fast_incomplete_llep_transfer();
        if(expert<0||static_cast<unsigned>(expert)>=bank_->expert_count)
            return {};
        return bank_->local_compute_mask[expert]
            ? MoEIntermediateShardRowSelection{MoEIntermediateShardRowDisposition::Local,participant}
            : MoEIntermediateShardRowSelection{MoEIntermediateShardRowDisposition::Remote,-1};
    }
    /**
     * @brief Issue a local projection from this SAME acquired, producer-validated row.
     * @param expert Actual device-selected expert, bounded before row access.
     * @param role Semantic gate/up/down role; no raw pointer can enter this API.
     * @return Empty for a legitimate remote/non-native route; broken membership traps.
     * @details The published ready-row contract owns immutable shape/source/plane
     * validity. Actual route target, masks, flags, slot and epoch remain live.
     */
    __device__ __forceinline__ PublishedNativeProjection localNativeProjection(
        int expert,PublishedNativeProjectionRole role) const
    {
        const auto membership=intermediateShardMembership(expert);
        if(membership.disposition!=MoEIntermediateShardRowDisposition::Local)
            return PublishedNativeProjection(nullptr);
        const auto& row=bank_->experts[expert];
        if(row.projection_set!=DeviceMoEProjectionSet::IntermediateShard)
            fail_fast_incomplete_llep_transfer();
        if(row.weight_format!=DeviceMoEWeightFormat::NativeQuantised)
            return PublishedNativeProjection(nullptr);
        switch(role) {
        case PublishedNativeProjectionRole::Gate:return PublishedNativeProjection(&row.gate);
        case PublishedNativeProjectionRole::Up:return PublishedNativeProjection(&row.up);
        case PublishedNativeProjectionRole::Down:return PublishedNativeProjection(&row.down);
        }
        __builtin_trap();
    }
    /**
     * @brief Preserve inactive/out-of-range route semantics without touching descriptor memory.
     * @param expert Actual selected expert or inactive sentinel. @param role Semantic projection role.
     * @return Empty for inactive/bounded-away routes, otherwise the original acquired local projection.
     */
    __device__ __forceinline__ PublishedNativeProjection localNativeProjectionOrInactive(
        int expert,PublishedNativeProjectionRole role) const
    {
        if(expert<0||static_cast<unsigned>(expert)>=bank_->expert_count)
            return PublishedNativeProjection(nullptr);
        return localNativeProjection(expert,role);
    }
    /** @brief Gate and up borrowed together from one checked original live row. */
    struct LocalNativeGateUp {
        PublishedNativeProjection gate;
        PublishedNativeProjection up;
        /** @return Whether this selected row is a ready local native expert. */
        __device__ bool local() const noexcept { return bool(gate); }
    };

    /**
     * @brief Resolve one live route and issue both local native roles together.
     * @param expert The original selected ID, or the original inactive sentinel.
     * @return Empty when out of range or remote, otherwise both exact local roles.
     * @details Reuses the normal current bank locality checks. Both projections
     * refer to the same ready row and are valid only in this acquired epoch.
     */
    __device__ __forceinline__ LocalNativeGateUp localNativeGateUpOrInactive(int expert) const
    {
        if(expert<0 || static_cast<unsigned>(expert)>=bank_->expert_count ||
           intermediateShardMembership(expert).disposition!=MoEIntermediateShardRowDisposition::Local)
            return {PublishedNativeProjection(nullptr),PublishedNativeProjection(nullptr)};
        const auto& row=bank_->experts[expert];
        if(row.projection_set!=DeviceMoEProjectionSet::IntermediateShard ||
           row.weight_format!=DeviceMoEWeightFormat::NativeQuantised)
            fail_fast_incomplete_llep_transfer();
        return {PublishedNativeProjection(&row.gate),PublishedNativeProjection(&row.up)};
    }
    /** @return Original runtime whose immutable acquired epoch issued this view. */
    __device__ __forceinline__ RuntimeView* runtime() const {return runtime_;}
    /** @return Original immutable placement bank, not a copied descriptor table. */
    __device__ __forceinline__ const RouterBankView<RuntimeView>& bank() const
    {return *bank_;}
};
/**
 * @brief Preserve original runtime shape checks after canonical token acquisition.
 * @param acquired Original issued bank; construction already rejects index>1.
 * @param num_experts Original complete expert extent. @param top_k Original selected width.
 * @return Original runtime active-epoch and shape predicate, without reselecting placement.
 * @details The former redundant index predicate was always true for an issued
 * token. Invalid acquisition still prints the precise canonical diagnostic and
 * traps before this helper or any bank-address resolution can execute.
 */
template<class RuntimeView>
__device__ __forceinline__ bool runtime_shape_ok(
    const AcquiredRuntimeDecodeBank<RuntimeView>& acquired,int num_experts,int top_k)
{
    const auto* runtime=acquired.runtime();
    return runtime && runtime->active_epoch!=0u &&
        runtime->expert_count==static_cast<uint32_t>(num_experts) &&
        runtime->top_k==static_cast<uint32_t>(top_k);
}
/**
 * @brief Bind the original acquired lease to its validated fragment dispatch shape.
 * @details Runtime policy, participant and extent are immutable while this lease is live.
 * Individual route IDs are still bounded before accessing the original locality mask.
 */
template<class RuntimeView,unsigned Experts>
class ValidatedOriginalFragmentBank final {
    using Token=AcquiredRuntimeDecodeBank<RuntimeView>;
    Token original_;
    int participant_;
    int top_k_;

    /** @brief Construct from the original lease and validated fragment participant. */
    __device__ explicit ValidatedOriginalFragmentBank(Token token,int participant,int top_k)
        :original_(token),participant_(participant),top_k_(top_k){}
public:
    /** @brief Validate immutable active shape and policy before the selector.
     * @param token Original epoch token from checked acquisition.
     * @param top_k Original admitted positive route width.
     * @return A fragment-scoped view of the same acquired bank, without publication.
     */
    __device__ __forceinline__ static ValidatedOriginalFragmentBank bind(Token token,int top_k) {
        static_assert(Experts>0 && Experts<=kDeviceMoEMaxExperts);
        auto* runtime=token.runtime();
        const int participant=static_cast<int>(runtime->participant_id);
        if(top_k<=0 || top_k>kDeviceMoEMaxTopK || participant<0 ||
           token.bank().expert_count!=Experts ||
           !runtime_shape_ok(token,Experts,top_k))
            fail_fast_incomplete_llep_transfer();
        runtime_require_decode_row_policy(runtime,
            llaminar2::RoutedExpertRowExecutionPolicy::IntermediateShardLocal);
        return ValidatedOriginalFragmentBank(token,participant,top_k);
    }
    /** @return The same original lease issued before binding. */
    __device__ __forceinline__ Token original() const {return original_;}
    /** @return Exact selected width proved during this binding. */
    __device__ __forceinline__ int topK() const {return top_k_;}
    /** @param expert Original selected expert ID. @return Participant-local membership or Invalid. */
    __device__ __forceinline__ MoEIntermediateShardRowSelection select(int expert) const {
        if(static_cast<unsigned>(expert)>=Experts)
            return {};
        return original_.bank().local_compute_mask[expert]
            ? MoEIntermediateShardRowSelection{MoEIntermediateShardRowDisposition::Local,participant_}
            : MoEIntermediateShardRowSelection{MoEIntermediateShardRowDisposition::Remote,-1};
    }
};
/**
 * @brief Publish one issued token to peer lanes using caller-owned existing LDS.
 * @tparam RuntimeView Original ABI-authenticated runtime.
 * @details A single CTA leader publishes before an existing local-memory join.
 * Borrowers run only after that edge. Storage remains live until all phase
 * consumers retire; private bytes cannot be replaced through an integer API.
 */
template<class RuntimeView>class RuntimeDecodeExecutionBankStorage final
{
    using Token=AcquiredRuntimeDecodeBank<RuntimeView>;
    alignas(Token) unsigned char bytes_[sizeof(Token)];
public:
    /** @brief Trivial storage construction performs no device work or initialization. */
    RuntimeDecodeExecutionBankStorage()=default;
    /** @param token Issued original view. Publish its exact trivially-copyable object representation. */
    __device__ __forceinline__ void publish(Token token)
    {static_assert(std::is_trivially_copyable_v<Token> && sizeof(Token)==2*sizeof(void*));__builtin_memcpy(bytes_,&token,sizeof(token));}
    /** @return Same issued view after the caller's mandatory LDS publication edge. */
    __device__ __forceinline__ Token borrow() const
    {Token result(nullptr,nullptr);__builtin_memcpy(&result,bytes_,sizeof(result));return result;}
};
/** @brief Select original membership validation supported by the caller and producer. */
enum class RuntimeDecodePublicationValidation : unsigned char {
    Checked,
    ProducerCertified,
    ValidatedOriginalFragment,
    BoundOriginalFragment
};
/** @brief Publish the original selected descriptor packet through its live acquired bank.
 * @tparam UpdateRuntimeHistogram Original integer demand policy.
 * @tparam RowPolicy Original owner or fragment selection policy.
 * @tparam PublishDescriptors Original public table publication policy.
 * @tparam RuntimeView Existing complete ABI-asserted device runtime view.
 */
template<bool UpdateRuntimeHistogram,
 llaminar2::RoutedExpertRowExecutionPolicy RowPolicy=llaminar2::RoutedExpertRowExecutionPolicy::ParticipantAssigned,
 bool PublishDescriptors=false,class RuntimeView,
 RuntimeDecodePublicationValidation Validation=RuntimeDecodePublicationValidation::Checked,
 unsigned FragmentExperts=0,bool BindHistogramWriter=false>
__device__ __forceinline__ bool
    runtime_publish_independent_decode_dispatch_from_issued_source(
        std::conditional_t<FragmentExperts!=0,ValidatedOriginalFragmentBank<RuntimeView,FragmentExperts>,
                           AcquiredRuntimeDecodeBank<RuntimeView>> source,
        const int *selected_ids,
        const float *selected_weights,
        int num_experts,
        int top_k,
        float *legacy_indices,
        float *legacy_weights,
        bool write_legacy_outputs,
        const int32_t *active_row_count_device,
        llaminar2::MoEDecodeDescriptorPublication publication)
    {
        static_assert(Validation==RuntimeDecodePublicationValidation::Checked||
            Validation==RuntimeDecodePublicationValidation::ProducerCertified||
            Validation==RuntimeDecodePublicationValidation::ValidatedOriginalFragment||
            Validation==RuntimeDecodePublicationValidation::BoundOriginalFragment);
        static_assert((Validation==RuntimeDecodePublicationValidation::BoundOriginalFragment)==
                      (FragmentExperts>0));
        static_assert(Validation==RuntimeDecodePublicationValidation::Checked||
            RowPolicy==llaminar2::RoutedExpertRowExecutionPolicy::IntermediateShardLocal);
        const auto acquired=[&] {
            if constexpr(FragmentExperts!=0)return source.original();
            else return source;
        }();
        auto* runtime=acquired.runtime();
        static_assert(!BindHistogramWriter||
            (UpdateRuntimeHistogram&&Validation==RuntimeDecodePublicationValidation::BoundOriginalFragment));
        static_assert(RowPolicy == llaminar2::RoutedExpertRowExecutionPolicy::ParticipantAssigned ||
                      RowPolicy == llaminar2::RoutedExpertRowExecutionPolicy::IntermediateShardLocal);
        const int slot = static_cast<int>(threadIdx.x);
        if (slot >= top_k)
            return false;
        const bool shape_ok = [&] {
            if constexpr(Validation==RuntimeDecodePublicationValidation::BoundOriginalFragment)
                return true;
            else return ::llaminar2::rocm::router_native::runtime_shape_ok(acquired,num_experts,top_k);
        }();
        if (!::llaminar2::rocm::router_native::runtime_route_assignment_ledger_ok(runtime, top_k))
        {
            LLAMINAR_ROUTER_PHASE_FAIL(
                "unique-owner decode route ledger is absent or undersized");
            return false;
        }

        const auto &bank = acquired.bank();
        if constexpr (RowPolicy == llaminar2::RoutedExpertRowExecutionPolicy::ParticipantAssigned)
        {
            if (shape_ok && ::llaminar2::rocm::router_native::runtime_multi_resident_expert_count(bank) != 0u)
            {
                LLAMINAR_ROUTER_PHASE_FAIL(
                    "unique-owner decode graph observed a replicated expert");
                return false;
            }
        }

        if constexpr(Validation!=RuntimeDecodePublicationValidation::BoundOriginalFragment)
            ::llaminar2::rocm::router_native::runtime_require_decode_row_policy(runtime, RowPolicy);
        const bool row_active = ::llaminar2::rocm::router_native::runtime_decode_row_active(active_row_count_device);
        const bool collect_histogram =
            UpdateRuntimeHistogram && !BindHistogramWriter &&
            ::llaminar2::rocm::router_native::runtime_histogram_admits_rows(*runtime);
        const int expert_id = row_active ? selected_ids[slot] : -1;
        const float weight = row_active ? selected_weights[slot] : 0.0f;
        bool local_compute = false;
        if (shape_ok)
        {
            int participant = -1;
            if constexpr (RowPolicy == llaminar2::RoutedExpertRowExecutionPolicy::IntermediateShardLocal)
            {
                if (expert_id != -1)
                {
                    MoEIntermediateShardRowSelection selection;
                    if constexpr(Validation==RuntimeDecodePublicationValidation::BoundOriginalFragment) {
                        selection=source.select(expert_id);
                        if(selection.disposition==MoEIntermediateShardRowDisposition::Invalid)
                            fail_fast_incomplete_llep_transfer();
                    } else if constexpr(Validation==RuntimeDecodePublicationValidation::ValidatedOriginalFragment) {
                        selection=acquired.publishedOriginalFragmentMembership(expert_id);
                        if(selection.disposition==MoEIntermediateShardRowDisposition::Invalid)
                            fail_fast_incomplete_llep_transfer();
                    } else if constexpr(Validation==RuntimeDecodePublicationValidation::ProducerCertified) {
                        selection=acquired.intermediateShardMembership(expert_id);
                    } else {
                        selection=::llaminar2::rocm::router_native::runtime_intermediate_shard_row(runtime,bank,expert_id);
                    }
                    local_compute = selection.disposition == llaminar2::MoEIntermediateShardRowDisposition::Local;
                    participant = selection.participant;
                }
            }
            else
            {
                const bool selected = expert_id >= 0 && expert_id < num_experts;
                local_compute = selected && bank.local_compute_mask[expert_id] != 0u;
                participant = selected ? ::llaminar2::rocm::router_native::runtime_expert_owner(bank, expert_id) : -1;
            }
            runtime->topk_expert_ids[slot] =
                local_compute ? expert_id : -1;
            runtime->topk_weights[slot] =
                local_compute ? weight : 0.0f;
            runtime->route_participant_ids[slot] =
                participant;
            if constexpr (PublishDescriptors)
            {
                if (local_compute)
                {
                    const auto &descriptor = bank.experts[expert_id];
                    bool ready = descriptor.weight_format == publication.format;
                    if constexpr (RowPolicy == llaminar2::RoutedExpertRowExecutionPolicy::ParticipantAssigned)
                        ready = ready && descriptor.local_slot >= 0 &&
                            (bank.resident_participant_mask[expert_id] & ::llaminar2::rocm::router_native::runtime_participant_bit(runtime->participant_id)) &&
                            llaminar2::deviceMoEExpertProjectionReady(descriptor, publication.projections);
                    llaminar2::publishMoEDecodeDescriptor(publication, descriptor, expert_id, ready);
                }
            }
        }

        if (write_legacy_outputs && legacy_indices && legacy_weights)
        {
            legacy_indices[slot] = static_cast<float>(expert_id);
            legacy_weights[slot] = weight;
        }

        if constexpr (UpdateRuntimeHistogram)
        {
            if constexpr(BindHistogramWriter) {
                if(expert_id>=0&&expert_id<num_experts) {
                    const auto writer=bindRuntimeDecodeHistogramWriter(*runtime);
                    if(writer.selected) {
                        atomicAdd(reinterpret_cast<unsigned long long*>(&writer.selected[expert_id]),1ULL);
                        if(local_compute)
                            atomicAdd(reinterpret_cast<unsigned long long*>(&writer.local[expert_id]),1ULL);
                    }
                }
            } else if (collect_histogram && expert_id >= 0 && expert_id < num_experts)
            {
                atomicAdd(
                    reinterpret_cast<unsigned long long *>(
                        &::llaminar2::rocm::router_native::runtime_selected_histogram(*runtime, 0u)[expert_id]),
                    1ULL);
                if (shape_ok && local_compute)
                {
                    atomicAdd(
                        reinterpret_cast<unsigned long long *>(
                            &::llaminar2::rocm::router_native::runtime_local_histogram(*runtime, 0u)[expert_id]),
                        1ULL);
                }
            }
        }
        return local_compute;
    }
/**
 * @brief Preserve the original checked or producer-certified publisher API for existing callers.
 */
template<bool UpdateRuntimeHistogram,
 llaminar2::RoutedExpertRowExecutionPolicy RowPolicy=llaminar2::RoutedExpertRowExecutionPolicy::ParticipantAssigned,
 bool PublishDescriptors=false,class RuntimeView,
 RuntimeDecodePublicationValidation Validation=RuntimeDecodePublicationValidation::Checked>
__device__ __forceinline__ void runtime_publish_independent_decode_dispatch_from_bank(
    AcquiredRuntimeDecodeBank<RuntimeView> acquired,const int* selected_ids,const float* selected_weights,
    int num_experts,int top_k,float* legacy_indices,float* legacy_weights,bool write_legacy_outputs,
    const int32_t* active_row_count_device,llaminar2::MoEDecodeDescriptorPublication publication)
{
    runtime_publish_independent_decode_dispatch_from_issued_source<
        UpdateRuntimeHistogram,RowPolicy,PublishDescriptors,RuntimeView,Validation>(
            acquired,selected_ids,selected_weights,num_experts,top_k,legacy_indices,legacy_weights,
            write_legacy_outputs,active_row_count_device,publication);
}
/**
 * @brief Publish one semantic fragment slot and return the same original local-compute result.
 * @details The sole typed input owns the original bank, geometry, and policy. The
 * bool is the exact local result. The native owner retains its selected descriptor output contract.
 */
template<bool PublishDescriptors,bool BindHistogramWriter,bool UpdateRuntimeHistogram=true,class RuntimeView,unsigned Experts>
__device__ __forceinline__ bool runtime_publish_bound_original_fragment_dispatch(
    ValidatedOriginalFragmentBank<RuntimeView,Experts> fragment,const int* selected_ids,
    const float* selected_weights,float* legacy_indices,float* legacy_weights,
    const int32_t* active_row_count_device,llaminar2::MoEDecodeDescriptorPublication publication)
{
    return runtime_publish_independent_decode_dispatch_from_issued_source<
        UpdateRuntimeHistogram,llaminar2::RoutedExpertRowExecutionPolicy::IntermediateShardLocal,PublishDescriptors,RuntimeView,
        RuntimeDecodePublicationValidation::BoundOriginalFragment,Experts,BindHistogramWriter>(
            fragment,selected_ids,selected_weights,Experts,fragment.topK(),legacy_indices,legacy_weights,
            true,active_row_count_device,publication);
}
/**
 * @brief Preserve original standalone acquisition before the single publication body.
 * @tparam UpdateRuntimeHistogram Original demand policy. @tparam RowPolicy Original route policy.
 * @tparam PublishDescriptors Original descriptor publication contract. @tparam RuntimeView Original runtime ABI.
 * @param runtime Original device state. @param selected_ids Original CTA-local selected IDs.
 * @param selected_weights Original CTA-local selected weights. @param num_experts Original expert extent.
 * @param top_k Original selected width. @param legacy_indices Original optional complete ID output.
 * @param legacy_weights Original optional weight output. @param write_legacy_outputs Original output policy.
 * @param active_row_count_device Original live0/1 authority. @param publication Original selected tables.
 */
template<bool UpdateRuntimeHistogram,
 llaminar2::RoutedExpertRowExecutionPolicy RowPolicy=llaminar2::RoutedExpertRowExecutionPolicy::ParticipantAssigned,
 bool PublishDescriptors=false,class RuntimeView>
__device__ __forceinline__ void runtime_publish_independent_decode_dispatch(
    RuntimeView* runtime,const int* selected_ids,const float* selected_weights,int num_experts,int top_k,
    float* legacy_indices,float* legacy_weights,bool write_legacy_outputs,
    const int32_t* active_row_count_device,llaminar2::MoEDecodeDescriptorPublication publication)
{
    if(static_cast<int>(threadIdx.x)>=top_k)return;
    runtime_publish_independent_decode_dispatch_from_bank<UpdateRuntimeHistogram,RowPolicy,PublishDescriptors>(
        AcquiredRuntimeDecodeBank<RuntimeView>::acquire(runtime),selected_ids,selected_weights,num_experts,top_k,
        legacy_indices,legacy_weights,write_legacy_outputs,active_row_count_device,publication);
}
/** @brief Explicit transfer implementation; indexed remains the default for every existing caller. */
enum class TopKTransferPolicy : std::uint8_t { Indexed, RowNativeRoot };
/**
 * @brief Move tournament bits without changing the comparator or ordered root graph.
 * @tparam Policy Original indexed moves or explicit row-native lane0-root specialization.
 * @tparam T Exact float/int32 word. @param value Current node bits.
 * @param offset Original tournament distance32,16,8,4,2,1. @return Source node bits.
 * @pre All64 lanes participate. Only lane0's final root is consumed or broadcast.
 * @details32/16 stay indexed. At each smaller level the lane0 dependency cone
 * is respectively lanes0..7,0..3,0..1,0; its source never crosses a16-lane row.
 * Other lanes' unused roots may differ. No comparator node or operand changes.
 */
template<TopKTransferPolicy Policy,class T>
__device__ __forceinline__ T topKDownBits(T value,int offset)
{
    if constexpr(Policy==TopKTransferPolicy::Indexed)return __shfl_down(value,offset);
    else {
        static_assert(Policy==TopKTransferPolicy::RowNativeRoot);
        using ::llaminar2::rocm::moe_wave_contract::rowShiftNativeBits;
        switch(offset) {
        case 8:return rowShiftNativeBits<8>(value);
        case 4:return rowShiftNativeBits<4>(value);
        case 2:return rowShiftNativeBits<2>(value);
        case 1:return rowShiftNativeBits<1>(value);
        default:return __shfl_down(value,offset); // Original32/16 stages.
        }
    }
}
/** @brief Original `moe_topk_pair_better` shared without changing comparisons or rounding. */
__device__ __forceinline__ bool moe_topk_pair_better(
        float candidate_value,
        int candidate_id,
        float best_value,
        int best_id)
    {
        return (candidate_value > best_value) |
               ((candidate_value == best_value) & (candidate_id < best_id));
    }
/** The monotone finite key and inverted ID together fit exactly in the mantissa/exponent
 * word of a positive, normal double. No floating conversion or numeric arithmetic
 * occurs. An input NaN remains an unordered, quiet double and preserves its original
 * FP32 bits in the otherwise opaque low word. Signed zero ties use the original ID.
 */
__device__ __forceinline__ uint64_t routeKey(float weight, unsigned id) {
    const unsigned u=__builtin_bit_cast(unsigned,weight);
    const unsigned magnitude=u&0x7fffffffu;
    const unsigned normalized=magnitude==0?0:u;
    const unsigned ordered=(normalized>>31)?~normalized:(normalized^0x80000000u);
    const unsigned high=magnitude>0x7f800000u?0x7ff80000u:0x20000000u+(ordered>>2);
    const unsigned low=magnitude>0x7f800000u?u:(ordered<<30)|(1023u-id);
    return (uint64_t(high)<<32)|low;
}
__device__ __forceinline__ bool keyBetter(uint64_t candidate,uint64_t old) {
    return __builtin_bit_cast(double,candidate)>__builtin_bit_cast(double,old);
}
__device__ __forceinline__ float packedQueue512(const float* probs,int experts,int extent,
    int* ids,float* weights){
    const unsigned lane=unsigned(threadIdx.x)%64;
    if(threadIdx.x>=64)return 0.0f;
    constexpr unsigned missing=kDeviceMoEMaxExperts;
    static_assert(kDeviceMoEMaxExperts==512);
    const uint64_t empty=routeKey(-INFINITY,missing);
    uint64_t q[8];
#pragma unroll
    for(unsigned part=0;part<8;++part){
        const unsigned id=64*part+lane;
        q[part]=id<unsigned(experts)?routeKey(probs[id],id):empty;
    }
#define SWAP_PACKED(A,B) do {const uint64_t a=q[A],b=q[B];const bool swap=keyBetter(b,a);q[A]=swap?b:a;q[B]=swap?a:b;}while(false)
        /* Five compare-swaps form a complete descending network for four. */
        SWAP_PACKED(0, 1);
        SWAP_PACKED(2, 3);
        SWAP_PACKED(0, 2);
        SWAP_PACKED(1, 3);
        SWAP_PACKED(1, 2);

        {
            // Two sorted four-element queues merge with a fixed nine-comparator
            // odd/even network. All values/IDs stay in scalar registers; no
            // thread-local expert-sized storage or repeated LDS scan is needed.
            SWAP_PACKED(4, 5);
            SWAP_PACKED(6, 7);
            SWAP_PACKED(4, 6);
            SWAP_PACKED(5, 7);
            SWAP_PACKED(5, 6);
            SWAP_PACKED(0, 4);
            SWAP_PACKED(2, 6);
            SWAP_PACKED(2, 4);
            SWAP_PACKED(1, 5);
            SWAP_PACKED(3, 7);
            SWAP_PACKED(3, 5);
            SWAP_PACKED(1, 2);
            SWAP_PACKED(3, 4);
            SWAP_PACKED(5, 6);
        }

#undef SWAP_PACKED
    float sum=0.0f;
    for(int slot=0;slot<extent;++slot) {
        uint64_t top=q[0];
        const auto step=[&](auto shift){
            constexpr int off=decltype(shift)::value;
            const unsigned hi=topKDownBits<TopKTransferPolicy::RowNativeRoot>(__builtin_bit_cast(int,unsigned(top>>32)),off);
            const unsigned lo=topKDownBits<TopKTransferPolicy::RowNativeRoot>(__builtin_bit_cast(int,unsigned(top)),off);
            const uint64_t other=(uint64_t(hi)<<32)|lo;
            top=keyBetter(other,top)?other:top;
        };
        step(std::integral_constant<int,32>{});step(std::integral_constant<int,16>{});
        step(std::integral_constant<int,8>{});step(std::integral_constant<int,4>{});
        step(std::integral_constant<int,2>{});step(std::integral_constant<int,1>{});
        const unsigned hi=unsigned(top>>32),lo=unsigned(top);
        const unsigned id=hi==0x7ff80000u?missing:(1023u-(lo&1023u));
        const unsigned winner=__builtin_amdgcn_readfirstlane(id);
        if(lane==0) {
            // One authoritative reload preserves the exact sign of zero and NaN payloads.
            const float value=id<unsigned(experts)?probs[id]:hi==0x7ff80000u?__builtin_bit_cast(float,lo):-INFINITY;
            ids[slot]=int(id);weights[slot]=value;sum+=value;
        }
        if(winner<unsigned(experts)&&lane==(winner&63u)){
#pragma unroll
            for(unsigned i=0;i<7;++i)q[i]=q[i+1];
            q[7]=empty;
        }
    }
    return sum;
}
/** @brief Original `moe_select_topk_probabilities_wave64_register_queue` shared without changing comparisons or rounding. */
template<int Candidates=4,bool SelectQueueSwaps=false,TopKTransferPolicy Transfer=TopKTransferPolicy::Indexed,bool SumAtSelection=false,bool PackedQueue=false>
__device__ __forceinline__ float
    moe_select_topk_probabilities_wave64_register_queue(
        const float *__restrict__ expert_probs,
        int num_experts,
        int top_k,
        int *__restrict__ selected_ids,
        float *__restrict__ selected_weights)
    {
        static_assert(Candidates == 4 || Candidates == 8);
        static_assert(!PackedQueue || (Transfer==TopKTransferPolicy::RowNativeRoot&&SumAtSelection),
            "Packed selection requires the ordered native root and lane-zero sum");
        if constexpr(PackedQueue&&Candidates==8)
            return packedQueue512(expert_probs,num_experts,top_k,selected_ids,selected_weights);
        constexpr int kWaveWidth = 64;
        const int lane = threadIdx.x & (kWaveWidth - 1);
        if (threadIdx.x >= kWaveWidth)
            return 0.0f;

        int id0 = lane;
        int id1 = lane + kWaveWidth;
        int id2 = lane + 2 * kWaveWidth;
        int id3 = lane + 3 * kWaveWidth;
        int id4 = lane + 4 * kWaveWidth;
        int id5 = lane + 5 * kWaveWidth;
        int id6 = lane + 6 * kWaveWidth;
        int id7 = lane + 7 * kWaveWidth;
        float value0 = id0 < num_experts ? expert_probs[id0] : -INFINITY;
        float value1 = id1 < num_experts ? expert_probs[id1] : -INFINITY;
        float value2 = id2 < num_experts ? expert_probs[id2] : -INFINITY;
        float value3 = id3 < num_experts ? expert_probs[id3] : -INFINITY;
        float value4 = -INFINITY, value5 = -INFINITY;
        float value6 = -INFINITY, value7 = -INFINITY;
        if constexpr (Candidates == 8)
        {
            value4 = id4 < num_experts ? expert_probs[id4] : -INFINITY;
            value5 = id5 < num_experts ? expert_probs[id5] : -INFINITY;
            value6 = id6 < num_experts ? expert_probs[id6] : -INFINITY;
            value7 = id7 < num_experts ? expert_probs[id7] : -INFINITY;
        }

        if (id0 >= num_experts || value0 != value0)
            id0 = kDeviceMoEMaxExperts;
        if (id1 >= num_experts || value1 != value1)
            id1 = kDeviceMoEMaxExperts;
        if (id2 >= num_experts || value2 != value2)
            id2 = kDeviceMoEMaxExperts;
        if (id3 >= num_experts || value3 != value3)
            id3 = kDeviceMoEMaxExperts;
        if constexpr (Candidates == 8)
        {
            if (id4 >= num_experts || value4 != value4) id4 = kDeviceMoEMaxExperts;
            if (id5 >= num_experts || value5 != value5) id5 = kDeviceMoEMaxExperts;
            if (id6 >= num_experts || value6 != value6) id6 = kDeviceMoEMaxExperts;
            if (id7 >= num_experts || value7 != value7) id7 = kDeviceMoEMaxExperts;
        }

#define LLAMINAR_MOE_SORT_PAIR_DESC(A, B)                                    \
        do                                                                    \
        {                                                                     \
            const bool replace = ::llaminar2::rocm::router_native::moe_topk_pair_better(value##B, id##B, value##A, id##A); \
            if constexpr (SelectQueueSwaps)                                   \
            {                                                                 \
                const float original_value = value##A;                         \
                const int original_id = id##A;                                 \
                value##A = replace ? value##B : original_value;                 \
                id##A = replace ? id##B : original_id;                           \
                value##B = replace ? original_value : value##B;                 \
                id##B = replace ? original_id : id##B;                           \
            }                                                                 \
            else if (replace)                                                 \
            {                                                                 \
                const float swap_value = value##A;                            \
                const int swap_id = id##A;                                    \
                value##A = value##B;                                          \
                id##A = id##B;                                                \
                value##B = swap_value;                                        \
                id##B = swap_id;                                              \
            }                                                                 \
        } while (false)

        /* Five compare-swaps form a complete descending network for four. */
        LLAMINAR_MOE_SORT_PAIR_DESC(0, 1);
        LLAMINAR_MOE_SORT_PAIR_DESC(2, 3);
        LLAMINAR_MOE_SORT_PAIR_DESC(0, 2);
        LLAMINAR_MOE_SORT_PAIR_DESC(1, 3);
        LLAMINAR_MOE_SORT_PAIR_DESC(1, 2);
        if constexpr (Candidates == 8)
        {
            // Two sorted four-element queues merge with a fixed nine-comparator
            // odd/even network. All values/IDs stay in scalar registers; no
            // thread-local expert-sized storage or repeated LDS scan is needed.
            LLAMINAR_MOE_SORT_PAIR_DESC(4, 5);
            LLAMINAR_MOE_SORT_PAIR_DESC(6, 7);
            LLAMINAR_MOE_SORT_PAIR_DESC(4, 6);
            LLAMINAR_MOE_SORT_PAIR_DESC(5, 7);
            LLAMINAR_MOE_SORT_PAIR_DESC(5, 6);
            LLAMINAR_MOE_SORT_PAIR_DESC(0, 4);
            LLAMINAR_MOE_SORT_PAIR_DESC(2, 6);
            LLAMINAR_MOE_SORT_PAIR_DESC(2, 4);
            LLAMINAR_MOE_SORT_PAIR_DESC(1, 5);
            LLAMINAR_MOE_SORT_PAIR_DESC(3, 7);
            LLAMINAR_MOE_SORT_PAIR_DESC(3, 5);
            LLAMINAR_MOE_SORT_PAIR_DESC(1, 2);
            LLAMINAR_MOE_SORT_PAIR_DESC(3, 4);
            LLAMINAR_MOE_SORT_PAIR_DESC(5, 6);
        }
#undef LLAMINAR_MOE_SORT_PAIR_DESC

        float selected_sum=0.0f;
        for (int slot = 0; slot < top_k; ++slot)
        {
            float best_value = value0;
            int best_id = id0;
            const auto append_level=[&](auto distance) __attribute__((always_inline)) {
                const int offset=distance;
                const float other_value = topKDownBits<Transfer>(best_value, offset);
                const int other_id = topKDownBits<Transfer>(best_id, offset);
                if (::llaminar2::rocm::router_native::moe_topk_pair_better(
                        other_value,
                        other_id,
                        best_value,
                        best_id))
                {
                    best_value = other_value;
                    best_id = other_id;
                }
            };
            if constexpr(Transfer==TopKTransferPolicy::RowNativeRoot) {
                append_level(std::integral_constant<int,32>{});
                append_level(std::integral_constant<int,16>{});
                append_level(std::integral_constant<int,8>{});
                append_level(std::integral_constant<int,4>{});
                append_level(std::integral_constant<int,2>{});
                append_level(std::integral_constant<int,1>{});
            } else {
                for (int offset = kWaveWidth / 2; offset > 0; offset >>= 1)
                    append_level(offset);
            }

            // Full wave0 is reconverged here; its lane0 root alone controls every
            // queue pop and selected output. No float conversion is involved.
            const int winner = Transfer==TopKTransferPolicy::Indexed ? __shfl(best_id, 0) :
                __builtin_amdgcn_readfirstlane(best_id);
            if (lane == 0)
            {
                selected_ids[slot] = best_id;
                selected_weights[slot] = best_value;
                // Lane zero already owns the next exact weight in router order.
                if constexpr(SumAtSelection) selected_sum += best_value;
            }
            if (winner < num_experts &&
                lane == (winner & (kWaveWidth - 1)))
            {
                value0 = value1;
                id0 = id1;
                value1 = value2;
                id1 = id2;
                value2 = value3;
                id2 = id3;
                if constexpr (Candidates == 8)
                {
                    value3 = value4; id3 = id4;
                    value4 = value5; id4 = id5;
                    value5 = value6; id5 = id6;
                    value6 = value7; id6 = id7;
                    value7 = -INFINITY; id7 = kDeviceMoEMaxExperts;
                }
                else
                {
                    value3 = -INFINITY;
                    id3 = kDeviceMoEMaxExperts;
                }
            }
        }
        return selected_sum;
    }
/** @brief Original `router_fp16_operand` shared without changing comparisons or rounding. */
__device__ __forceinline__ float router_fp16_operand(float value)
{
    return __half2float(__float2half_rn(value));
}
/** @brief Exact original phase-owned shared arrays; reused only after a completed CTA/grid edge. */
struct RouterSelectionShared {
 float expert_probs[kDeviceMoEMaxExperts];float shared_max[8];float shared_sum[8];
 int selected_ids[kDeviceMoEMaxTopK];float selected_weights[kDeviceMoEMaxTopK];
};
static_assert(sizeof(RouterSelectionShared)==2240);
/** @brief Compile-time ownership of pre-normalization selector intermediates. */
enum class SelectorIntermediateStorage { Shared, ThreadPair };
/**
 * @brief Borrow original storage or retain two same-thread values through joins.
 * @tparam Policy Exact intermediate ownership, with Shared preserving default behavior.
 * @details ThreadPair requires the existing num_experts<=512/logical256 contract.
 * Each live thread owns experts tid and tid+256. Fields are overwritten only after
 * that same expert's original logit is consumed by expf; no peer reads them.
 */
template<SelectorIntermediateStorage Policy>
struct SelectorIntermediateValues {
 static_assert(Policy==SelectorIntermediateStorage::Shared||Policy==SelectorIntermediateStorage::ThreadPair,
     "Selector intermediate storage must name a declared policy");
 float first=0.0f,second=0.0f; ///< Named values; no dynamically indexed private array.
 /** @brief Read original max input and retain only under ThreadPair.
  * @param logits Immutable phase input. @param expert Live expert. @param tid Original logical thread.
  * @return Exact original logit bits. */
 __device__ __forceinline__ float maximumInput(const float* logits,int expert,unsigned tid) {
  const float value=logits[expert];
  if constexpr(Policy==SelectorIntermediateStorage::ThreadPair) {
   if(expert==int(tid))first=value;else second=value;
  }
  return value;
 }
 /** @brief Read same-thread logit for expf.
  * @param logits Original immutable source. @param expert Live expert. @param tid Original logical thread.
  * @return Original logit bits without an extra read for ThreadPair. */
 __device__ __forceinline__ float exponentialInput(const float* logits,int expert,unsigned tid) const {
  if constexpr(Policy==SelectorIntermediateStorage::ThreadPair)return expert==int(tid)?first:second;
  else return logits[expert];
 }
 /** @brief Retain the exact original exponential until normalization.
  * @param shared Original probability plane. @param expert Live expert. @param tid Original thread.
  * @param value Original expf result. */
 __device__ __forceinline__ void publishExponential(float* shared,int expert,unsigned tid,float value) {
  if constexpr(Policy==SelectorIntermediateStorage::ThreadPair) {
   if(expert==int(tid))first=value;else second=value;
  } else shared[expert]=value;
 }
 /** @brief Read exact original exponential at its original normalization edge.
  * @param shared Original probability plane. @param expert Live expert. @param tid Original thread.
  * @return Retained or original LDS exponential bits. */
 __device__ __forceinline__ float normalizationInput(const float* shared,int expert,unsigned tid) const {
  if constexpr(Policy==SelectorIntermediateStorage::ThreadPair)return expert==int(tid)?first:second;
  else return shared[expert];
 }
};
/**
 * @brief Execute the original logical256 selection arithmetic into caller-owned LDS.
 * @tparam SelectQueueSwaps Existing comparator scheduling policy.
 * @tparam LogicalThreads Original256-lane arithmetic tree inside the physical CTA.
 * @tparam WriteProbabilities Whether the original public logits may be overwritten now.
 * @tparam JoinMemory Original standalone scope by default; a phase-local caller
 * may explicitly own only LDS edges when global publication is separately joined.
 * @tparam Transfer Explicit tournament bit moves; all numerical nodes remain original.
 * @tparam Intermediate Explicit same-thread intermediate ownership; default Shared unchanged.
 * @param logits Immutable original projection logits, shared across CTA-local selections.
 * @param published_logits Original probability destination, required only when WriteProbabilities.
 * @param num_experts Original expert bound. @param top_k Original selected packet width.
 * @param normalize_weights Original serial normalization policy.
 * @param scratch CTA-owned original selection planes, retained through their last consumer.
 * @details This is the single canonical arithmetic body. Selection-only callers
 * must defer public probability overwrite until every CTA finished reading logits.
 */
template<bool SelectQueueSwaps=false,unsigned LogicalThreads=256,bool WriteProbabilities=true,
    WorkgroupJoinMemory JoinMemory=WorkgroupJoinMemory::GlobalAndLDS,
    TopKTransferPolicy Transfer=TopKTransferPolicy::Indexed,
    SelectorIntermediateStorage Intermediate=SelectorIntermediateStorage::Shared,bool SumAtSelection=false,bool PackedQueue=false>
__device__ __forceinline__ void selectProbabilities(const float* logits,float* published_logits,
    int num_experts,int top_k,bool normalize_weights,RouterSelectionShared& scratch)
{
    static_assert(LogicalThreads==256);
    static_assert(Intermediate==SelectorIntermediateStorage::Shared||Intermediate==SelectorIntermediateStorage::ThreadPair,
        "Selector intermediate storage must name a declared policy");
    static_assert(Intermediate!=SelectorIntermediateStorage::ThreadPair||
        kDeviceMoEMaxExperts<=2*LogicalThreads,
        "Two named selector values must cover the complete declared expert capacity");
    // Keep selector-only lane/warp coordinates out of later decoder phases.
    // The value remains this exact lane; the empty boundary emits no memory
    // operation and changes neither participation nor reduction arithmetic.
    unsigned selector_thread=threadIdx.x;
    asm volatile("" : "+v"(selector_thread));
    SelectorIntermediateValues<Intermediate> retained;
    auto& expert_probs=scratch.expert_probs;
    auto& shared_max=scratch.shared_max;
    auto& shared_sum=scratch.shared_sum;

    const int lane = selector_thread % warpSize;
    const int warp_id = selector_thread / warpSize;
    float max_val = -FLT_MAX;
    for (int expert = selector_thread; expert < num_experts && selector_thread<LogicalThreads;
         expert += LogicalThreads)
    {
        max_val = fmaxf(max_val, retained.maximumInput(logits,expert,selector_thread));
    }
    max_val = rocm_moe_wave_reduce_max_f32(max_val);
    if (selector_thread<LogicalThreads && lane == 0)
        shared_max[warp_id] = max_val;
    workgroupJoin<JoinMemory>();

    // Every consuming wave reads the same published roots. Their slots stay immutable
    // until the probability plane has retired, so a second CTA publication is unnecessary.
    if constexpr (SumAtSelection) {
        if (selector_thread < LogicalThreads) {
            const unsigned groups=(LogicalThreads+warpSize-1)/warpSize;
            const float member=lane < groups ? shared_max[lane] : -FLT_MAX;
            max_val=__shfl(rocm_moe_wave_reduce_max_f32(member),0);
        }
    } else {
    if (warp_id == 0)
    {
        const int num_warps =
            (LogicalThreads + warpSize - 1) / warpSize;
        max_val = lane < num_warps ? shared_max[lane] : -FLT_MAX;
        max_val = rocm_moe_wave_reduce_max_f32(max_val);
        if (lane == 0)
            shared_max[0] = max_val;
    }
    workgroupJoin<JoinMemory>();
    max_val = shared_max[0];

    }
    float sum = 0.0f;
    for (int expert = selector_thread; expert < num_experts && selector_thread<LogicalThreads;
         expert += LogicalThreads)
    {
        const float value = expf(retained.exponentialInput(logits,expert,selector_thread) - max_val);
        retained.publishExponential(expert_probs,expert,selector_thread,value);
        sum += value;
    }
    sum = rocm_moe_wave_reduce_sum_f32(sum);
    if (selector_thread<LogicalThreads && lane == 0)
        shared_sum[warp_id] = sum;
    workgroupJoin<JoinMemory>();

    if constexpr (SumAtSelection) {
        if (selector_thread < LogicalThreads) {
            const unsigned groups=(LogicalThreads+warpSize-1)/warpSize;
            const float member=lane < groups ? shared_sum[lane] : 0.0f;
            sum=__shfl(rocm_moe_wave_reduce_sum_f32(member),0);
        }
    } else {
    if (warp_id == 0)
    {
        const int num_warps =
            (LogicalThreads + warpSize - 1) / warpSize;
        sum = lane < num_warps ? shared_sum[lane] : 0.0f;
        sum = rocm_moe_wave_reduce_sum_f32(sum);
        if (lane == 0)
            shared_sum[0] = sum;
    }
    workgroupJoin<JoinMemory>();

        sum=shared_sum[0];
    }
    const float inv_sum=sum>0.0f ? 1.0f/sum : 0.0f;
    for (int expert = selector_thread; expert < num_experts && selector_thread<LogicalThreads;
         expert += LogicalThreads)
    {
        const float probability = retained.normalizationInput(expert_probs,expert,selector_thread) * inv_sum;
        expert_probs[expert] = probability;
        if constexpr(WriteProbabilities)published_logits[expert] = probability;
    }
    workgroupJoin<JoinMemory>();

    auto& selected_ids=scratch.selected_ids;
    auto& selected_weights=scratch.selected_weights;
    float fused_selected_sum=0.0f;
    if (num_experts > 256)
        fused_selected_sum=::llaminar2::rocm::router_native::moe_select_topk_probabilities_wave64_register_queue<8, SelectQueueSwaps, Transfer,SumAtSelection,PackedQueue>(
            expert_probs, num_experts, top_k, selected_ids, selected_weights);
    else
        fused_selected_sum=::llaminar2::rocm::router_native::moe_select_topk_probabilities_wave64_register_queue<4, SelectQueueSwaps, Transfer,SumAtSelection>(
            expert_probs, num_experts, top_k, selected_ids, selected_weights);

    if constexpr(SumAtSelection) {
        // The writer and these consumers are all in the original winning wave.
        if (warp_id==0 && normalize_weights) {
            const float topk_sum=__shfl(fused_selected_sum,0);
            if(topk_sum>0.0f && lane<top_k)selected_weights[lane] *= 1.0f/topk_sum;
        }
    } else if (selector_thread == 0)
    {
        if (normalize_weights)
        {
            float topk_sum = 0.0f;
            for (int slot = 0; slot < top_k; ++slot)
                topk_sum += selected_weights[slot];
            if (topk_sum > 0.0f)
            {
                const float inv_topk_sum = 1.0f / topk_sum;
                for (int slot = 0; slot < top_k; ++slot)
                    selected_weights[slot] *= inv_topk_sum;
            }
        }
    }
    // Keep the serial FP32 normalization above unchanged. Every selected
    // slot's lane must observe its final bits before publishing owner metadata
    // and exact integer demand. Inactive lanes publish nothing.
    workgroupJoin<JoinMemory>();
}
/** @brief Original physical256-lane probability/top-k/normalization/publication phase.
 * @tparam UpdateRuntimeHistogram Original integer demand policy.
 * @tparam RowPolicy Original runtime row selector.
 * @tparam PublishDescriptors Preserve selected descriptor public outputs.
 * @tparam SelectQueueSwaps Explicit original comparator scheduling control; defaultfalse.
 * @tparam RuntimeView Existing runtime view, never a copied/new layout.
 * @param logits Original source logits, overwritten with probabilities.
 * @param runtime Original request-acquired device runtime and placement authority.
 * @param legacy_indices Original complete selected expert-ID output, when declared.
 * @param legacy_weights Original complete selected FP32 weight output, when declared.
 * @param num_experts Exact source expert extent, bounded by512.
 * @param top_k Original selected packet width, bounded by16.
 * @param normalize_weights Original serial selected-slot normalization policy.
 * @param write_legacy_outputs Whether both original complete outputs are declared.
 * @param active_row_count_device Original optional device0/1 authority; never host-read.
 * @param publication Original selected descriptor destinations/format/readiness policy.
 * @param scratch Original2240-byte shared storage supplied by caller.
 * @details All256 physical threads enter. Publication may return for inactive
 * slot lanes; an enclosing cooperative caller must still execute its grid join.
 */
template<bool UpdateRuntimeHistogram,
 llaminar2::RoutedExpertRowExecutionPolicy RowPolicy=llaminar2::RoutedExpertRowExecutionPolicy::ParticipantAssigned,
 bool PublishDescriptors=false,bool SelectQueueSwaps=false,unsigned LogicalThreads=256,class RuntimeView>
__device__ __forceinline__ void selectAndPublish(
    float *__restrict__ logits,
    RuntimeView *__restrict__ runtime,
    float *__restrict__ legacy_indices,
    float *__restrict__ legacy_weights,
    int num_experts,
    int top_k,
    bool normalize_weights,
    bool write_legacy_outputs,
        const int32_t *__restrict__ active_row_count_device,
        llaminar2::MoEDecodeDescriptorPublication publication,RouterSelectionShared& scratch)
{
    static_assert(LogicalThreads==256);
    static_assert(std::is_standard_layout_v<RuntimeView>);
    static_assert(sizeof(RuntimeView)==moe_runtime_abi::kLayerRuntimeBytes);
    static_assert(sizeof(RouterBankView<RuntimeView>)==moe_runtime_abi::kPlacementBankBytes);
    static_assert(offsetof(RuntimeView,participant_count)==moe_runtime_abi::kParticipantCountOffset);
    static_assert(offsetof(RuntimeView,route_participant_ids)==moe_runtime_abi::kRouteParticipantIdsOffset);
    static_assert(offsetof(RuntimeView,intermediate_shard)==moe_runtime_abi::kIntermediateShardBindingOffset);
    static_assert(offsetof(RuntimeView,runtime_histogram_banks)==moe_runtime_abi::kRuntimeHistogramBanksOffset);
    static_assert(offsetof(RuntimeView,runtime_histogram_active_bank)==moe_runtime_abi::kRuntimeHistogramActiveBankOffset);
    static_assert(offsetof(RuntimeView,overlay_epoch_ticket)==moe_runtime_abi::kOverlayEpochTicketOffset);
    static_assert(offsetof(RuntimeView,overlay_placement_banks)==moe_runtime_abi::kOverlayPlacementBanksOffset);
    static_assert(offsetof(RuntimeView,overlay_epoch_status)==moe_runtime_abi::kOverlayEpochStatusOffset);
    if (!logits || !runtime || num_experts <= 0 ||
        num_experts > kDeviceMoEMaxExperts || top_k <= 0 ||
        top_k > kDeviceMoEMaxTopK)
    {
        return;
    }

    selectProbabilities<SelectQueueSwaps,LogicalThreads,true>(logits,logits,num_experts,top_k,normalize_weights,scratch);
    ::llaminar2::rocm::router_native::runtime_publish_independent_decode_dispatch<UpdateRuntimeHistogram, RowPolicy, PublishDescriptors>(
        runtime, scratch.selected_ids, scratch.selected_weights, num_experts, top_k,
        legacy_indices, legacy_weights, write_legacy_outputs, active_row_count_device, publication);
}
/**
 * @brief One original logical128-thread FP16 expert dot inside a physical128/256 CTA.
 * @tparam JoinMemory Workgroup scope; standalone default remains global+LDS.
 * @param hidden Original FP32 mixed row, half-RNE at each original operand load.
 * @param gate_weights Original contiguous half expert rows.
 * @param logits Original public FP32 projection outputs.
 * @param d_model Exact original K; odd K retains the scalar expression.
 * @param num_experts Actual source expert bound, including an odd final paired task.
 * @param expert Indexed output task; absent final team still reaches the CTA barrier.
 * @param team_lane Original0..127 arithmetic lane, not physical thread index.
 * @param wave_totals Caller-exclusive storage for at least128/warpSize FP32 totals.
 * @details Each physical team has disjoint wave_totals. Caller retires all
 * scratch readers with a CTA barrier before reuse; kernel completion is enough
 * for the standalone original. Existing wave64 contract is explicit above.
 */
template<WorkgroupJoinMemory JoinMemory=WorkgroupJoinMemory::GlobalAndLDS>
__device__ __forceinline__ void projectionExpert128(const float* hidden,const __half* gate_weights,
 float* logits,int d_model,int num_experts,int expert,unsigned team_lane,float* wave_totals) {
 constexpr int TeamThreads=128;
 const int wavesPerTeam=TeamThreads/warpSize;
 float sum=0.0f;
 if(expert<num_experts) {
  const __half* w=gate_weights+static_cast<size_t>(expert)*d_model;
  if((d_model&1)==0) {
   const int pair_count=d_model>>1;const __half2* w2=reinterpret_cast<const __half2*>(w);
   for(int i=team_lane;i<pair_count;i+=TeamThreads) {
    const float2 ww=__half22float2(w2[i]);const int h=i<<1;
    sum+=::llaminar2::rocm::router_native::router_fp16_operand(hidden[h])*ww.x+::llaminar2::rocm::router_native::router_fp16_operand(hidden[h+1])*ww.y;
   }
  } else for(int i=team_lane;i<d_model;i+=TeamThreads)
   sum+=::llaminar2::rocm::router_native::router_fp16_operand(hidden[i])*__half2float(w[i]);
 }
 sum=rocm_moe_wave_reduce_sum_f32(sum);
 const int lane=team_lane%warpSize,team_wave=team_lane/warpSize;
 if(lane==0)wave_totals[team_wave]=sum;
 workgroupJoin<JoinMemory>();
 if(team_wave==0) {
  sum=lane<wavesPerTeam?wave_totals[lane]:0.0f;
  sum=rocm_moe_wave_reduce_sum_f32(sum);
  if(lane==0&&expert<num_experts)logits[expert]=sum;
 }
}
#undef LLAMINAR_ROUTER_PHASE_FAIL
} // namespace llaminar2::rocm::router_native
