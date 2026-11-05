/**
 * @file PooledImmutableLocalLookupGraph.h
 * @brief Declarative complete local pooled lookup wiring through the existing real-provider boundary.
 *
 * Native owner and coordinator phases retain their original arbitrary-owner
 * interfaces. This builder admits only the explicitly materialized local edge;
 * absent directed transport is a hard error, never a topology capability.
 * Source-success and failure cleanup branches are fixed captured nodes.
 */
#pragma once
#include <optional>
#include "PooledImmutableDemandFrontier.h"
#include "execution/compute_stages/stages/qwen4exp/PooledImmutableSourcePhaseStage.h"
#include "execution/compute_stages/stages/qwen4exp/PooledImmutableNativePhaseStage.h"
#include "execution/local_execution/graph/ComputeGraph.h"
namespace llaminar2 {
struct PooledImmutableReplicatedReaderCompletion;
/**
 * @brief Original P1 phase inventory retained by the actual recurring source plan.
 *
 * This cold descriptor owns no runtime state or storage. Stage identities and
 * dependencies refer to the one complete local graph; shared-reader ACKs are
 * authenticated separately before its original ReadersRetired phase.
 */
struct PooledImmutableLookupGraphEndpoints;
struct PooledImmutableDirectedLookupEndpoints;
namespace qwen4exp {class PooledImmutableDirectedTransport;}
class PooledImmutableLocalRetirementCompletion final {
public:
    /** @return Same original real-provider boundary retained by the complete local transaction. */
    const auto& sourceBoundary()const noexcept{return boundary;}
    /** @return Original first local native phase. */
    const std::string& firstNode()const noexcept{return first;}
    /** @return Original terminal source observation. */
    const std::string& lastNode()const noexcept{return last;}
    /** @return Exact original native Read producer. */
    const std::string& readNode()const noexcept{return read;}
    /** @return Original last-reader frontier preceding storage release. */
    const std::string& readersRetiredNode()const noexcept{return readersRetired;}
    /** @return Actual original shared-reader completion, absent only for singleton composition. */
    const auto& sharedReaders()const noexcept{return readers;}
private:
    friend PooledImmutableLookupGraphEndpoints appendPooledImmutableLocalLookupGraph(ComputeGraph&,BufferArena&,
        std::shared_ptr<qwen4exp::ImmutableLookupExecutionBoundary>,
        std::shared_ptr<const qwen4exp::IPooledImmutableNativeKernels>,qwen4exp::pooled_native::OriginalRows,
        qwen4exp::PooledNativeOperandSpans,DeviceGenerationControlBinding,qwen4exp::PooledNativeTensorSpan,
        const std::string&,const std::string&,std::shared_ptr<PooledImmutableLocalRetirementCompletion>*);
    friend PooledImmutableDirectedLookupEndpoints appendPooledImmutableDirectedLookupGraph(ComputeGraph&,BufferArena&,
        std::shared_ptr<qwen4exp::ImmutableLookupExecutionBoundary>,std::shared_ptr<const qwen4exp::PooledImmutableDirectedTransport>,
        std::shared_ptr<const qwen4exp::IPooledImmutableNativeKernels>,qwen4exp::pooled_native::OriginalRows,
        qwen4exp::PooledNativeOperandSpans,DeviceGenerationControlBinding,qwen4exp::PooledNativeTensorSpan,
        const std::string&,const std::string&);
    friend void requirePooledLocalRetirementCompletion(const ComputeGraph&,const PooledImmutableLocalRetirementCompletion&);
    /** @brief One original local phase and its exact predecessor inventory. */
    struct Phase { std::string name; const IComputeStage* stage; std::vector<std::string> dependencies; };
    /**
     * @brief Capture original graph inputs; callers cannot fabricate a terminal inventory.
     * @param source Same materialized real-provider boundary.
     * @param binding Canonical controller borrowed by these exact phases.
     * @param rows Original counted/fixed ID extent. @param ids Exact Begin ID subview.
     * @param result Original FP32 Read output subview.
     */
    PooledImmutableLocalRetirementCompletion(std::shared_ptr<qwen4exp::ImmutableLookupExecutionBoundary> source,
        DeviceGenerationControlBinding binding,qwen4exp::pooled_native::OriginalRows rows,qwen4exp::PooledNativeTensorSpan ids,qwen4exp::PooledNativeTensorSpan result)
        :boundary(std::move(source)),control(std::move(binding)),original(std::move(rows)),inputIds(std::move(ids)),output(std::move(result)){}
    std::shared_ptr<qwen4exp::ImmutableLookupExecutionBoundary> boundary;
    DeviceGenerationControlBinding control;
    qwen4exp::pooled_native::OriginalRows original;
    qwen4exp::PooledNativeTensorSpan inputIds,output;
    std::string first,last,read,readersRetired;
    std::vector<Phase> phases;
    std::shared_ptr<const PooledImmutableReplicatedReaderCompletion> readers;
};
/**
 * @brief Authenticate the original P1 phase chain and every actual shared-reader ACK.
 * @param graph Original graph containing the unchanged complete local transaction.
 * @param original Cold descriptor issued by the original local builder.
 * @throws std::invalid_argument For substituted stages, roots or missing terminal dependencies.
 */
void requirePooledLocalRetirementCompletion(const ComputeGraph& graph,
    const PooledImmutableLocalRetirementCompletion& original);
/** @brief Exact endpoints for actual hash/control producer and next model consumer dependencies. */
struct PooledImmutableLookupGraphEndpoints {
    std::string first,last;
    std::optional<PooledImmutableDemandFrontier> demand; ///< Actual original current-demand split, absent for unrelated phase endpoints.
};
/**
 * @brief Append a complete captured local transaction with actual provider/ACK/Ready/read/release ownership.
 * @param graph Existing participant-local model graph, retaining its actual source boundary.
 * @param arena Original already-bound owner/coordinator/source/output/controller roots.
 * @param boundary Original real-provider pooled arm, retained through terminal and final readers.
 * @param code Stateless owner/coordinator bundle for the exact native backend.
 * @param original Original complete counted IDs and actual width, preserving stable coalescing/permutation.
 * @param begin Actual ID/count/admission tensor roles; no host values are copied.
 * @param control Same canonical request controller borrowed by every source branch.
 * @param output Exact prepared original-order FP32 output span.
 * @param identity Unique stable node prefix; no execution policy is selected by strings.
 * @param outputIdentity Optional canonical model snapshot node name for the actual native reader.
 * @param retirement Optional original phase inventory for recurring/shared-reader composition.
 * @return Actual first/last graph nodes for explicit external producer/consumer dependencies.
 * @throws std::exception For missing directed topology, foreign roots or capture-time construction.
 */
PooledImmutableLookupGraphEndpoints appendPooledImmutableLocalLookupGraph(ComputeGraph& graph,BufferArena& arena,
    std::shared_ptr<qwen4exp::ImmutableLookupExecutionBoundary> boundary,
    std::shared_ptr<const qwen4exp::IPooledImmutableNativeKernels> code,qwen4exp::pooled_native::OriginalRows original,
    qwen4exp::PooledNativeOperandSpans begin,DeviceGenerationControlBinding control,
    qwen4exp::PooledNativeTensorSpan output,const std::string& identity,const std::string& outputIdentity={},
    std::shared_ptr<PooledImmutableLocalRetirementCompletion>* retirement=nullptr);
/**
 * @brief Append the actual PLE hash-ID lookup using the original sixteen-head token count authority.
 * @param graph Existing participant-local graph. @param arena Original bound model arena.
 * @param boundary Original real-provider split source domain. @param code Exact native backend bundle.
 * @param tokens Original complete token-query descriptor, including counted M1/empty replay.
 * @param ids Original packed INT64 hash IDs, without narrowing or a converted shadow buffer.
 * @param count Original token-count arena subview, present exactly for counted queries.
 * @param control Canonical device generation controller. @param output Actual FP32 embedding span.
 * @param identity Stable local protocol node prefix. @param outputIdentity Actual model snapshot node name.
 * @return Complete captured transaction endpoints, with no host count read or new count scalar.
 */
PooledImmutableLookupGraphEndpoints appendPooledImmutablePLELookupGraph(ComputeGraph& graph,BufferArena& arena,
    std::shared_ptr<qwen4exp::ImmutableLookupExecutionBoundary> boundary,
    std::shared_ptr<const qwen4exp::IPooledImmutableNativeKernels> code,DeviceRowRange tokens,
    qwen4exp::PooledNativeTensorSpan ids,std::optional<qwen4exp::PooledNativeTensorSpan> count,
    DeviceGenerationControlBinding control,qwen4exp::PooledNativeTensorSpan output,
    const std::string& identity,const std::string& outputIdentity);

}
