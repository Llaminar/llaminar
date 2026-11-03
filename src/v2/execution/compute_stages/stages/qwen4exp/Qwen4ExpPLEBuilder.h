/**
 * @file Qwen4ExpPLEBuilder.h
 * @brief Declarative native n-gram lookup, similarity gating and lexical convolution.
 *
 * This participant-local builder composes the canonical prepared GEMMs with
 * backend-neutral model primitives. It retains immutable prior histories and
 * exposes candidate normalized rows to the separate accepted-state authority.
 * Native VRAM, RAM or SSD mmap storage is resolved at admission.
 */
#pragma once
namespace llaminar2::qwen4exp {class PooledImmutableRetainedParticipants;}
#include "ImmutableLookupStage.h"
#include "GraphLocalLookupFrame.h"
#include "Qwen4ExpFixedHeadLookupStage.h"
#include "Qwen4ExpHyperconnectionBuilder.h"
#include "execution/qwen4exp/PooledImmutableLocalLookupGraph.h"
#include <variant>

namespace llaminar2
{
/** @brief Complete immutable geometry and arithmetic of one lexical operation. */
struct Qwen4ExpPLEContext
{
    DeviceId device;
    const IMPIContext *mpi=nullptr;
    std::string prefix;
    Qwen4ExpStageRows rows;
    int hidden;
    float epsilon;
    qwen4exp::NgramHashPolicy hash;
    TensorType nativeFormat;
    int nativeRowElements;
    std::size_t nativeRowBytes;
    Qwen4ExpProjectionArithmetic arithmetic=Qwen4ExpProjectionArithmetic::Ordinary;
    /** @brief Original complete token-query descriptor under explicit verifier arithmetic.
     * @details Includes the exact primitive count owner/capacity/origin/repeat. Omission is
     * valid only when primitives are fully active; counted physical M1 keeps its descriptor.
     * Ordinary arithmetic does not accept a counted verifier descriptor.
     */
    std::optional<DeviceRowRange> projectionRows;
};
/** @brief Frozen exact projection/gain/convolution bindings; table storage is separate. */
struct Qwen4ExpPLEWeights
{
    Qwen4ExpPreparedProjection key,value;
    Qwen4ExpImmutableWeightBinding keyNorm,queryNorm,convolutionNorm,convolution;
};
/** @brief All scratch and immutable request history comes from the admitted arena. */
struct Qwen4ExpPLEBuffers
{
    Qwen4ExpIntInput tokens;
    Qwen4ExpIntBinding priorTokens,ids;
    Qwen4ExpFloatBinding embedding,key,value,query,gated,normalized;
    Qwen4ExpFloatBinding priorConvolution,trunk;
};
/** @brief Exact original split-native source domain, canonical controller and token-count authority. */
struct Qwen4ExpPooledLookupStorage {
    BufferArena* arena;
    std::shared_ptr<qwen4exp::ImmutableLookupExecutionBoundary> boundary;
    std::shared_ptr<const qwen4exp::IPooledImmutableNativeKernels> code;
    DeviceGenerationControlBinding control;
    DeviceRowRange originalRows; ///< Original token rows; sixteen-head expansion borrows this exact count.
    std::optional<Qwen4ExpIntBinding> count;
    std::shared_ptr<qwen4exp::PooledImmutableRetainedParticipants> participants; ///< Original model-owned recurring remote programs, absent only for an actually local layout.
};
/**
 * @brief Query-only compute participant in the original shared replicated PLE cohort.
 *
 * The foreign graph owns its arena/controller/query descriptor and receives its
 * exact live embedding. It borrows the same retained source participants and
 * owns neither a CPU channel nor a source cache or physical pin authority.
 */
struct Qwen4ExpPooledReaderLookupStorage {
    BufferArena* arena;
    std::shared_ptr<qwen4exp::PooledImmutableRetainedParticipants> participants;
    DeviceGenerationControlBinding control;
    DeviceRowRange originalRows;
    std::optional<Qwen4ExpIntBinding> count; ///< Same original arena-owned primitive count, retained through capture/reset.
};
/**
 * @brief Graph-local exact-once current-demand publication, never request execution state.
 * The retained boundary owns the one device generation/sequence and pinned response.
 * This construction capability owns no physical bytes and never observes token values.
 */
class Qwen4ExpEarlyLookupFrame final {
public:
    /** @brief Append the actual native hash and Publish before any model trunk root. */
    static std::unique_ptr<Qwen4ExpEarlyLookupFrame> publish(ComputeGraph&,const Qwen4ExpPLEContext&,
        Qwen4ExpIntInput,Qwen4ExpIntBinding,Qwen4ExpIntBinding,Qwen4ExpFloatBinding,
        ImmutableLookupStage::GPUStorage);
    Qwen4ExpEarlyLookupFrame(const Qwen4ExpEarlyLookupFrame&)=delete;
    Qwen4ExpEarlyLookupFrame& operator=(const Qwen4ExpEarlyLookupFrame&)=delete;
    /** @return Original Publish producer to order every trunk root. */
    const std::string& producer() const { return handoff_.producer(); }
    /** @brief Prove the issuing graph consumed its one frame before returning a complete graph. */
    void requireConsumed() const;
    /** @brief Authenticate exact graph/query/history/source, then transfer one delayed Consume stage. */
    struct Consumption { std::unique_ptr<IComputeStage> stage;std::vector<std::string> dependencies; };
    Consumption consume(ComputeGraph&,ImmutableLookupStage::Params,std::uint64_t hashIdentity,const std::vector<std::string>& trunk);
private:
    /** @brief Bind immutable original declarations; only publish can issue a frame. */
    Qwen4ExpEarlyLookupFrame(ComputeGraph&,ImmutableLookupStage::Params,std::uint64_t,std::string);
    ImmutableLookupStage::Params params_;
    GraphLocalLookupFrame handoff_;
};
/**
 * @brief Exact current-invocation resident gather retained beside primary trunk work.
 *
 * This move-only construction capability owns no execution state. The existing
 * immutable resident table and arena output remain original owners. The native
 * event family is retained explicitly; no response bank or second copy exists.
 */
class Qwen4ExpResidentReadAheadFrame final {
public:
    /** @brief Append Hash and the original native resident gather before model roots. */
    static std::unique_ptr<Qwen4ExpResidentReadAheadFrame> publish(ComputeGraph&,const Qwen4ExpPLEContext&,
        Qwen4ExpIntInput,Qwen4ExpIntBinding,Qwen4ExpIntBinding,Qwen4ExpFloatBinding,
        ImmutableLookupStage::ResidentGPUStorage,std::shared_ptr<ReadAheadNativeResourceFamily>);
    Qwen4ExpResidentReadAheadFrame(const Qwen4ExpResidentReadAheadFrame&)=delete;
    Qwen4ExpResidentReadAheadFrame& operator=(const Qwen4ExpResidentReadAheadFrame&)=delete;
    /** @return Query frontier; primary trunk never waits on resident gather. */
    const std::string& frontier()const noexcept{return hash_;}
    /** @brief Authenticate original graph/source/query and consume the one handoff. */
    std::vector<std::string> consume(ComputeGraph&,ImmutableLookupStage::Params,std::uint64_t,
        const std::vector<std::string>&);
    /** @brief Attach the native join to the actual first projection after nodes exist. */
    void sealBranch(ComputeGraph&,const std::string& firstConsumer);
    /** @brief Require a consumed handoff and sealed native event branch. */
    void requireConsumed()const;
private:
    /** @brief Bind the one issued producer and its complete immutable identities. */
    Qwen4ExpResidentReadAheadFrame(ComputeGraph&,ImmutableLookupStage::Params,std::uint64_t,std::string,std::string,
        std::shared_ptr<ReadAheadNativeResourceFamily>);
    ImmutableLookupStage::Params params_;
    GraphLocalLookupFrame handoff_;
    std::string hash_;
    std::shared_ptr<ReadAheadNativeResourceFamily> resources_;
    bool sealed_=false; ///< Graph construction state only, never native execution state.
};
/** @brief Reusable flat graph wiring with independent accepted-state publication. */
class Qwen4ExpPLEBuilder final
{
public:
    /** @brief Explicit whole-storage versus fixed-range composition; no automatic substitution. */
    using LookupStorage=std::variant<ImmutableLookupStage::Storage,Qwen4ExpFixedHeadLookupStorage,Qwen4ExpPooledLookupStorage,Qwen4ExpPooledReaderLookupStorage>;
    /**
     * @brief Append lexical computation using its explicitly admitted whole-table or fixed-head storage.
     * @param graph Existing participant-local declarative graph.
     * @param context Original query geometry/arithmetic and complete source hash policy.
     * @param weights Original prepared model weights.
     * @param buffers Exact existing PLE views and accepted/prior history.
     * @param storage Checked CPU/hosted/resident whole-table or fixed-head composition; no automatic substitution.
     * @param dependencies Existing original token/trunk/history producers.
     * @return Terminal updated-trunk node, leaving candidate histories to their existing authority.
     * @throws std::exception Before graph mutation for absent producer/cohort, bad buffers or source mismatch.
     */
    static std::string append(ComputeGraph &graph,const Qwen4ExpPLEContext &context,
        const Qwen4ExpPLEWeights &weights,const Qwen4ExpPLEBuffers &buffers,
        LookupStorage storage,const std::vector<std::string> &dependencies,Qwen4ExpEarlyLookupFrame* early=nullptr,
        Qwen4ExpResidentReadAheadFrame* residentEarly=nullptr);
};
} // namespace llaminar2
