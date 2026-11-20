/**
 * @file ROCmGroupedFFNFrontier.h
 * @brief Device-bound publication and launch policy for a counted ROCm verifier FFN.
 */
#pragma once
#include "utils/DebugEnv.h"
#include "kernels/common/DeviceNativeQuantisedMatrixDesc.h"
#include "kernels/common/DeviceRowRange.h"
#include "kernels/common/VerifierDispatchTable.h"
#include "kernels/common/MoEProjectionNumericalContract.h"
#include "tensors/NativeQuantisedFormatInfo.h"
#include "execution/moe/MoEGroupedPlanDemand.h"
#include "backends/DeviceId.h"
#include <array>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <variant>
#include <cstddef>
#include <cstdint>

namespace llaminar2 {
class TensorBase;
class ITensor;
struct DeviceMoELayerRuntime;
class DeviceMoERuntimeTable;
namespace rocm::grouped_ffn {
inline constexpr int capacity=4, hidden=2560, routedWidth=128, sharedWidth=160, topK=10, experts=512;
/** @return The full physical rows qualified by the counted all-format launch. */
inline constexpr bool supportsFormatRows(int rows) noexcept { return rows==capacity || rows==6; }
inline constexpr std::size_t sharedCodesBytes=capacity*sharedWidth;
inline constexpr std::size_t sharedScalesBytes=capacity*(sharedWidth/32)*sizeof(float);
inline constexpr std::size_t sharedGateBytes=capacity*sizeof(float);
/** @brief Cold-selected routed width for the exact Q4_K/Q5_1 stage. */
enum class RoutedGeometry : int { F128=128, F160=160 };
inline constexpr int routedColumns(RoutedGeometry geometry) { return static_cast<int>(geometry); }

/** @brief Physical output pitch of the measured original Q5_K gate/up reader. */
enum class FixedQ5GatePitch : int { F128=128, F160=160, F256=256 };

/** @brief Routed pitch of the original Q5_1 down projection with one K32 packet per partition. */
enum class FixedQ51DownPitch : int { F128=128, F256=256 };

/** @brief Exact immutable reachable routed decoder pair. */
enum class RoutedDecoder { Q4K_Q51, Q4K_Q80, Q5K_Q80 };

/** @brief Reachable physical decoders and original arithmetic policies for each projection.
 * Gate and up share one movable table declaration. The immutable shared
 * projections retain independent identities for gate, up and down.
 */
struct WeightFormats {
    NativeQuantisedExecutionFormatEnvelope routedGateUp;
    NativeQuantisedExecutionFormatEnvelope routedDown;
    std::array<NativeQuantisedExecutionFormatEnvelope,3> shared;

    /** @return Physical union required for immutable per-device decoder tables. */
    constexpr std::uint32_t executionMask() const noexcept
    {
        return routedGateUp.execution_codebook_mask|routedDown.execution_codebook_mask|
            shared[0].execution_codebook_mask|shared[1].execution_codebook_mask|shared[2].execution_codebook_mask;
    }
};

/** @brief Publish immutable lookup tables for every reachable indexed-grid projection before capture.
 * @param formats Complete original role envelopes. @param device Exact admitted ROCm ordinal.
 * @return Whether all required per-device immutable grids are ready in this runtime generation.
 */
bool prepareFormatTables(const WeightFormats& formats,int device);

/** @brief Identify the measured physical formats that benefit from K32 sums. */
constexpr bool supportsExactK32Sums(RoutedDecoder decoder) noexcept
{
    return decoder==RoutedDecoder::Q4K_Q51 || decoder==RoutedDecoder::Q5K_Q80;
}
constexpr unsigned routedGateCodebook(RoutedDecoder decoder)
{
    return decoder==RoutedDecoder::Q5K_Q80?7u:5u;
}
constexpr unsigned routedDownCodebook(RoutedDecoder decoder)
{
    return decoder==RoutedDecoder::Q4K_Q51?7u:19u;
}

/** @brief Producer of affine Q8 row sums for a measured captured contraction. */
enum class AffineSumSource { Inline, RouterPublication };

/** @brief Existing fixed-width native specialization with its measured decoder pair. */
struct MeasuredProgram {
    RoutedGeometry geometry;
    RoutedDecoder decoder;
    AffineSumSource sums=AffineSumSource::Inline;
};

/**
 * @brief A counted verifier program whose source identities come from the original preparation family.
 * Physical execution encodings may change when an expert moves. Its gate, up and down
 * arithmetic identities remain independent and determine the serial reduction order.
 */
class SourceFormatDeclaration {
    friend SourceFormatDeclaration declareSourceFormats(int,int,NativeQuantisedExecutionFormatEnvelope,
        NativeQuantisedExecutionFormatEnvelope,const std::array<NativeQuantisedSourceIdentity,3>&,
        const std::array<DeviceNativeQuantisedMatrixDesc,3>&);
    /** @brief Construct only after every original role has passed the source-envelope checks. */
    SourceFormatDeclaration(int routed,int shared,WeightFormats masks,unsigned decoderFamily,
        const std::array<NativeQuantisedSourceIdentity,3>& original)
        : routedColumns_(routed),sharedColumns_(shared),formats_(masks),family_(decoderFamily),original_(original) {}
    int routedColumns_, sharedColumns_;
    WeightFormats formats_;
    unsigned family_;
    std::array<NativeQuantisedSourceIdentity,3> original_;
public:
    /** @return Physical routed width on the current participant. */
    int routedColumns() const noexcept {return routedColumns_;}
    /** @return Physical shared-expert width on the current participant. */
    int sharedColumns() const noexcept {return sharedColumns_;}
    /** @return Capture-stable envelopes of all physical readers and arithmetic policies. */
    const WeightFormats& formats() const noexcept {return formats_;}
    /** @return Decoder-structure family selected from all five envelopes. */
    unsigned family() const noexcept {return family_;}
    /** @return Original gate, up and down source identities in projection order. */
    const auto& routedSources() const noexcept {return original_;}

    /**
     * @brief Match the original and reachable physical encodings of the four gate/up operands.
     * @param routed Original format of both movable routed projections.
     * @param shared Original format of both immutable shared projections.
     * @param sharedWeights Prepared descriptors already validated by the graph recorder.
     * @return True when the ordered gate reader stays valid through every admitted expert move.
     * Down has a separate launch and cannot change this producer's tile or arithmetic.
     */
    bool hasOriginalGateReaders(const NativeQuantisedFormatInfo& routed,const NativeQuantisedFormatInfo& shared,
        const std::array<DeviceNativeQuantisedMatrixDesc,3>& sharedWeights) const noexcept
    {
        const auto exact=[](NativeQuantisedExecutionFormatEnvelope envelope,unsigned codebook) {
            const auto bit=nativeQuantisedCodebookMaskBit(codebook);
            return envelope.execution_codebook_mask==bit && envelope.policy_codebook_mask==bit;
        };
        const NativeQuantisedSourceIdentity source{routed.codebook_id,routed.is_superblock,true};
        if(!exact(formats().routedGateUp,routed.codebook_id))return false;
        for(unsigned role=0;role<2;++role) {
            const auto& desc=sharedWeights[role];
            if(routedSources()[role]!=source || !exact(formats().shared[role],shared.codebook_id) ||
               !desc.source_identity_present || desc.source_codebook_id!=shared.codebook_id ||
               desc.codebook_id!=shared.codebook_id || bool(desc.source_is_superblock)!=shared.is_superblock)
                return false;
        }
        return true;
    }
};

/** @brief Six immutable serial trees for one captured stage on its admitted ROCm participant. */
class SourceFormatProgram final : public SourceFormatDeclaration {
    std::array<int,3> routedPartitions_,sharedPartitions_;
    /** @brief Only the canonical serial decoder query may issue counts for these sources. */
    SourceFormatProgram(const SourceFormatDeclaration& declaration,
                        std::array<int,3> routed,std::array<int,3> shared)
        : SourceFormatDeclaration(declaration),routedPartitions_(routed),sharedPartitions_(shared) {}
public:
    /**
     * @brief Resolve all six independent source reductions before graph capture.
     * @param declaration Authenticated original roles and their reachable physical representations.
     * @return A program carrying the exact serial-M1 policy for every role.
     * @throws std::invalid_argument If the canonical serial configuration rejects a role.
     */
    static SourceFormatProgram prepare(const SourceFormatDeclaration& declaration);
    /** @return Fixed gate, up and down partition counts for a routed expert. */
    const auto& routedPartitions() const noexcept {return routedPartitions_;}
    /** @return Fixed gate, up and down partition counts for the immutable shared expert. */
    const auto& sharedPartitions() const noexcept {return sharedPartitions_;}
    /**
     * @brief Select the measured separate-reader Q5_K schedule after original source preparation.
     * @param sharedWeights Immutable shared descriptors passed to the existing capture launcher.
     * @return Measured pitch when every reachable role has the fixed physical reader and serial tree.
     */
    std::optional<FixedQ5GatePitch> fixedQ5GatePitch(
        const std::array<DeviceNativeQuantisedMatrixDesc,3>& sharedWeights) const noexcept
    {
        const int pitch=routedColumns();
        if((pitch!=128 && pitch!=160 && pitch!=256) || sharedColumns()!=160)return std::nullopt;
        constexpr auto q5Id=native_quantised_formats::Q5_1.codebook_id;
        constexpr auto q8Id=native_quantised_formats::Q8_0.codebook_id;
        constexpr auto parts=MoEProjectionNumericalContract::ordered_k_partitions;
        constexpr auto q5=nativeQuantisedCodebookMaskBit(q5Id);
        constexpr auto q8=nativeQuantisedCodebookMaskBit(q8Id);
        constexpr NativeQuantisedSourceIdentity q5k{native_quantised_formats::Q5_K.codebook_id,true,true};
        const auto& masks=formats();
        if(masks.routedGateUp.execution_codebook_mask!=q5 || masks.routedGateUp.policy_codebook_mask!=q5)
            return std::nullopt;
        for(unsigned role=0;role<2;++role) {
            const auto& shared=sharedWeights[role];
            if(routedSources()[role]!=q5k || routedPartitions_[role]!=parts ||
               sharedPartitions_[role]!=parts || shared.n!=sharedColumns() || shared.k!=hidden ||
               shared.codebook_id!=q8Id || !shared.source_identity_present ||
               shared.source_codebook_id!=q8Id || shared.source_is_superblock ||
               masks.shared[role].execution_codebook_mask!=q8 ||
               masks.shared[role].policy_codebook_mask!=q8)
                return std::nullopt;
        }
        return static_cast<FixedQ5GatePitch>(pitch);
    }

    /**
     * @brief Bind the paired-packet down reader to the original numerical and placement contract.
     * @param sharedDown Original shared down projection, validated alongside this program.
     * @return The measured routed pitch when every reachable physical reader retains its layout.
     * @details Q5_K and Q5_1 share a physical codebook. Only the plain Q5_1 source with
     * one original K32 packet per down partition has been qualified for this schedule.
     */
    std::optional<FixedQ51DownPitch> fixedQ51DownPitch(const DeviceNativeQuantisedMatrixDesc& sharedDown) const noexcept
    {
        using namespace native_quantised_formats;
        const int pitch=routedColumns();
        if((pitch!=128 && pitch!=256) || sharedColumns()!=sharedWidth) return std::nullopt;
        constexpr NativeQuantisedSourceIdentity gate{Q4_K.codebook_id,true,true};
        constexpr NativeQuantisedSourceIdentity down{Q5_1.codebook_id,false,true};
        const auto singleReader=[](NativeQuantisedExecutionFormatEnvelope formats,unsigned codebook) {
            const auto mask=nativeQuantisedCodebookMaskBit(codebook);
            return formats.execution_codebook_mask==mask && formats.policy_codebook_mask==mask;
        };
        if(routedSources()!=std::array{gate,gate,down} ||
           !singleReader(formats().routedGateUp,Q4_1.codebook_id) ||
           !singleReader(formats().routedDown,Q5_1.codebook_id) ||
           !singleReader(formats().shared[2],Q8_0.codebook_id) ||
           !sharedDown.source_identity_present || sharedDown.source_codebook_id!=Q8_0.codebook_id ||
           sharedDown.source_is_superblock || routedPartitions_[2]!=pitch/32 ||
           sharedPartitions_[2]!=sharedWidth/32) return std::nullopt;
        return static_cast<FixedQ51DownPitch>(pitch);
    }

    /** @return Whether the two immutable reduction trees admit the measured joint Q4_1 reader.
     * The physical Q5_1 reader was slower when paired on original Q5_K TP shards.
     * Check reachable encodings as well as counts so the graph can survive expert movement.
     */
    bool pairsGateAndUp() const noexcept
    {
        const auto q4=nativeQuantisedCodebookMaskBit(native_quantised_formats::Q4_1.codebook_id);
        const auto q8=nativeQuantisedCodebookMaskBit(native_quantised_formats::Q8_0.codebook_id);
        const auto& masks=formats();
        return routedPartitions_[0]==routedPartitions_[1] && sharedPartitions_[0]==sharedPartitions_[1] &&
            masks.routedGateUp.execution_codebook_mask==q4 && masks.shared[0].execution_codebook_mask==q8 &&
            masks.shared[1].execution_codebook_mask==q8;
    }
};

using Program=std::variant<MeasuredProgram,SourceFormatProgram>;

/** @brief Capture receipt joining one program to the original prepared table layer. */
class BoundSourceFormatProgram final {
    Program program_;
    DeviceMoELayerRuntime* runtime_;
    int ordinal_;
    native_quantised::VerifierDispatchTable table_;
    int multiprocessors_;

    /** @brief Bind both source and launch policy to the runtime table's actual owner. */
    BoundSourceFormatProgram(const Program& program,DeviceMoELayerRuntime* runtime,
        const native_quantised::NativeQuantisedGPUHardware& hardware)
        :program_(program),runtime_(runtime),ordinal_(hardware.ordinal),
         table_(native_quantised::selectVerifierDispatchTable(hardware)),multiprocessors_(hardware.compute_units) {}
public:
    /** @brief Bind only through the original owning ROCm runtime table.
     * @throws std::invalid_argument If the layer or its original source disagrees with the program.
     */
    static BoundSourceFormatProgram prepare(DeviceMoERuntimeTable& table,int modelLayer,
        const Program& program,
        const std::array<DeviceNativeQuantisedMatrixDesc,3>& shared={});
    /** @return Whether both addresses and the admitted device identify the original layer. */
    bool matches(const DeviceMoELayerRuntime* runtime,int ordinal) const noexcept
    {
        return runtime_==runtime && ordinal_==ordinal;
    }
    /** @return Kernel scheduling table selected for this captured participant. */
    native_quantised::VerifierDispatchTable dispatchTable() const noexcept { return table_; }
    /** @return Two reusable workgroups per HIP multiprocessor. This is a CU on MI50 and a WGP on RDNA3. */
    std::int64_t reusableGateBlocks() const noexcept { return 2LL * multiprocessors_; }
    /** @return The program admitted by that table's original sources. */
    const Program& program() const noexcept {return program_;}
};

/** @return Participant-local routed intermediate width embedded in the two leaves. */
inline int routedColumns(const Program& program)
{
    if(const auto* measured=std::get_if<MeasuredProgram>(&program))return routedColumns(measured->geometry);
    return std::get<SourceFormatProgram>(program).routedColumns();
}

/** @return Independent participant-local shared intermediate width. */
inline int sharedColumns(const Program& program)
{
    if(const auto* formats=std::get_if<SourceFormatProgram>(&program))return formats->sharedColumns();
    return sharedWidth;
}

/** @return Whether the selected leaves consume the measured F160 router sum layout. */
inline bool usesExactK32Sums(const Program& program)
{
    const auto* measured=std::get_if<MeasuredProgram>(&program);
    return measured && measured->sums==AffineSumSource::RouterPublication;
}

/** @return The gate/up and down table contracts sealed before placement can change. */
inline std::array<NativeQuantisedExecutionFormatEnvelope,2> routedFormats(const Program& program)
{
    if(const auto* formats=std::get_if<SourceFormatProgram>(&program))
        return {formats->formats().routedGateUp,formats->formats().routedDown};
    const auto decoder=std::get<MeasuredProgram>(program).decoder;
    const auto gate=nativeQuantisedCodebookMaskBit(routedGateCodebook(decoder));
    const auto down=nativeQuantisedCodebookMaskBit(routedDownCodebook(decoder));
    return {{{gate,gate},{down,down}}};
}

inline constexpr std::size_t bankIdentityBytes=3*sizeof(std::uint64_t);

/** @brief Disjoint typed tail in the already admitted, serial frontier input scratch.
 * The shared SwiGLU codes occupy [0,640). Router and gate publishers write only their own
 * live K32 sums through the same graph stream, before either consumer can read them.
 */
struct ExactK32Sums {
    std::int16_t hiddenRows[capacity*(hidden/32)];
    std::int16_t routedRows[capacity*topK*(160/32)];
};
static_assert(sharedCodesBytes % alignof(ExactK32Sums)==0);
inline constexpr auto k32SumWorkspaceBytes=sharedCodesBytes+sizeof(ExactK32Sums);

/** @brief Consume a scoped opt-in during graph construction, before capture identity seals. */
inline bool exactK32SumsOptIn(bool qualified_default=false)
{
    static const std::optional<bool> enabled=[]() -> std::optional<bool> {
        const char* mode=DebugEnv::envValue("LLAMINAR_QFN_ROCM_GROUPED_FFN_AFFINE_SUMS");
        if(!mode)return std::nullopt;
        if(std::strcmp(mode,"0")==0)return false;
        if(std::strcmp(mode,"1")==0)return true;
        throw std::invalid_argument("LLAMINAR_QFN_ROCM_GROUPED_FFN_AFFINE_SUMS accepts 0 or 1");
    }();
    return enabled.value_or(qualified_default);
}

/** @brief Form the disjoint typed sum view after the caller validates its lease extent. */
inline ExactK32Sums* exactK32Sums(void* scratch)
{
    return reinterpret_cast<ExactK32Sums*>(static_cast<std::byte*>(scratch)+sharedCodesBytes);
}


/** @brief Router publication handed directly to the qualified two-node transaction. */
struct DirectBankRouter {
    ITensor* indices=nullptr;
    ITensor* weights=nullptr;
    MoEGroupedPlanDemand demand=MoEGroupedPlanDemand::None;
};

/** @brief Original graph-owned source and arena operands, separate from mutable route state. */
struct LocalPublication {
    DeviceRowRange rows;
    std::array<DeviceNativeQuantisedMatrixDesc,3> sharedWeights; ///< Fixed shared gate, up, down.
    TensorBase* sharedInputGate;
    TensorBase* sharedGateScratch;
    TensorBase* sharedUpScratch;
    TensorBase* routedOutput;
    TensorBase* sharedOutput;
    TensorBase* combined;
    DirectBankRouter directRouter{};
    Program program;
    std::optional<BoundSourceFormatProgram> boundSourceProgram;
};

/** @brief Canonical original mutable plan views from the complete grouped-plan owner. */
struct RoutePlan {
    const std::int32_t* live;
    const std::int32_t* originalToGrouped;
    const std::int32_t* originalExpertIds;
    const float* groupWeights;
    const std::int8_t* hiddenCodes;
    const float* hiddenScales;
    const float* hiddenFP32;
    const DeviceNativeQuantisedMatrixDesc* routedGate;
    const DeviceNativeQuantisedMatrixDesc* routedUp;
    const DeviceNativeQuantisedMatrixDesc* routedDown;
};

/** @brief Exact runtime, routing inputs, and device workspace for the acquired bank identity. */
struct DirectBankInput {
    DeviceMoELayerRuntime* runtime=nullptr;
    const float* indices=nullptr;
    const float* weights=nullptr;
    MoEGroupedPlanDemand demand=MoEGroupedPlanDemand::None;
    std::uint64_t* bankIdentity=nullptr;
    ExactK32Sums* affineSums=nullptr; ///< Same stream and graph workspace as the two-role scratch.
};

/** @brief Every native publication is bound to the same original stream and workspace lifetime. */
struct Output {
    std::int8_t* routedCodes;
    float* routedScales;
    std::int8_t* sharedCodes; ///< The original prefill input scratch is dead when router Q8 is selected.
    float* sharedScales; ///< The corresponding original input scale scratch.
    float* gateValues; ///< The original shared gate scratch.
    const float* sharedInputGate;
    float* sharedGateScratch;
    float* sharedUpScratch;
    float* routedOutput;
    float* sharedOutput;
    float* combined;
    float* routeContributions;
};

/**
 * @brief Enqueue the measured paired gate/up publisher and ordered token-local down.
 * @param plan Original device-owned count, compaction and descriptor views.
 * @param sharedWeights Original admitted immutable shared descriptor triple.
 * @param output All original accumulators plus explicitly admitted workspace.
 * @param device Exact admitted ROCm ordinal.
 * @param stream Exact non-default stream following the original plan.
 * @return True when both native submissions succeed.
 */
bool launch(const RoutePlan& plan,const std::array<DeviceNativeQuantisedMatrixDesc,3>& sharedWeights,
            const Output& output,int device,void* stream,const DirectBankInput& direct={},
            RoutedGeometry geometry=RoutedGeometry::F128,RoutedDecoder decoder=RoutedDecoder::Q4K_Q51,
            const BoundSourceFormatProgram* bound=nullptr);

/** @brief Record the counted native gate/up and down leaves for a sealed format declaration.
 * @param plan Original device-owned counts, router rows and reachable expert descriptors.
 * @param rows Original unsliced verifier count owner and physical graph capacity.
 * @param shared Original immutable shared gate, up and down matrices.
 * @param output Existing graph workspace and publication targets.
 * @param device Exact admitted ROCm ordinal.
 * @param stream Exact non-default graph stream.
 * @param direct Original placement-bank acquisition and deferred-demand policy.
 * @param bound Original authenticated runtime, source formats and participant geometry.
 * @return Whether both native submissions succeeded.
 */
bool launchFormatProgram(const RoutePlan& plan,const DeviceRowRange& rows,
    const std::array<DeviceNativeQuantisedMatrixDesc,3>& shared,
    const Output& output,int device,void* stream,const DirectBankInput& direct,const BoundSourceFormatProgram& bound);

} // namespace rocm::grouped_ffn
} // namespace llaminar2
