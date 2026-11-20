/**
 * @file ROCmResidentMoEFFN.h
 * @brief Borrowed persistent HIP arguments for one original local FFN transaction.
 *
 * The backend preparation hook alone creates these bindings from its private
 * table/workspace owners. Original partition planes are reused after their last
 * consumer; no new live memory ledger, weight copy or route mirror is created.
 */
#pragma once

#include "kernels/common/DeviceRowRange.h"
#include "kernels/common/NativeCooperativeKernel.h"
#include "kernels/common/ResidentMoEFFNProgramEvidence.h"
#include <optional>
#include "kernels/common/DeviceNativeQuantisedMatrixDesc.h"
#include "tensors/NativeQuantisedFormatInfo.h"
#include "execution/moe/MoEIntermediateShardDescriptorDeclaration.h"
#include <array>
#include <limits>
#include <stdexcept>
#include <type_traits>

#if defined(__HIPCC__) || defined(__CUDACC__)
#define LLAMINAR_RESIDENT_FFN_HD __host__ __device__
#else
#define LLAMINAR_RESIDENT_FFN_HD
#endif

namespace llaminar2 {struct DeviceMoELayerRuntime;}
namespace llaminar2::rocm
{
struct NativeLayoutRouterFFNArgs;
struct FullNativeHCFFNArgs;
/** @brief Explicit cold diagnostic program selection; retained native identities never change. */
enum class ResidentMoEFFNProgram { Original, PartitionLocal, OrdinaryPhases, NativeLayoutRouter };
/** @return This worker's cold program selection, never a live execution decision. */
ResidentMoEFFNProgram currentResidentMoEFFNProgram() noexcept;
/** @brief Select a new prepared owner's structural diagnostic before capture. */
class ScopedResidentMoEFFNProgram final
{
public:
    /** @param program Exact implemented program. @throws std::logic_error During capture.
     * @throws std::invalid_argument For an unknown program. */
    explicit ScopedResidentMoEFFNProgram(ResidentMoEFFNProgram program);
    /** @brief Restore prior cold selection; already retained graphs keep their sealed native function. */
    ~ScopedResidentMoEFFNProgram();
    ScopedResidentMoEFFNProgram(const ScopedResidentMoEFFNProgram&)=delete;
    ScopedResidentMoEFFNProgram& operator=(const ScopedResidentMoEFFNProgram&)=delete;
private:
    ResidentMoEFFNProgram previous_;
};

/** @brief Decoder structures bound into one finite family, independent of model/source residency. */
enum class ResidentMoEFFNDecoderStructure : std::uint8_t
{
    AffineAndByte=1, ///< Q4/Q5 affine payloads, signed Q8 and expanded INT8/min.
    LinearPackedAndSubscale=2, ///< Linear packed Q4/Q5 and Q2/Q3/Q6 subscale payloads.
    IQ3AndIQ4Grid=4, ///< IQ3/IQ4 lookup grids, including original IQ4_XS metadata.
    IQ1AndIQ2Grid=8 ///< IQ1/IQ2 low-bit lookup grids and correction metadata.
};

/** @brief Exact original physical decoders reachable by each immutable projection declaration. */
struct ResidentMoEFFNExecutionMasks
{
    std::uint32_t routed_gate=0,routed_down=0,shared_gate=0,shared_down=0;
    /** @return Complete reachable physical union, never the source-policy mask or current residency. */
    constexpr std::uint32_t combined() const noexcept
    {return routed_gate|routed_down|shared_gate|shared_down;}
};

/**
 * @brief Sealed smallest decoder-structure union covering every reachable original format.
 *
 * The fifteen nonempty unions are explicit first-class compiled programs. The
 * complete union remains a supported program with unqualified economy, never
 * an automatic recovery after another admission fails. Descriptor movement
 * cannot expand the captured source envelope or change the chosen program.
 */
class ResidentMoEFFNDecoderProgram final
{
public:
    static constexpr std::uint32_t affine_and_byte=(1u<<5)|(1u<<7)|(1u<<19)|
        (1u<<kNativeQuantisedExpandedInt8MinCodebook);
    static constexpr std::uint32_t linear_packed_and_subscale=(1u<<0)|(1u<<6)|(1u<<8)|(1u<<9)|(1u<<10);
    static constexpr std::uint32_t iq3_and_iq4_grid=(1u<<4)|(1u<<11)|(1u<<12)|
        (1u<<kNativeQuantisedIQ4XSSourceScaleCodebook)|(1u<<kNativeQuantisedMXFP4Codebook);
    static constexpr std::uint32_t iq1_and_iq2_grid=(1u<<13)|(1u<<14)|(1u<<15)|(1u<<16)|(1u<<17);
    static constexpr std::uint32_t all_decoders=affine_and_byte|linear_packed_and_subscale|iq3_and_iq4_grid|iq1_and_iq2_grid;

    /**
     * The linear | iq3/iq4 | iq1/iq2 union compiles without MXFP4: its decode
     * arm takes that union's router kernel one register over the gfx906
     * budget. An envelope needing MXFP4 with exactly these groups is covered
     * by the complete union instead.
     */
    static constexpr unsigned mxfp4_excluded_groups=2u|4u|8u;

    /**
     * @param groups Nonempty union of the four decoder-structure bits.
     * @return Physical decoder mask compiled for that family.
     */
    static constexpr std::uint32_t familyMaskFor(unsigned groups) noexcept
    {
        const std::uint32_t mask=((groups&1)?affine_and_byte:0)|((groups&2)?linear_packed_and_subscale:0)|
            ((groups&4)?iq3_and_iq4_grid:0)|((groups&8)?iq1_and_iq2_grid:0);
        return groups==mxfp4_excluded_groups?mask&~(1u<<kNativeQuantisedMXFP4Codebook):mask;
    }

    /**
     * @brief Compute one explicit compiled family mask at compile time.
     * @tparam Groups Nonempty union of the four decoder-structure bits.
     * @return Physical decoder mask instantiated by this family.
     */
    template<unsigned Groups> static constexpr std::uint32_t familyMask() noexcept
    {
        static_assert(Groups>0&&Groups<16);
        return familyMaskFor(Groups);
    }
    /**
     * @brief Select the unique smallest group union from complete immutable execution envelopes.
     * @param required Every physical decoder reachable by all routed/shared projections.
     * @return Original required mask and its sealed compiled group union.
     * @throws std::invalid_argument For an empty mask or an unsupported physical decoder.
     */
    static ResidentMoEFFNDecoderProgram fromExecutionMask(std::uint32_t required)
    {
        if(!required||(required&~all_decoders))
            throw std::invalid_argument("Resident FFN decoder registry requires a complete supported execution envelope");
        auto groups=static_cast<std::uint8_t>((required&affine_and_byte?1:0)|
            (required&linear_packed_and_subscale?2:0)|(required&iq3_and_iq4_grid?4:0)|(required&iq1_and_iq2_grid?8:0));
        // Smallest compiled family covering the envelope: the group union, or
        // the complete union where that family omits a required decoder.
        if(required&~familyMaskFor(groups))groups=15;
        return ResidentMoEFFNDecoderProgram(required,groups);
    }
    /** @return Original exact required physical envelope, including authenticated promotions. */
    std::uint32_t requiredMask() const noexcept {return required_;}
    /** @return Nonempty decoder-structure union selecting one of fifteen compiled families. */
    std::uint8_t groups() const noexcept {return groups_;}
    /** @return Superset compiled for this program; exact role envelopes remain independently enforced. */
    std::uint32_t executionMask() const noexcept {return familyMaskFor(groups_);}
private:
    /** @brief Retain only the validated immutable physical envelope and its unique group cover. */
    ResidentMoEFFNDecoderProgram(std::uint32_t required,std::uint8_t groups):required_(required),groups_(groups) {}
    std::uint32_t required_;
    std::uint8_t groups_;
};

/** @brief Check exact physical role membership without admitting a wider compiled family. */
LLAMINAR_RESIDENT_FFN_HD constexpr bool residentMoEFFNDecoderAllows(std::uint32_t execution,unsigned codebook) noexcept
{return codebook<32&&((execution>>codebook)&1u)!=0;}

/** @brief Exact arithmetic-equivalent scheduler specialization selected at cold admission. */
enum class ResidentMoEFFNSchedulerWidth { UInt32, UInt64 };
/**
 * @brief Choose an exact phase-local integer scheduler without narrowing tensor byte addresses.
 * @param tasks Maximum useful physical CTA tasks, calculated from immutable original source geometry.
 * @return UInt32 through UINT32_MAX, otherwise the same UInt64 scheduling algorithm.
 * @throws std::invalid_argument For an absent task extent.
 * @note Overflow-safe cursor advancement allows the final extent itself to be
 * UINT32_MAX. No rounded four-team stride exists in the partition-local program.
 */
constexpr ResidentMoEFFNSchedulerWidth residentMoEFFNSchedulerWidthForExtent(std::uint64_t tasks)
{
    if(!tasks)throw std::invalid_argument("Resident FFN requires a positive scheduler task extent");
    return tasks<=std::numeric_limits<std::uint32_t>::max()
        ?ResidentMoEFFNSchedulerWidth::UInt32:ResidentMoEFFNSchedulerWidth::UInt64;
}
/**
 * @brief Name the exact compiled family/width entry without querying a device or admitting resources.
 * @param program Immutable source-envelope decoder program.
 * @param width Original geometry's checked scheduler width.
 * @return Native function identity used by both admission and retained-node authentication.
 * @throws std::invalid_argument For an invalid width or corrupted program identity.
 */
const void* residentMoEFFNKernelFunction(ResidentMoEFFNDecoderProgram program,ResidentMoEFFNSchedulerWidth width);

/**
 * @brief Cold task extent for four original 64-column teams per physical 256-thread CTA.
 *
 * Only scheduler indices may narrow. Tensor offsets, original K partitions and
 * arena strides retain their original authority and arithmetic. The rounded
 * worker stride must fit too, so a near-UINT32_MAX extent selects the identical
 * uint64 algorithm rather than imposing a new geometry limit.
 */
class ResidentMoEFFNTaskGeometry final
{
public:
    /**
     * @brief Validate a projection phase without reading device-owned routes or live rows.
     * @param columns Original participant-local output width.
     * @param routes Admitted route capacity, one for the shared branch.
     * @param partitions Maximum original source-policy partition count.
     * @return Immutable logical task extent for this phase.
     * @throws std::invalid_argument For a zero axis.
     * @throws std::overflow_error If the task product exceeds uint64.
     */
    static ResidentMoEFFNTaskGeometry projection(std::uint64_t columns,std::uint64_t routes,
        std::uint64_t partitions)
    {
        if(!columns||!routes||!partitions)
            throw std::invalid_argument("Resident FFN requires positive task geometry");
        const auto tiles=columns/64+(columns%64!=0);
        if(routes>std::numeric_limits<std::uint64_t>::max()/tiles||
            partitions>std::numeric_limits<std::uint64_t>::max()/(tiles*routes))
            throw std::overflow_error("Resident FFN logical task extent exceeds uint64");
        return ResidentMoEFFNTaskGeometry(tiles*routes*partitions);
    }
    /** @return Exact logical64-column task count, including original policy partitions. */
    std::uint64_t tasks() const noexcept {return tasks_;}
    /** @return Whether both task extent and four-team rounded stride fit uint32. */
    bool uses32BitScheduler() const noexcept
    {return tasks_<=std::uint64_t(std::numeric_limits<std::uint32_t>::max()/4)*4;}
    /**
     * @param maximumResidentGrid Exact loaded-program cooperative occupancy ceiling.
     * @return Work-sized physical CTA count intersected with exact cooperative residency.
     * @throws std::invalid_argument For absent cooperative residency.
     */
    unsigned grid(unsigned maximumResidentGrid) const
    {
        if(!maximumResidentGrid)
            throw std::invalid_argument("Resident FFN requires nonzero admitted residency");
        const auto useful=tasks_/4+(tasks_%4!=0);
        return static_cast<unsigned>(useful<maximumResidentGrid?useful:maximumResidentGrid);
    }
private:
    /** @brief Seal a nonzero overflow-checked phase extent. */
    explicit ResidentMoEFFNTaskGeometry(std::uint64_t tasks):tasks_(tasks) {}
    std::uint64_t tasks_;
};

/** @brief Original tile/route/partition ordering decoded from one independent team task. */
template<class Index> struct ResidentMoEFFNTaskCoordinates
{Index tile,route,partition;};

/**
 * @brief Decode the production scheduler without changing any arithmetic partition boundary.
 * @tparam Index Cold-selected unsigned scheduler width.
 * @param task Active phase-local logical task.
 * @param tiles Positive original 64-column tile count.
 * @param routes Positive admitted route count.
 * @return Original tile-fastest, route-next, partition-last coordinates.
 */
template<class Index>
LLAMINAR_RESIDENT_FFN_HD constexpr ResidentMoEFFNTaskCoordinates<Index>
residentMoEFFNTaskCoordinates(Index task,Index tiles,Index routes)
{
    static_assert(std::is_unsigned_v<Index>);
    const Index row=task/tiles;
    return {task%tiles,row%routes,row/routes};
}

/**
 * @brief Advance a validated positive-stride task cursor without terminal addition overflow.
 * @tparam Index Cold-selected unsigned scheduler width.
 * @param task Current active task, strictly less than end.
 * @param end Exact phase extent admitted by the cold geometry.
 * @param stride Four teams times the admitted physical resident grid.
 * @return Next active task, or end as an absorbing retired cursor.
 */
template<class Index>
LLAMINAR_RESIDENT_FFN_HD constexpr Index residentMoEFFNNextTask(Index task,Index end,Index stride)
{
    static_assert(std::is_unsigned_v<Index>);
    return task>=end||end-task<=stride?end:task+stride;
}

/** @brief Original source-policy partition table, copied as scalar kernargs. */
struct ResidentMoEPartitionPolicy
{
    std::uint64_t p0=0,p1=0,p2=0,p3=0;
    unsigned maximum=0;
};
/** @brief Sealed borrowed device geometry, excluding every live host routing observation. */
struct ResidentMoEFFNArgs
{
    DeviceMoELayerRuntime* runtime=nullptr; ///< Original device authority; fused routing publishes through it.
    DeviceMoEIntermediateShardBinding original;
    const float* hidden=nullptr;
    DeviceRowRange rows=DeviceRowRange::fullyActive(1);
    const DeviceNativeQuantisedMatrixDesc *gates=nullptr,*ups=nullptr,*downs=nullptr;
    const DeviceNativeQuantisedMatrixDesc *shared_gate=nullptr,*shared_up=nullptr,*shared_down=nullptr;
        ///< Original immutable shared descriptors in admitted grouped table slots; never addresses into kernargs.
    const float* shared_input_gate=nullptr;
    int hidden_columns=0,routed_columns=0,shared_columns=0,experts=0,top_k=0;
    ResidentMoEFFNExecutionMasks execution; ///< Exact original reachable physical role envelopes, sealed in capture identity.
    unsigned resident_grid=0; ///< Cold workload ceiling intersected with exact native residency; part of capture identity.
    std::int8_t* hidden_q8=nullptr;
    float* hidden_scales=nullptr;
    float *gate_partials=nullptr,*up_partials=nullptr,*down_partials=nullptr;
    std::int8_t* swiglu_q8=nullptr;
    float* swiglu_scales=nullptr;
    float *route_contributions=nullptr,*routed_output=nullptr,*shared_output=nullptr,*combined_output=nullptr;
    float *shared_gate_output=nullptr,*shared_up_output=nullptr;
    ResidentMoEPartitionPolicy routed_gate_policy,routed_down_policy,shared_gate_policy,shared_down_policy;
};
/** @brief Original immutable source masks required to resolve all reachable descriptor policies. */
struct ResidentMoEFFNPolicyMasks
{
    std::uint32_t routed_gate=0,routed_down=0,shared_gate=0,shared_down=0;
    ResidentMoEFFNExecutionMasks execution; ///< Independent physical decoder authority; never source arithmetic policy bits.
};
/**
 * @brief Sealed native program; ordinary phases never carry a fabricated cooperative admission.
 * @details Existing backend ownership retains the source/arena leases. This
 * value contains immutable launch/resource observations and, only for actual
 * cooperative execution, its exact native residency admission.
 */
enum class RetainedM1NativeHistogramPolicy : std::uint32_t;
class PreparedMoEFFNProgram final
{
public:
    /** @return Exact typed submission mechanism, entries and cold resource evidence. */
    const ResidentMoEFFNProgramEvidence& evidence() const noexcept {return evidence_;}
private:
    friend PreparedMoEFFNProgram prepareFullNativeHCFFN(DeviceId,FullNativeHCFFNArgs&,ResidentMoEFFNPolicyMasks,std::uint64_t);
    friend PreparedMoEFFNProgram prepareRetainedM1HCFFN(DeviceId,FullNativeHCFFNArgs&,ResidentMoEFFNPolicyMasks,
        std::uint64_t,RetainedM1NativeHistogramPolicy);
    friend void launchFullNativeHCFFN(const PreparedMoEFFNProgram&,const FullNativeHCFFNArgs&,void*,std::uint64_t);
    friend PreparedMoEFFNProgram prepareNativeLayoutRouterFFN(DeviceId,NativeLayoutRouterFFNArgs&,ResidentMoEFFNPolicyMasks,std::uint64_t);
    friend void launchNativeLayoutRouterFFN(const PreparedMoEFFNProgram&,const NativeLayoutRouterFFNArgs&,void*,std::uint64_t);
    friend PreparedMoEFFNProgram prepareResidentMoEFFN(DeviceId,ResidentMoEFFNArgs&,ResidentMoEFFNPolicyMasks,std::uint64_t);
    friend void launchResidentMoEFFN(const PreparedMoEFFNProgram&,const ResidentMoEFFNArgs&,void*,std::uint64_t);
    /** @brief Seal complete observations and reject inconsistent mechanism/resource identities. */
    PreparedMoEFFNProgram(ResidentMoEFFNProgramEvidence evidence,
        std::optional<NativeCooperativeKernelAdmission> cooperative);
    ResidentMoEFFNProgramEvidence evidence_;
    std::optional<NativeCooperativeKernelAdmission> cooperative_;
};
/**
 * @param program Authenticated original execution envelope, including admitted future representations.
 * @return Exact ordinary gate/up and down entry identities, without device queries.
 * @note The two entries emit the complete existing native decoder inventory;
 * independent per-role execution masks remain enforced at every descriptor.
 */
std::array<const void*,2> ordinaryMoEFFNKernelFunctions(ResidentMoEFFNDecoderProgram program);

/**
 * @brief Resolve original generated policies and exact native resources before capture.
 * @param device Original worker's ROCm participant.
 * @param args Validated persistent bindings; receives original policy tables and workload-sized resident grid.
 * @param masks Authenticated source-policy reachability, including future same-source representations.
 * @param generation Existing original workspace generation, required nonzero.
 * @return Sealed cooperative or two-ordinary-phase program; unsupported declarations fail before native work.
 */
PreparedMoEFFNProgram prepareResidentMoEFFN(DeviceId device,ResidentMoEFFNArgs& args,
    ResidentMoEFFNPolicyMasks masks,std::uint64_t generation);
/** @brief Submit the exact sealed ordered native entries on their prepared stream. */
void launchResidentMoEFFN(const PreparedMoEFFNProgram& program,const ResidentMoEFFNArgs& args,
    void* stream,std::uint64_t generation);
}
#undef LLAMINAR_RESIDENT_FFN_HD
