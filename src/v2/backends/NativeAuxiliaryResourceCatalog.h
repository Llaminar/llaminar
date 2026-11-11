/** @file NativeAuxiliaryResourceCatalog.h
 * @brief One installed auxiliary allowance selector with separate original live creation authority.
 */
#pragma once
#include "NativeOrderingEventProgram.h"
#include "NativeContextOpaqueContract.h"
#include "NativeContextRetirement.h"
#include "PooledRemoteGraphProgram.h"
#include <variant>
#include <optional>
#include <span>
#include <string>
#include <string_view>
namespace llaminar2 {
namespace test {struct NativeOrderingEventGroupReplayAccess;struct PooledRemoteGraphHostReplayAccess;}
/** @brief Exact fixed auxiliary program, never a caller-authored event count. */
class NativeAuxiliaryResourceProgram final {
public:
    /** @return Exact peer-lane completion event and optional two timestamp events. */
    static constexpr NativeAuxiliaryResourceProgram peerTransferEvents(bool timing){return NativeAuxiliaryResourceProgram(PeerEvents{timing});}
    constexpr bool isPeerTransferEvents()const noexcept{return std::holds_alternative<PeerEvents>(value_);}
    constexpr bool peerTransferTiming()const{return std::get<PeerEvents>(value_).timing;}
    /** @return One auxiliary stream with immutable scheduling class, without fictitious events. */
    static constexpr NativeAuxiliaryResourceProgram auxiliaryStream(){return NativeAuxiliaryResourceProgram(Stream{});}
    static constexpr NativeAuxiliaryResourceProgram readAheadSide(){return NativeAuxiliaryResourceProgram(Side{});}
    static constexpr NativeAuxiliaryResourceProgram ordering(NativeOrderingEventProgram p){return NativeAuxiliaryResourceProgram(p);}
    /** @return Portable source program only; placement stays in the original instance descriptor. */
    static NativeAuxiliaryResourceProgram pooledRemoteOwner(const PooledRemoteGraphProgram& p){return NativeAuxiliaryResourceProgram(p.qualificationClass());}
    constexpr bool isAuxiliaryStream()const noexcept{return std::holds_alternative<Stream>(value_);}
    constexpr bool isReadAheadSide()const noexcept{return std::holds_alternative<Side>(value_);}
    constexpr bool isOrdering()const noexcept{return std::holds_alternative<NativeOrderingEventProgram>(value_);}
    constexpr bool isPooledRemoteOwner()const noexcept{return std::holds_alternative<PooledRemoteGraphProgramClass>(value_);}
    const NativeOrderingEventProgram& orderingProgram()const{return std::get<NativeOrderingEventProgram>(value_);}
    const PooledRemoteGraphProgramClass& remoteProgramClass()const{return std::get<PooledRemoteGraphProgramClass>(value_);}
    std::string identity()const;
    constexpr bool operator==(const NativeAuxiliaryResourceProgram&)const=default;
private:
    friend struct detail::NativeAuxiliaryQualifiedClass;
    struct PeerEvents {bool timing;constexpr bool operator==(const PeerEvents&)const=default;};
    constexpr explicit NativeAuxiliaryResourceProgram(PeerEvents p):value_(p){}
    struct Stream {constexpr bool operator==(const Stream&)const=default;};
    constexpr explicit NativeAuxiliaryResourceProgram(Stream p):value_(p){}
    struct Side {constexpr bool operator==(const Side&)const=default;};
    constexpr explicit NativeAuxiliaryResourceProgram(Side p):value_(p){}
    constexpr explicit NativeAuxiliaryResourceProgram(NativeOrderingEventProgram p):value_(p){}
    constexpr explicit NativeAuxiliaryResourceProgram(PooledRemoteGraphProgramClass p):value_(p){}
    std::variant<Stream,Side,NativeOrderingEventProgram,PooledRemoteGraphProgramClass,PeerEvents> value_;
};
/** @brief Reviewed versioned family envelope; exact geometry remains in the separate capture identity. */
struct NativeAuxiliaryRemoteFamilyScope final {
    std::uint32_t version=0;GPUFlatGraphLimits native;
    PooledRemoteGraphArgumentLayout arguments;
    /** @return Whether this declares the sole supported remote kernel-chain/argument ABI family. */
    constexpr bool valid()const noexcept{
        return version==PooledRemoteGraphProgramClass::familyVersion() && native.valid() &&
            native.allowedKinds==GPUFlatGraphLimits::bit(GPUFlatGraphNodeKind::Kernel) &&
            std::uint64_t{native.maximumEdges}+1>=native.maximumNodes && arguments.count && arguments.bytes &&
            arguments.alignment && !(arguments.alignment&(arguments.alignment-1));
    }
    /** @return Source-authored native geometry/compiled arguments are a valid supported kernel chain; empirical maxima are evidence only. */
    constexpr bool accepts(const GPUFlatGraphLimits& requested,const PooledRemoteGraphArgumentLayout& abi)const noexcept{
        return valid() && requested.valid() && requested.allowedKinds==native.allowedKinds &&
            std::uint64_t{requested.maximumEdges}+1==requested.maximumNodes && abi.count && abi.bytes && abi.alignment && !(abi.alignment&(abi.alignment-1));
    }
    /** @return Stable reviewed scope metadata, never an allowance derived from live resource observations. */
    std::string identity()const;
};
namespace detail {
/** @brief Supported native backend and typed resource family; exact runtime/provider metadata remains evidence. */
struct NativeAuxiliaryQualifiedClass final {
    /** @brief Representative row metadata only; cannot mint capture geometry or ownership. */
    static constexpr NativeAuxiliaryResourceProgram reviewedRemoteProgram(std::array<std::uint8_t,32> dispatch,GPUFlatGraphLimits limits){
        return NativeAuxiliaryResourceProgram(PooledRemoteGraphProgramClass(dispatch,limits));
    }

    DeviceType backend; ///< Actual supported native backend; software build metadata is not an allowance gate.
    NativeAuxiliaryResourceProgram program;
    std::size_t hostBytes,deviceBytes; ///< Remote hostBytes is the conservative per1024-declared-node unit.
    std::string_view evidence;
    NativePhysicalLifetime lifetime;
    std::optional<NativeAuxiliaryRemoteFamilyScope> remoteFamily={};
};
}
/**
 * @brief Cold supported-backend/program allowance; never a live source or creation capability.
 *
 * Cross-rank BOM validation may price the original remote backend without a
 * local worker. No generation, source identity or native handle is represented;
 * original live selection and PMA validation remain mandatory at materialization.
 */
struct NativeAuxiliaryResourceAllowance final {
    std::size_t hostBytes,deviceBytes;
    std::string evidence;
    NativePhysicalLifetime lifetime;
    NativeAuxiliaryResourceProgram program;
    std::optional<NativeAuxiliaryRemoteFamilyScope> remoteFamily={};
};
/** @brief Immutable selected limits, distinct from physical materialization or a live ledger. */
struct NativeAuxiliaryResourceSelection final {
    std::size_t hostBytes,deviceBytes;
    std::string evidence,sourceIdentity,resourceSource;
    NativePhysicalLifetime lifetime;
    std::uint64_t contextGeneration;
    std::optional<NativeAuxiliaryResourceProgram> program; ///< Exact requested capture class, even when a reviewed family row selected it.
    std::optional<NativeAuxiliaryRemoteFamilyScope> remoteFamily={};
};
/** @brief Cold BOM uses installed backend/program bounds; live creation separately authenticates original immutable source. */
class NativeAuxiliaryResourceCatalog final {
public:
    /**
     * @brief Price the installed supported family without local native state.
     * @param backend Exact original observed GPU backend, never a guessed local worker.
     * @param program Closed immutable operation family and declared geometry.
     * @return Limits/evidence/lifetime only; no live generation or source capability.
     * @throws std::invalid_argument For a non-GPU backend.
     * @throws std::runtime_error When no reviewed installed family accepts the program.
     */
    static NativeAuxiliaryResourceAllowance allowance(DeviceType,NativeAuxiliaryResourceProgram);
    /** @brief Authenticate the original live local worker/generation/source for actual creation. */
    static NativeAuxiliaryResourceSelection require(DeviceId,NativeAuxiliaryResourceProgram);
    /** @return Existing portable native runtime/hardware/count class; application source text is not an input. */
    static std::string portableSourceIdentity(const NativeContextBirthKey&);
    static std::string liveSourceIdentity(const NativeContextBirthKey&);
    /** @return Named native operation family, independent of application source revisions. */
    static const std::string& resourceSourceIdentity();
private:
    friend struct test::NativeOrderingEventGroupReplayAccess;
    friend struct test::PooledRemoteGraphHostReplayAccess;
    /**
     * @brief Share row validation between cold BOM and original live creation.
     * @param backend Original resource backend, independent of the receiving rank.
     * @param program Operation family and its admitted geometry.
     * @param rows Installed limits or the explicit replay fixture's immutable rows.
     * @return Matching limits without a live context identity or creation permit.
     * @throws std::exception For invalid families or an unsupported program.
     */
    static NativeAuxiliaryResourceAllowance selectAllowance(DeviceType,NativeAuxiliaryResourceProgram,std::span<const detail::NativeAuxiliaryQualifiedClass>);
    static NativeAuxiliaryResourceSelection select(const NativeContextBirthKey&,std::uint64_t,NativeAuxiliaryResourceProgram,std::string_view,std::span<const detail::NativeAuxiliaryQualifiedClass>);
};
}
