/** @file NativeAuxiliaryResourceCatalog.h
 * @brief Closed auxiliary native program classes from original prepared-source metadata.
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
    static constexpr NativeAuxiliaryResourceProgram readAheadSide(){return NativeAuxiliaryResourceProgram(Side{});}
    static constexpr NativeAuxiliaryResourceProgram ordering(NativeOrderingEventProgram p){return NativeAuxiliaryResourceProgram(p);}
    /** @return Portable source program only; placement stays in the original instance descriptor. */
    static NativeAuxiliaryResourceProgram pooledRemoteOwner(const PooledRemoteGraphProgram& p){return NativeAuxiliaryResourceProgram(p.qualificationClass());}
    constexpr bool isReadAheadSide()const noexcept{return std::holds_alternative<Side>(value_);}
    constexpr bool isOrdering()const noexcept{return std::holds_alternative<NativeOrderingEventProgram>(value_);}
    constexpr bool isPooledRemoteOwner()const noexcept{return std::holds_alternative<PooledRemoteGraphProgramClass>(value_);}
    const NativeOrderingEventProgram& orderingProgram()const{return std::get<NativeOrderingEventProgram>(value_);}
    const PooledRemoteGraphProgramClass& remoteProgramClass()const{return std::get<PooledRemoteGraphProgramClass>(value_);}
    std::string identity()const;
    constexpr bool operator==(const NativeAuxiliaryResourceProgram&)const=default;
private:
    friend struct detail::NativeAuxiliaryQualifiedClass;
    struct Side {constexpr bool operator==(const Side&)const=default;};
    constexpr explicit NativeAuxiliaryResourceProgram(Side p):value_(p){}
    constexpr explicit NativeAuxiliaryResourceProgram(NativeOrderingEventProgram p):value_(p){}
    constexpr explicit NativeAuxiliaryResourceProgram(PooledRemoteGraphProgramClass p):value_(p){}
    std::variant<Side,NativeOrderingEventProgram,PooledRemoteGraphProgramClass> value_;
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
/** @brief Immutable selected limits, distinct from physical materialization or a live ledger. */
struct NativeAuxiliaryResourceSelection final {
    std::size_t hostBytes,deviceBytes;
    std::string evidence,sourceIdentity,resourceSource;
    NativePhysicalLifetime lifetime;
    std::uint64_t contextGeneration;
    std::optional<NativeAuxiliaryResourceProgram> program; ///< Exact requested capture class, even when a reviewed family row selected it.
    std::optional<NativeAuxiliaryRemoteFamilyScope> remoteFamily={};
};
/** @brief Installed selector consumes the original immutable source key, never queries SDKs or hashes DSOs. */
class NativeAuxiliaryResourceCatalog final {
public:
    static NativeAuxiliaryResourceSelection require(DeviceId,NativeAuxiliaryResourceProgram);
    /** @return Existing portable native runtime/hardware/count class; application source text is not an input. */
    static std::string portableSourceIdentity(const NativeContextBirthKey&);
    static std::string liveSourceIdentity(const NativeContextBirthKey&);
    /** @return Named native operation family, independent of application source revisions. */
    static const std::string& resourceSourceIdentity();
private:
    friend struct test::NativeOrderingEventGroupReplayAccess;
    friend struct test::PooledRemoteGraphHostReplayAccess;
    static NativeAuxiliaryResourceSelection select(const NativeContextBirthKey&,std::uint64_t,NativeAuxiliaryResourceProgram,std::string_view,std::span<const detail::NativeAuxiliaryQualifiedClass>);
};
}
