/**
 * @file PooledRemoteGraphProgram.h
 * @brief Closed source-authored mapped-flat and native-General storage-owner contracts.
 *
 * Only the actual directed-owner builder authors this descriptor. Its portable
 * dispatch signature covers the ordered native launch/publication recipe and
 * argument ABI. Placement/geometry remain in the instance capture identity.
 * No API accepts caller resource limits, node ceilings or a native graph handle.
 */
#pragma once
#include "GPUFlatGraphStructure.h"
#include "planning/PhysicalMemoryAuthority.h"
#include <array>
#include <cstdint>
#include <string>
#include <optional>
#include <variant>
#include "PeerAccessCoverage.h"
namespace llaminar2 {
namespace qwen4exp {class PooledImmutableDirectedOwnerProgram;}
namespace detail {struct NativeAuxiliaryQualifiedClass;}
/** @brief Complete immutable physical geometry; live row counts and execution state are excluded. */
struct PooledRemoteGraphGeometry final {
    PhysicalMemoryAllocatorIdentity coordinator,owner;
    std::uint32_t coordinate=0,participants=0,capacity=0,maximumOriginalIds=0,maximumUniqueIds=0;
    std::uint64_t sourceRows=0,rowBytes=0,slotStride=0,nativeLayoutIdentity=0;
    std::uint32_t rowKeyBits=0,nonemptyRootMask=0;
    /** Original explicit None/Complete payload selection, when endpoints share a vendor. */
    std::optional<PeerAccessCoverage> peerAccess;
    bool operator==(const PooledRemoteGraphGeometry&)const=default;
};
/** @brief Compiled argument envelope of this closed family; no SDK introspection or resource allowance. */
struct PooledRemoteGraphArgumentLayout final {
    std::uint32_t count=0;std::size_t bytes=0,alignment=0;
    bool operator==(const PooledRemoteGraphArgumentLayout&)const=default;
};
/** @brief Immutable portable compiled program class; this metadata grants no materialization authority. */
class PooledRemoteGraphProgramClass final {
public:
    /** @return The closed source family schema, independent of per-capture geometry. */
    static constexpr std::uint32_t familyVersion()noexcept{return 1;}
    /** @return Actual compiled owner/transfer argument envelope; creates no native resource. */
    static PooledRemoteGraphArgumentLayout argumentLayout()noexcept;
    /** @return Source recipe/ABI digest, excluding actual ordinals, UUIDs and pointers. */
    constexpr const std::array<std::uint8_t,32>& dispatchSignature()const noexcept{return dispatch_;}
    /** @return Source-derived cold scratch/native-kind envelope. */
    constexpr const GPUFlatGraphLimits& limits()const noexcept{return limits_;}
    /** @return Stable complete source-class metadata, never a file/provider hash operation. */
    std::string identity()const;
    constexpr bool operator==(const PooledRemoteGraphProgramClass&)const=default;
private:
    friend class PooledRemoteGraphProgram;
    friend struct detail::NativeAuxiliaryQualifiedClass;
    /** @brief Build only as a source descriptor projection or reviewed immutable catalog data. */
    constexpr PooledRemoteGraphProgramClass(std::array<std::uint8_t,32> dispatch,GPUFlatGraphLimits limits)
        :dispatch_(dispatch),limits_(limits){}
    std::array<std::uint8_t,32> dispatch_;
    GPUFlatGraphLimits limits_;
};
/** @brief One exact explicitly routed remote-owner definition, authored before original admission. */
class PooledRemoteGraphProgram final {
public:
    /** @return Complete original placement/geometry used by the actual owner builder. */
    const PooledRemoteGraphGeometry& geometry()const noexcept{return geometry_;}
    /** @return Portable source class consumed by the existing auxiliary selector. */
    const PooledRemoteGraphProgramClass& qualificationClass()const noexcept{return class_;}
    /** @return Same pure scope used for planning scratch and actual native inspection. */
    const GPUFlatGraphLimits& limits()const noexcept{return class_.limits();}
    /** @return Capture-stable identity of every geometry, placement and source-class field. */
    std::uint64_t captureIdentity()const noexcept;
    bool operator==(const PooledRemoteGraphProgram&)const=default;
private:
    friend class qwen4exp::PooledImmutableDirectedOwnerProgram;
    /**
     * @brief Author through the actual directed builder/shared native launch plans only.
     * @param geometry Exact original physical/native layout, without table/generation placeholders.
     * @param dispatchSignature SHA256 of ordered per-role dispatch, publication and argument ABI.
     * @param limits Node/kind/edge bounds derived by that same source recipe, never user input.
     * @throws std::invalid_argument For incomplete or unsupported geometry/scope.
     */
    PooledRemoteGraphProgram(PooledRemoteGraphGeometry geometry,
        std::array<std::uint8_t,32> dispatchSignature,GPUFlatGraphLimits limits);
    PooledRemoteGraphGeometry geometry_;
    PooledRemoteGraphProgramClass class_;
};
/**
 * @brief Closed native-library storage-owner semantics under existing General admission.
 *
 * The source builder alone creates this declaration. It has no portable flat
 * limits, installed mapped-host class or user-supplied resource allowance.
 */
class PooledNativeOwnerGraphProgram final {
public:
    /** @return Complete original source/owner placement and immutable protocol geometry. */
    const PooledRemoteGraphGeometry& geometry()const noexcept{return geometry_;}
    /** @return Compiled semantic recipe identity, never native-library kernel counts. */
    const std::array<std::uint8_t,32>& semanticSignature()const noexcept{return semantic_;}
    /** @return Immutable source geometry and semantic identity for serial-slot reuse. */
    std::uint64_t captureIdentity()const noexcept;
    bool operator==(const PooledNativeOwnerGraphProgram&)const=default;
private:
    friend class qwen4exp::PooledImmutableDirectedOwnerProgram;
    /** @brief Only the source-authored native recipe constructs a General storage declaration. */
    PooledNativeOwnerGraphProgram(PooledRemoteGraphGeometry,std::array<std::uint8_t,32>);
    PooledRemoteGraphGeometry geometry_;
    std::array<std::uint8_t,32> semantic_;
};
/** @brief Exactly one mapped-flat or native-General contract for one original storage definition. */
using PooledStorageOwnerProgram=std::variant<PooledRemoteGraphProgram,PooledNativeOwnerGraphProgram>;
/** @return Original geometry shared by the two explicit admission contracts. */
inline const PooledRemoteGraphGeometry& pooledStorageOwnerGeometry(const PooledStorageOwnerProgram& program){
    return std::visit([](const auto& original)->const PooledRemoteGraphGeometry&{return original.geometry();},program);
}
/** @return Exact typed source identity without projecting native work into mapped flat limits. */
inline std::uint64_t pooledStorageOwnerIdentity(const PooledStorageOwnerProgram& program)noexcept{
    return std::visit([](const auto& original){return original.captureIdentity();},program);
}

}
