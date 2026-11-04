/**
 * @file NativeOrderingEventBudgetProjection.h
 * @brief Sealed original-source bounds projection for ordering and remote graph planning shared by actual planning and recorded replay.
 *
 * Production construction uses only the installed selector and actual original
 * source metadata. Recorded certificates are confined to the named replay friend;
 * callers cannot supply numeric limits or replace the production catalog.
 */
#pragma once
#include "NativeOrderingEventProgram.h"
#include "planning/PhysicalMemoryAuthority.h"
#include <memory>
#include <vector>
#include "NativeAuxiliaryResourceCatalog.h"
namespace llaminar2 {
class GPUContextProvenance;
namespace test {struct NativeOrderingEventGroupReplayAccess;}
/** @brief Immutable certificate source, never a live reservation or capacity ledger. */
class NativeOrderingEventBudgetProjection final {
public:
    /** @return Same installed allowance origin for cold BOM; actual group creation separately authenticates its original local context. */
    static NativeOrderingEventBudgetProjection installed();
    /** @return Closed reviewed group contribution on original CPU/GPU observations. */
    PhysicalMemoryPlan bom(NativeOrderingEventProgram,PhysicalMemoryResource host,PhysicalMemoryResource gpu)const;
    /** @return Same actual remote-host projection from an installed or closed recorded source; creates no native worker. */
    PhysicalMemoryPlan remoteHostBOM(const PooledRemoteGraphProgram&,PhysicalMemoryResource host,PhysicalMemoryResource gpu)const;
    /** @return Exact reviewed floor-to-actual bound growth for the same original program role. */
    PhysicalMemoryPlan growthBOM(NativeOrderingEventProgram floor,NativeOrderingEventProgram actual,PhysicalMemoryResource host,PhysicalMemoryResource gpu)const;
private:
    friend struct test::NativeOrderingEventGroupReplayAccess;
    friend class NativeOrderingEventGroup;
    struct RecordedBound {DeviceId device;NativeOrderingEventProgram program;std::size_t hostBytes,deviceBytes;std::string evidence,sourceIdentity;NativePhysicalLifetime lifetime;};
    struct RecordedRemoteBound {DeviceId device;std::size_t hostBytes,nativeObjectBytes;std::string evidence,sourceIdentity;NativeAuxiliaryRemoteFamilyScope family;};
    static NativeOrderingEventBudgetProjection fromRecorded(std::vector<RecordedBound>,std::vector<RecordedRemoteBound> remote={});
    NativeAuxiliaryResourceSelection select(NativeOrderingEventProgram,DeviceId)const;
    /**
     * @brief Project cold limits from the original installed or recorded origin.
     * @param program Exact ordering operation and nonempty root set.
     * @param device Original allocator endpoint, which may belong to another rank.
     * @return Bounds without a local worker, generation or materialization permit.
     * @throws std::exception For a missing origin, program or matching record.
     */
    NativeAuxiliaryResourceAllowance bounds(NativeOrderingEventProgram,DeviceId)const;
    /** @brief Installed origin authenticates the original source/PMA before any group reservation. */
    void requireOriginalSource(const std::shared_ptr<PhysicalMemoryAuthority>&,const GPUContextProvenance&)const;
    struct State;
    explicit NativeOrderingEventBudgetProjection(std::shared_ptr<const State>);
    std::shared_ptr<const State> state_;
};
}
