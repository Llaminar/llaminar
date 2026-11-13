/**
 * @file ProjectionRowWorkBinding.h
 * @brief Cold binding of original causal publications to participant-local projections.
 *
 * The declaration borrows admitted tensors. Same-owner constant offsets permit
 * strided addressing; arbitrary owners use an exact immutable address table in
 * SerialGraphFamily workspace. Preparation freezes original addresses and leases.
 * Replay checks identity and reads only the original device/CPU publications.
 */
#pragma once
#include "kernels/common/ProjectionRowWork.h"
#include "memory/BufferId.h"
#include "backends/DeviceId.h"
#include "execution/local_execution/device/ProjectionRowPublicationTable.h"
#include "execution/local_execution/device/WorkspaceDescriptor.h"
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace llaminar2 {
class TensorBase;
class DeviceWorkspaceManager;
class CapturedDeviceStorageLease;
/** @brief One original owner with two immutable INT32 subview offsets. */
struct ProjectionRowSource final {
    const TensorBase *tensor=nullptr;
    BufferId buffer=BufferId::_COUNT; ///< Missing arena identity is invalid, never hidden-state ownership.
    std::size_t frontierByteOffset=0,liveByteOffset=0;
};
/** @brief Declarative causal demand; arithmetic is deliberately absent. */
struct ProjectionRowWorkDeclaration final {
    int rowsPerRequest=0,sourceCapacity=0,firstRequiredLogicalRow=0;
    std::vector<ProjectionRowSource> sources;
};
/** @brief Retain original source identity and any immutable workspace table. */
class ProjectionRowWorkBinding final {
public:
    /** @brief Validate original tensor subviews and exact physical row geometry without reading values. */
    ProjectionRowWorkBinding(ProjectionRowWorkDeclaration declaration,DeviceId device,int physicalRows,
        std::optional<DeviceRowRange> prefix={},std::optional<DeviceRequestRowRanges> requests={});
    /** @brief Add only the exact table BOM when owners cannot prove a strided layout. */
    void appendWorkspaceRequirements(WorkspaceRequirements &requirements) const;
    /** @brief Freeze addresses and publish an indirect table once outside native capture. */
    void prepare(DeviceWorkspaceManager *workspace,void *stream);
    /** @brief Reject replacement of original source/table/workspace/stream identity. */
    void requirePrepared(DeviceWorkspaceManager *workspace,void *stream) const;
    /**
     * @brief Require authoritative CPU metadata after its original producer edge.
     * @details Cold preparation remains address-only. GPU execution retains its
     * native graph producer edges and performs no host validity/count read here.
     * @throws std::logic_error For unprepared work or stale CPU source authority.
     */
    void requirePublishedInputs() const;
    /** @return Original immutable work descriptor; never a live host-derived range. */
    const ProjectionRowWork &work() const;
    /** @return Exact original declaration used for coherence and diagnostics. */
    const ProjectionRowWorkDeclaration &declaration() const noexcept {return declaration_;}
    /** @return Optional original indirect publication proof for diagnostics. */
    const std::shared_ptr<const ProjectionRowPublicationTable> &table() const noexcept {return table_;}
    /** @return Complete construction identity; prepared native identities are separately checked. */
    std::uint64_t signature() const noexcept;
    /** @return Whether preparation has frozen all native addresses. */
    bool prepared() const noexcept {return work_.has_value();}
    /** @return Exact immutable setup stream, used only for retained identity validation. */
    void *recordingStream() const noexcept {return stream_;}
private:
    /** @return Exact already-resident source address, without transfer or value read. */
    const std::int32_t *nativeBase(const ProjectionRowSource &source) const;
    const ProjectionRowWorkDeclaration declaration_;
    const DeviceId device_;
    const int physicalRows_;
    const std::optional<DeviceRowRange> prefix_;
    const std::optional<DeviceRequestRowRanges> requests_;
    int strideWords_=0; ///< Positive only after same-owner and exact-offset proof.
    std::string tableName_; ///< Exact owner/offset tuple; no lossy hash selects storage.
    std::optional<ProjectionRowWork> work_;
    std::vector<const std::int32_t *> bases_;
    std::vector<std::size_t> sourceBytes_;
    std::vector<std::shared_ptr<const CapturedDeviceStorageLease>> sourceLeases_;
    std::shared_ptr<const ProjectionRowPublicationTable> table_;
    std::uint64_t workspaceId_=0;
    void *stream_=nullptr;
};
}
