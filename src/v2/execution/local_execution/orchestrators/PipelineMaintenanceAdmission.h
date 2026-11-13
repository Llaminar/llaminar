/**
 * @file PipelineMaintenanceAdmission.h
 * @brief Original native pipeline minimum admission bound for independent stage cadence controllers.
 *
 * The derived bound limits the next verifier; it never overwrites any stage's
 * clock or selects maintenance. One scalar per original participant is reduced
 * to the actual tail leader and broadcast only within the actual tail domain.
 * Cold native recordings are cloned into complete per-device transactions.
 */
#pragma once
#include "collective/ILocalTPContext.h"
#include "backends/IGPUGraphCapture.h"
#include "planning/PhysicalMemoryAuthority.h"
#include "execution/local_execution/device/WorkspaceBufferLease.h"
#include "execution/moe/DeviceMoERebalanceController.h"
#include <memory>
#include <span>
#include <cstdint>
#include <limits>
#include <vector>
#include <optional>
#include <atomic>
#include <barrier>
#include <future>
namespace llaminar2 {
namespace test { struct PipelineMaintenanceAdmissionReplayAccess; }
/** @brief One original controller contribution, or an explicitly inactive authority. */
class PipelineMaintenanceSource final {
public:
    /** @brief Default absence is the explicitly inactive authority. */
    PipelineMaintenanceSource() = default;
    /** @return Whether native INT32 MIN exactly represents the canonical positive-int period declaration. */
    bool valid() const noexcept {
        constexpr auto maximum = uint32_t(std::numeric_limits<int32_t>::max());
        return owner_ ? owner_->contains(offsetof(DeviceMoERebalanceGraphControllerState, decode_rounds_until_maintenance), sizeof(uint32_t)) &&
            initial_period_ && initial_period_ <= maximum && recurring_period_ && recurring_period_ <= maximum
            : !initial_period_ && !recurring_period_;
    }
    /** @return Original retained controller field; null names an actually inactive authority. */
    const uint32_t *remaining() const {
        return owner_ ? static_cast<const uint32_t *>(owner_->data(offsetof(DeviceMoERebalanceGraphControllerState,
            decode_rounds_until_maintenance))) : nullptr;
    }
    /** @return Original physical controller device, absent only for inactive authority. */
    std::optional<DeviceId> device() const { return owner_ ? std::optional{owner_->device()} : std::nullopt; }
private:
    friend class PipelineDeviceGeneration;
    /** @brief Only the original pipeline controller owner may issue an active workspace field. */
    PipelineMaintenanceSource(std::shared_ptr<const WorkspaceBufferLease> owner, uint32_t initial, uint32_t recurring)
        : owner_(std::move(owner)), initial_period_(initial), recurring_period_(recurring) {}
    std::shared_ptr<const WorkspaceBufferLease> owner_;
    uint32_t initial_period_ = 0, recurring_period_ = 0;
};
/** @brief Frozen derived-budget source recordings and exact per-participant physical roots. */
class PipelineMaintenanceAdmission final {
public:
    /** @brief Retire source recordings and exact physical roots before publishing immutable teardown observations. */
    ~PipelineMaintenanceAdmission() noexcept;
    /** @brief Retain and record the original admitted native membership before any model-family capture.
     * @param native Original rank-owned native pipeline context. @param tail_first Actual tail leader coordinate.
     * @param tail_context Original tail TP context, null only for a singleton tail.
     * @param sources Original per-device controller declarations; inactive sources use no fake clock.
     * @param memory Same final PMA shared by every original stage. @param streams Exact original setup streams.
     */
    static std::shared_ptr<PipelineMaintenanceAdmission> create(ILocalTPContext &native, int tail_first,
        ILocalTPContext *tail_context, std::span<const PipelineMaintenanceSource> sources,
        const std::shared_ptr<PhysicalMemoryAuthority> &memory, std::span<void *const> streams);
    /** @return One derived minimum, immutable inactive contribution/status and a distinct mutable marker per original participant. */
    static constexpr std::size_t bytes() noexcept { return 4 * sizeof(uint32_t); }
    /** @return Cold native recording for this exact original participant; never a standalone executable. */
    const IGPUGraphCapture *recording(std::size_t participant) const;
    /** @return Derived global remaining bound on an actual tail member, never a controller clock. */
    const uint32_t *remaining(std::size_t participant) const;
    /** @return Immutable zero status only for an actually inactive tail controller. */
    const uint32_t *inactiveStatus(std::size_t participant) const;
    /** @return The inactive tail's neutral edge marker, acknowledged only by the existing preparation kernel. */
    uint32_t *inactiveMarker(std::size_t participant) const;
    /** @return Original participant's physical device, rejecting foreign coordinates. */
    DeviceId device(std::size_t participant) const;
private:
    friend struct test::PipelineMaintenanceAdmissionReplayAccess;
    /** @brief Submit the exact native scalar plan; shared-body replay records its real root/member/extent operands. */
    static bool enqueueOriginalMinimum(ILocalTPContext &native, ILocalTPContext *tail, int tail_first, int member,
        const void *source, void *result, void *stream);
    /** @brief Drain ready original workers before releasing borrowed capture phases, including pre-entry failures. */
    static void joinOriginalRecordings(ILocalTPContext &native, std::span<std::future<void>> captures,
        std::span<std::atomic<bool>> entered, std::barrier<> &begun, std::barrier<> &recorded,
        std::barrier<> &ended, std::exception_ptr submission_error);
    struct Participant;
    /** @brief Freeze the sole original native context borrow before materialization. */
    PipelineMaintenanceAdmission(ILocalTPContext &native, std::size_t tail_first);
    enum class Phase { Preparing, Recorded };
    Phase phase_ = Phase::Preparing; ///< Published only after the entire original recording cohort succeeds.
    NativeLocalTPContextBorrow original_;
    std::size_t tail_first_;
    std::vector<GlobalDeviceAddress> members_;
    std::vector<std::shared_ptr<Participant>> participants_;
};
} // namespace llaminar2
