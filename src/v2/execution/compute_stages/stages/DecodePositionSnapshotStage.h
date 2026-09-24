/**
 * @file DecodePositionSnapshotStage.h
 * @brief Graph root freezing the live KV position for one ordinary decode row.
 *
 * Every model root depends on this stage. Reusing the existing scalar position
 * kernel keeps both GPU backends identical without a host callback, a second
 * cursor, or an out-of-graph prelude on each generation iteration.
 */
#pragma once

#include "../IComputeStage.h"
#include "../StageParamsBase.h"
#include "../../local_execution/graph/DeviceDecodePositionBinding.h"

namespace llaminar2
{
/** @brief Allocation-free position publication on the exact captured stream. */
class DecodePositionSnapshotStage final : public IComputeStage
{
public:
    /** @brief Borrowed persistent bindings; no request-specific host scalar. */
    struct Params
    {
        STAGE_PARAMS_COMMON_FIELDS;
        DeviceDecodePositionBinding binding;
    };
    static_assert(StageParamsRequired<Params>);

    /** @brief Reject incomplete, aliased, or foreign-backend bindings at setup. */
    explicit DecodePositionSnapshotStage(Params params);
    /** @brief Enqueue one scalar load/store before model roots, without observation. */
    bool execute(IDeviceContext *ctx) override;
    /** @return Stable semantic identity for diagnostics and capture checks. */
    ComputeStageType type() const override { return ComputeStageType::DECODE_POSITION_SNAPSHOT; }
    /** @return Stable graph-root name independent of the current position. */
    std::string name() const override { return "decode_position_snapshot"; }
    /** @return Only GPU backends implement this device-owned state boundary. */
    bool supportsBackend(ComputeBackendType backend) const override;
    /** @return The existing position kernel is completely capturable. */
    bool isGraphCapturable() const override { return true; }
    /** @return No rank coordination is hidden in this participant-local root. */
    bool isCollectiveStage() const override { return false; }
    /** @return Inputs are already resident and ordered by request admission. */
    CoherencePolicy coherencePolicy() const override { return CoherencePolicy::NONE; }
    /** @return The existing arena row produced for all model position consumers. */
    StageBufferContract bufferContract() const override;
protected:
    /** @return Static geometry only; diagnostics never observe the live count. */
    StageDumpInfo buildDumpInfoImpl() const override;
private:
    const Params params_; ///< Captured addresses remain fixed until graph retirement.
};
}
