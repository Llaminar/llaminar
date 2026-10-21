/**
 * @file ResidentImmutableRowTable.h
 * @brief Shared exact-source native row-table ownership on one admitted GPU.
 *
 * The existing serving path retains one complete original-format GPU copy per
 * actual consumer, shared across graph families. The explicit fixed-source
 * component retains only its checked original range and cannot expose a whole
 * lookup view. Loading uses only a released ordinary-loader
 * slot; no complete RAM table or hot host service exists. Captured stages retain
 * this const owner plus its exact-stream read token until executable retirement.
 */
#pragma once
#include "loaders/ImmutableRowTable.h"
#include "loaders/ResolvedImmutableRowTablePlacement.h"
#include "loaders/gpu_pipeline/ReleasedWeightLoadStaging.h"
#include "transfer/ImmutableNativeDeviceStorage.h"
#include "kernels/qwen4exp/ImmutableResidentLookupKernels.h"
#include "kernels/qwen4exp/ImmutableResidentTableRangeView.h"
#include <memory>
namespace llaminar2
{
class TransferEngine;
/** @brief One exact consumer setup stream and its drained ordinary-loader slot. */
struct ImmutableRowTableDevicePreparation
{
    DeviceId device; ///< Exact planned GPU, never a default-device selector.
    void *stream; ///< Explicit non-null setup upload stream.
    ReleasedWeightLoadStaging staging; ///< Original resolved slot, not recomputed table geometry.
};

/** @brief One authenticated source-native allocation and setup publication lifetime. */
class ResidentImmutableRowTable final
{
public:
    /**
     * @brief Upload exactly one admitted consumer copy through bounded initial I/O.
     * @param source Actual loader-authenticated shard/tensor coordinates.
     * @param placement Sealed concrete VRAM extent and exact consumer owners.
     * @param memory Same canonical PMA which admitted this rank's placement.
     * @param device One exact allocator from placement, never an inferred GPU.
     * @param staging Actual released ordinary-loader slot; no table-sized staging resolution.
     * @param setup_stream Exact non-null stream owning every upload and event.
     * @param transfers Canonical transfer authority, injectable for device-free tests.
     * @return Complete const source owner; no pointer escapes on partial failure.
     * @throws std::exception For invalid admission, changed source or failed native copy.
     */
    [[nodiscard]] static std::shared_ptr<const ResidentImmutableRowTable> prepare(
        ImmutableRowTableSource source, const ResolvedImmutableRowTablePlacement &placement,
        std::shared_ptr<PhysicalMemoryAuthority> memory, DeviceId device,
        const ReleasedWeightLoadStaging &staging, void *setup_stream,
        const TransferEngine &transfers);
    /**
     * @brief Upload only this exact fixed declaration's contiguous original-native owner range.
     * @param source Complete original loader source descriptor, never a fake sliced tensor.
     * @param placement Existing checked FixedHeadShards placement with its sole range list.
     * @param memory Same exact canonical PMA with per-owner native range BOM admitted.
     * @param device Exact actual fixed owner/consumer.
     * @param staging Its real drained original weight-loader slot.
     * @param setup_stream Explicit non-default upload producer stream.
     * @param transfers Sole public transfer/event authority.
     * @return Complete immutable range owner retaining full original source/file identity.
     * @throws std::exception Before publication for source, ownership, codec, admission or upload failure.
     * @note No distributed serving route or collective is installed by this component.
     */
    [[nodiscard]] static std::shared_ptr<const ResidentImmutableRowTable> prepareRange(
        ImmutableRowTableSource source,const ResolvedImmutableRowTablePlacement &placement,
        std::shared_ptr<PhysicalMemoryAuthority> memory,DeviceId device,
        const ReleasedWeightLoadStaging &staging,void *setup_stream,const TransferEngine &transfers);
    ~ResidentImmutableRowTable();
    ResidentImmutableRowTable(const ResidentImmutableRowTable &) = delete;
    ResidentImmutableRowTable &operator=(const ResidentImmutableRowTable &) = delete;
    /** @return Original file/tensor/quantization geometry, never transformed. */
    const ImmutableRowTableSource &source() const noexcept;
    /** @return Immutable file identity authenticated before and after the complete upload. */
    const ImmutableRowFileIdentity &identity() const noexcept;
    /** @return Exact sole GPU owning this real copy. */
    DeviceId device() const noexcept;
    /** @return Exact immutable allocation bytes, suitable for model-retention BOM observation. */
    std::size_t nativeBytes() const noexcept;
    /** @return Checked original range geometry, derived from this owner's sole source/range facts. */
    ImmutableRowTableSourceRange sourceRange() const;
    /** @return Explicit global/local native range view; no implicit whole-source conversion exists. */
    qwen4exp::ImmutableResidentTableRangeView rangeView() const;
    /** @return Exact whole-source view; partial-range owners reject before an old lookup can bind. */
    qwen4exp::ImmutableResidentTableView view() const;
    /** @brief Authenticate the retained source at explicit setup/reuse boundaries only. */
    void authenticateSource() const;
    /** @brief Acquire the setup producer event for one exact consumer before capture. */
    ImmutableNativeDeviceRead acquire(void *consumer_stream, const TransferEngine &transfers) const;
private:
    /** @brief Shared strict original-source upload implementation; placement guards run before this boundary. */
    static std::shared_ptr<const ResidentImmutableRowTable> prepareOwnedRange(
        ImmutableRowTableSource source,ImmutableRowTableSourceRange range,std::shared_ptr<PhysicalMemoryAuthority> memory,
        DeviceId device,const ReleasedWeightLoadStaging &staging,void *setup_stream,const TransferEngine &transfers);
    struct Impl;
    /** @brief Only complete successful preparation may publish a resident owner. */
    explicit ResidentImmutableRowTable(std::unique_ptr<Impl> state);
    std::unique_ptr<Impl> impl_;
};
}
