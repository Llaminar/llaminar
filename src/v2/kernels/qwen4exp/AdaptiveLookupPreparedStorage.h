/**
 * @file AdaptiveLookupPreparedStorage.h
 * @brief Reuse canonical prepared Tensor roots for one mutually-exclusive adaptive lookup frame.
 *
 * This binding retains existing roots and uses Qwen4ExpPreparedStorage to seal
 * exact pointers/device/runtime after ordinary materialization. It allocates no
 * payload, issues no PMA claims and publishes no caller-maintained state ledger.
 * Mandatory frame/protocol roots exist for C0; optional cache root is absent.
 */
#pragma once
#include "ImmutableAdaptiveLookupKernels.h"
#include "execution/qwen4exp/Qwen4ExpPreparedStorage.h"
#include <array>
#include <memory>
#include <utility>
#include <optional>
#include <atomic>
#include <mutex>
namespace llaminar2::qwen4exp
{
/** @brief Setup-only declaration and complete physical seal, strongly retaining all actual roots. */
class AdaptiveLookupPreparedStorage final
{
public:
    /** @brief Release original root borrows only after their enclosing native graphs/readers have retired. */
    ~AdaptiveLookupPreparedStorage() = default;
    AdaptiveLookupPreparedStorage(const AdaptiveLookupPreparedStorage &) = delete;
    AdaptiveLookupPreparedStorage &operator=(const AdaptiveLookupPreparedStorage &) = delete;
    AdaptiveLookupPreparedStorage(AdaptiveLookupPreparedStorage &&) = delete;
    AdaptiveLookupPreparedStorage &operator=(AdaptiveLookupPreparedStorage &&) = delete;
    /**
     * @brief Seal only ordinary-prepared storage; never allocate, copy or reset warmed values.
     * @throws std::exception On unprepared, foreign, rebound or misaligned original roots.
     * @note Setup serialization shares the same owner across graph families; every later reuse validates the
     * seal.
     */
    void seal();
    /**
     * @brief Reject root/type/extent/runtime replacement without touching mutable data.
     * @param plan Exact checked declaration retained by the source owner.
     * @throws std::exception On an unsealed owner or any declaration/physical identity drift.
     */
    void requireFor(const AdaptiveImmutableRowCacheNativePlan &plan) const;
    /**
     * @return Complete source geometry and original prepared device/backend/runtime identity.
     * @throws std::exception Unless ordinary preparation installed a still-valid physical seal.
     * @note The bundle adds source/codebook/launch identity; the stage adds query/controller identities.
     */
    std::uint64_t captureIdentity() const;

private:
    friend class IImmutableAdaptiveLookupKernels;
    friend class ImmutableLookupService;
    /**
     * @brief Declare canonical already-admitted roots without eager materialization.
     * @param plan Source-owned checked cache/frame geometry.
     * @param cache Optional model-lifetime canonical INT8 root, required exactly when slots>0.
     * @param frame Single mandatory canonical INT8 work-frame root shared only by mutually-exclusive graph
     * families.
     * @param protocol Original canonical INT32 root of exactly64B, retaining its lifetime sequence/admission.
     */
    AdaptiveLookupPreparedStorage(AdaptiveImmutableRowCacheNativePlan plan, std::shared_ptr<TensorBase> cache,
                                  std::shared_ptr<TensorBase> frame, std::shared_ptr<TensorBase> protocol);
    /** @brief Same checked declarations without recursively acquiring the setup mutex. */
    void requireForUnlocked(const AdaptiveImmutableRowCacheNativePlan &plan) const;
    /** @brief Return native root coordinates only after their complete seal was checked. */
    AdaptiveLookupDeviceStorage native() const;
    /** @brief Setup submission ownership only; no host copy of active device frame/cache state. */
    enum class Setup
    {
        Declared,
        MetadataSubmitted,
        Admitted,
        Failed
    };
    std::atomic<Setup> setup_{Setup::Declared}; ///< Failed partial submission is sticky and never retryable.
    /** @brief Reject invalid declarations before seal/materialization. */
    void declarations() const;
    mutable std::mutex
        setup_mutex_; ///< Serializes setup/seal/publication only; never mirrors device execution.
    AdaptiveImmutableRowCacheNativePlan plan_;
    std::shared_ptr<TensorBase> cache_, frame_, protocol_;
    std::array<TensorBase *, 3> roots_{};
    std::optional<Qwen4ExpPreparedStorage> prepared_;
};
} // namespace llaminar2::qwen4exp
