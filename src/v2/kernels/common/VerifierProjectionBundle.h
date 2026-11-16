/**
 * @file VerifierProjectionBundle.h
 * @brief Original fused-stage identity for construction-time verifier dispatch.
 *
 * A complete operation owns its performance key before a backend groups compatible
 * leaves. This borrowed identity lives only across the synchronous stage call;
 * neither its address nor its descriptor span enters a captured device packet.
 * Device row counts remain device-owned and are never observed by this scope.
 */
#pragma once

#include "backends/DeviceId.h"
#include "kernels/common/DeviceRowRange.h"
#include "tensors/TensorKernels.h"
#include <cstdint>
#include <optional>
#include <span>

namespace llaminar2
{
/** @brief Complete supported attention projection operations, in semantic order. */
enum class VerifierProjectionBundleKind : std::uint8_t
{
    AttentionQKV, ///< Original query, key, value descriptors in that order.
    AttentionKV,  ///< Original key and value descriptors; no unused query member.
};

/** @brief Validated, immutable identity of one original fused verifier call. */
class VerifierProjectionBundleBinding final
{
public:
    using Projections = std::span<const ITensorGemm::TensorProjectionDesc>;

    /**
     * @brief Bind the original stage before any compatible-policy subgrouping.
     * @param kind Complete operation with its fixed ordered member count.
     * @param device Participant owning the input, prepared engines and outputs.
     * @param input Original activation tensor, retained throughout the call.
     * @param rows Original physical/live row authority; its count is not read.
     * @param reduction Positive input/weight reduction width.
     * @param projections Stage-owned descriptor storage, borrowed synchronously.
     * @throws std::invalid_argument For incomplete members or inconsistent geometry.
     */
    [[nodiscard]] static VerifierProjectionBundleBinding bind(
        VerifierProjectionBundleKind kind, DeviceId device, const TensorBase *input,
        DeviceRowRange rows, int reduction, Projections projections);

    /** @return Whether an API call is the original operation, not a copied subgroup. */
    [[nodiscard]] bool matches(DeviceId device, const TensorBase *input,
        const DeviceRowRange &rows, int reduction, Projections projections) const noexcept;
    /** @return Complete semantic operation; members retain their original order. */
    [[nodiscard]] VerifierProjectionBundleKind kind() const noexcept { return kind_; }
    /** @return Participant admitted at the original stage boundary. */
    [[nodiscard]] DeviceId device() const noexcept { return device_; }
    /** @return Original input tensor; never a host mirror of its contents. */
    [[nodiscard]] const TensorBase *input() const noexcept { return input_; }
    /** @return Original unsplit row binding, including its device count address. */
    [[nodiscard]] const DeviceRowRange &rows() const noexcept { return rows_; }
    /** @return Original reduction width shared by all members. */
    [[nodiscard]] int reduction() const noexcept { return reduction_; }
    /** @return Stage-owned complete descriptors, valid only for this synchronous call. */
    [[nodiscard]] Projections projections() const noexcept { return projections_; }

private:
    /** @brief Only bind() may construct an authenticated descriptor identity. */
    VerifierProjectionBundleBinding(VerifierProjectionBundleKind kind, DeviceId device,
        const TensorBase *input, DeviceRowRange rows, int reduction, Projections projections) noexcept;

    VerifierProjectionBundleKind kind_;
    DeviceId device_;
    const TensorBase *input_;
    DeviceRowRange rows_;
    int reduction_;
    Projections projections_;
};

/** @brief Scope one original bundle on its capture worker; empty scopes hide parents. */
class ScopedVerifierProjectionBundle final
{
public:
    /** @param binding Original operation or an explicit unmeasured nested boundary. */
    explicit ScopedVerifierProjectionBundle(
        std::optional<VerifierProjectionBundleBinding> binding) noexcept;
    /** @brief Restore the exact enclosing construction scope; non-LIFO retirement is fatal. */
    ~ScopedVerifierProjectionBundle();
    ScopedVerifierProjectionBundle(const ScopedVerifierProjectionBundle &) = delete;
    ScopedVerifierProjectionBundle &operator=(const ScopedVerifierProjectionBundle &) = delete;
    /** @return This worker's current original operation, or null for unmeasured work. */
    [[nodiscard]] static const VerifierProjectionBundleBinding *current() noexcept;

private:
    std::optional<VerifierProjectionBundleBinding> binding_;
    const ScopedVerifierProjectionBundle *previous_;
    static thread_local const ScopedVerifierProjectionBundle *active_;
};
}
