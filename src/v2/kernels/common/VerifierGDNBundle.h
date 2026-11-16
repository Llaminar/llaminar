/**
 * @file VerifierGDNBundle.h
 * @brief Original four-role GDN provenance across copied verifier subgroups.
 *
 * These construction-only views borrow the stage's original descriptors and
 * exact copied subgroup storage. They never publish host metadata to a device
 * or observe live counts. Attention retains its separate original-span contract.
 */
#pragma once
#include "backends/DeviceId.h"
#include "kernels/common/DeviceRowRange.h"
#include "kernels/common/QuantizedActivationOperandPrecision.h"
#include "tensors/TensorKernels.h"
#include <array>
#include <optional>
#include <span>

namespace llaminar2
{
/** @brief Fixed semantic ordering, independent of backend grouping. */
enum class VerifierGDNRole : std::uint8_t { QKV, Z, Alpha, Beta };
class VerifierGDNSubgroup;

/** @brief Complete original GDN operation, including corrected QKV's row authority. */
class VerifierGDNBundleBinding final
{
public:
    using Projections = std::span<const ITensorGemm::TensorProjectionDesc>;
    /**
     * @brief Authenticate all four original roles before subgroup selection.
     * @param device Original participant. @param input Original activation owner.
     * @param rows Sibling verifier row authority; its count is never read.
     * @param reduction Original input width. @param projections QKV/Z/alpha/beta descriptors.
     * @param precision Original QKV arithmetic; siblings retain their own policies.
     * @param correction_rows Original corrected QKV rows; required for TwoTermQ8 at the same physical width.
     * @throws std::invalid_argument For incomplete roles or inconsistent original arithmetic.
     */
    [[nodiscard]] static VerifierGDNBundleBinding bind(DeviceId device,const TensorBase *input,
        DeviceRowRange rows,int reduction,Projections projections,
        QuantizedActivationOperandPrecision precision,std::optional<DeviceRowRange> correction_rows);
    /**
     * @brief Mint a view of one exact copied subgroup from ordered original roles.
     * @param roles Strictly increasing original role indices.
     * @param copied Exact copied descriptor storage consumed by the backend API.
     * @return Borrowed subgroup identity; the original binding and both spans must outlive it.
     * @throws std::invalid_argument For changed/repeated/reordered members or corrected-QKV reuse.
     */
    [[nodiscard]] VerifierGDNSubgroup subgroup(std::span<const VerifierGDNRole> roles,Projections copied) const;
    /** @return Original participant. */
    [[nodiscard]] DeviceId device() const noexcept {return device_;}
    /** @return Original activation object, never a host mirror. */
    [[nodiscard]] const TensorBase *input() const noexcept {return input_;}
    /** @return Original unsplit verifier rows. */
    [[nodiscard]] const DeviceRowRange &rows() const noexcept {return rows_;}
    /** @return Original shared input width. */
    [[nodiscard]] int reduction() const noexcept {return reduction_;}
    /** @return All four original descriptors, in semantic order. */
    [[nodiscard]] Projections projections() const noexcept {return projections_;}
    /** @return QKV's original precision policy. */
    [[nodiscard]] QuantizedActivationOperandPrecision precision() const noexcept {return precision_;}
    /** @return Corrected QKV's original row identity, if applicable. */
    [[nodiscard]] const std::optional<DeviceRowRange> &correctionRows() const noexcept {return correction_rows_;}
private:
    /** @brief Only the validating factory creates complete original bindings. */
    VerifierGDNBundleBinding(DeviceId device,const TensorBase *input,DeviceRowRange rows,
        int reduction,Projections projections,QuantizedActivationOperandPrecision precision,
        std::optional<DeviceRowRange> correction_rows) noexcept;
    DeviceId device_;
    const TensorBase *input_;
    DeviceRowRange rows_;
    int reduction_;
    Projections projections_;
    QuantizedActivationOperandPrecision precision_;
    std::optional<DeviceRowRange> correction_rows_;
};

/** @brief A backend subgroup can only be minted from a complete original binding. */
class VerifierGDNSubgroup final
{
public:
    /**
     * @brief Require the same active original owner and exact copied API arguments.
     * @param device Actual participant. @param input Actual activation object.
     * @param rows Actual original rows. @param reduction Actual input width.
     * @param copied Actual copied subgroup storage, before backend repartitioning.
     * @return False for stale owners, changed members or foreign storage.
     */
    [[nodiscard]] bool matches(DeviceId device,const TensorBase *input,const DeviceRowRange &rows,
        int reduction,VerifierGDNBundleBinding::Projections copied) const noexcept;
    /** @return Original complete binding; valid while its scope remains current. */
    [[nodiscard]] const VerifierGDNBundleBinding &original() const noexcept {return *original_;}
private:
    friend class VerifierGDNBundleBinding;
    friend class ScopedVerifierGDNSubgroup;
    /** @brief Copy bounded role indices while borrowing immutable descriptor storage. */
    VerifierGDNSubgroup(const VerifierGDNBundleBinding &original,
        std::span<const VerifierGDNRole> roles,VerifierGDNBundleBinding::Projections copied) noexcept;
    /** @return Whether every copied semantic field still matches its original role. */
    [[nodiscard]] bool membersMatch() const noexcept;
    const VerifierGDNBundleBinding *original_;
    std::array<VerifierGDNRole,4> roles_{};
    VerifierGDNBundleBinding::Projections copied_;
};

/** @brief Keep a neutral original GDN boundary even when native policy is unqualified. */
class ScopedVerifierGDNBundle final
{
public:
    /** @param binding Complete original operation, or an explicitly empty nested boundary. */
    explicit ScopedVerifierGDNBundle(std::optional<VerifierGDNBundleBinding> binding) noexcept;
    /** @brief Restore the exact enclosing scope; out-of-order retirement is fatal. */
    ~ScopedVerifierGDNBundle();
    ScopedVerifierGDNBundle(const ScopedVerifierGDNBundle&)=delete;
    ScopedVerifierGDNBundle &operator=(const ScopedVerifierGDNBundle&)=delete;
    /** @return Current complete operation on this recording worker, if present. */
    [[nodiscard]] static const VerifierGDNBundleBinding *current() noexcept;
private:
    std::optional<VerifierGDNBundleBinding> binding_;
    const ScopedVerifierGDNBundle *previous_;
    static thread_local const ScopedVerifierGDNBundle *active_;
};

/** @brief Scope an authenticated copied subgroup only while its original owner is active. */
class ScopedVerifierGDNSubgroup final
{
public:
    /** @param subgroup Minted subgroup or an explicit unmeasured nested call.
     * @throws std::invalid_argument If its original owner is no longer current. */
    explicit ScopedVerifierGDNSubgroup(std::optional<VerifierGDNSubgroup> subgroup);
    /** @brief Restore the exact enclosing copied subgroup in LIFO order. */
    ~ScopedVerifierGDNSubgroup();
    ScopedVerifierGDNSubgroup(const ScopedVerifierGDNSubgroup&)=delete;
    ScopedVerifierGDNSubgroup &operator=(const ScopedVerifierGDNSubgroup&)=delete;
    /** @return Current borrowed subgroup, or null for an unmeasured boundary. */
    [[nodiscard]] static const VerifierGDNSubgroup *current() noexcept;
private:
    std::optional<VerifierGDNSubgroup> subgroup_;
    const ScopedVerifierGDNSubgroup *previous_;
    static thread_local const ScopedVerifierGDNSubgroup *active_;
};
}
