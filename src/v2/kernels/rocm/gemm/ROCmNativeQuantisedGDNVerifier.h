/**
 * @file ROCmNativeQuantisedGDNVerifier.h
 * @brief Complete GDN provenance for measured native verifier selection.
 *
 * QKV and Z keep their original quantized sources even when corrected QKV is
 * excluded from the native packet. Alpha/beta retain their prepared floating
 * policies. These bounded copies exist on the graph-recording worker only.
 */
#pragma once
#include "ROCmNativeQuantisedVerifierLaunch.h"
#include "kernels/common/QuantizedActivationOperandPrecision.h"
#include "tensors/TensorKernels.h"

namespace llaminar2::rocm
{
/** @brief Original prepared floating control; no synthetic quantized identity. */
struct NativeQuantisedGDNControl
{
    const ITensorGemm *engine = nullptr; ///< Exact original prepared engine.
    ContiguousFloatingPointWeightDescriptor weights; ///< Resident physical operand.
    float *output = nullptr; ///< Distinct original control output.
    FloatingActivationOperandPrecision precision = FloatingActivationOperandPrecision::Native;
};
/** @brief All four original roles, independent of copied backend subgroups. */
struct NativeQuantisedGDNVerifierOperation
{
    int device, k;
    const TensorBase *input;
    DeviceRowRange rows;
    QuantizedActivationOperandPrecision precision;
    std::optional<DeviceRowRange> correction_rows;
    const int8_t *activations;
    const float *activation_scales;
    const int32_t *activation_sums;
    NativeQuantisedVerifierBundleMember qkv, z;
    NativeQuantisedGDNControl alpha, beta;
};
/** @brief Native GDN scope; an empty nested call hides the enclosing binding. */
class ScopedNativeQuantisedGDNVerifier final
{
public:
    /** @param operation Authenticated complete GDN stage or an unmeasured boundary. */
    explicit ScopedNativeQuantisedGDNVerifier(std::optional<NativeQuantisedGDNVerifierOperation> operation) noexcept;
    /** @brief Restore the exact enclosing scope; non-LIFO retirement is fatal. */
    ~ScopedNativeQuantisedGDNVerifier();
    ScopedNativeQuantisedGDNVerifier(const ScopedNativeQuantisedGDNVerifier&)=delete;
    ScopedNativeQuantisedGDNVerifier &operator=(const ScopedNativeQuantisedGDNVerifier&)=delete;
    /** @return Current original GDN binding, never a device-state mirror. */
    [[nodiscard]] static const NativeQuantisedGDNVerifierOperation *current() noexcept;
private:
    std::optional<NativeQuantisedGDNVerifierOperation> operation_;
    const ScopedNativeQuantisedGDNVerifier *previous_;
    static thread_local const ScopedNativeQuantisedGDNVerifier *active_;
};
/** @brief Authenticate native packet roles and the complete original GDN arithmetic.
 * @param operation All original roles. @param work Actual unsliced packet.
 * @param device Exact participant. @return Whether packet and original binding agree. */
[[nodiscard]] bool matchesNativeQuantisedGDNVerifier(const NativeQuantisedGDNVerifierOperation &operation,
    const NativeQuantisedVerifierWork &work,int device) noexcept;
/** @brief Resolve measured source/precision/shape/width keys without live count reads.
 * @param operation Complete original identity. @return Whether this width qualifies for WMMA. */
[[nodiscard]] bool measuredNativeQuantisedGDNVerifier(const NativeQuantisedGDNVerifierOperation &operation) noexcept;
}
