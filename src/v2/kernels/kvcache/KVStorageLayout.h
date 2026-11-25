/**
 * @file KVStorageLayout.h
 * @brief Native KV token/basis geometry shared by admission and paged cache construction.
 *
 * A request basis is not a page header: AQ8 keys keep the first serial key's
 * basis across every page, COW and prefix attachment. Archive payloads may
 * serialize it once per block, but that does not authorize reinitializing it
 * when a physical page is empty. Model codebooks/rotations and allocation
 * rounding are separate owners; this descriptor contains only native payload.
 */
#pragma once
#include "backends/DeviceId.h"
#include <cstddef>
#include <string>

namespace llaminar2
{
/** @brief Actual cache construction family, not a precision alias. */
enum class KVCacheFamily { AttentionOnly, Hybrid };
/** @brief Installed storage codecs; unknown tokens never select another codec. */
enum class KVStorageFormat { FP32, BF16, FP16, Q8_1, Q16_1, TQ4, TQ8 };
/** @brief Resolve existing public codec tokens with their established case/alias behavior. */
KVStorageFormat parseKVStorageFormat(const std::string &precision);
/** @brief Native logical payload extent; fixed basis bytes are included once when serialized. */
struct LogicalKVPayload
{
    std::size_t k_bytes = 0;
    std::size_t v_bytes = 0;
    /** @return Combined native payload; construction checks representability. */
    std::size_t totalBytes() const noexcept { return k_bytes + v_bytes; }
};
/** @brief Immutable basis scope; page assignment cannot change this policy. */
enum class KVKeyBasisScope { None, RequestFirstSerialKey };
/** @brief Native per-position and separate request-basis bytes for one local layer. */
class KVNativeStorageLayout final
{
public:
    const KVStorageFormat format;
    const KVKeyBasisScope basis_scope;
    const std::size_t key_position_bytes;
    const std::size_t value_position_bytes;
    const std::size_t request_basis_bytes;
    /**
     * @brief Resolve codec-native block geometry without observing cache state.
     * @param family Actual main/predictor cache family.
     * @param heads Participant-local KV heads.
     * @param head_dim Coordinates per head.
     * @param precision Canonical configured codec token.
     * @param device Actual CPU/CUDA/ROCm owner, not a tuning architecture.
     * @return Checked native payload geometry, not an allocation/grant decision.
     */
    static KVNativeStorageLayout resolve(KVCacheFamily family, int heads, int head_dim,
        const std::string &precision, DeviceId device);
    /** @brief Native page payload excluding the separately owned immutable request basis. */
    LogicalKVPayload pagePayload(int tokens) const;
    /** @brief Canonical archive payload including its one serialized request basis. */
    LogicalKVPayload archivePayload(int tokens) const;
private:
    /** @brief Only the validated native factory may issue a layout. */
    KVNativeStorageLayout(KVStorageFormat codec, KVKeyBasisScope basis, std::size_t key,
        std::size_t value, std::size_t anchor)
        : format(codec), basis_scope(basis), key_position_bytes(key), value_position_bytes(value), request_basis_bytes(anchor) {}
};
}
