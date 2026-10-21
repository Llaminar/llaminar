/**
 * @file RequestInputPositionAdmission.h
 * @brief Authenticate host-position ingress versus a captured canonical decode snapshot.
 *
 * External tokens enter one admitted request bank. Their positions either enter
 * from immutable host rows or are produced by the existing scalar captured root
 * from the canonical KV frontier. A snapshot policy must never upload invented
 * host positions, publish unwritten position storage, or initialize unused token
 * capacity. This checked setup value reads no mutable CPU/GPU execution state.
 */
#pragma once
#include "DeviceDecodePositionBinding.h"
#include <cstdint>
#include <optional>
#include <stdexcept>

namespace llaminar2
{
/** @brief Exact position producer for one host-token admission transaction. */
class RequestInputPositionAdmission final
{
public:
    /** @brief Complete mutually exclusive authorities for admitted positions. */
    enum class Source { HostAbsoluteRows, CapturedCanonicalKVSnapshot };
    /** @brief Authenticate position ownership before admission mutates or transfers anything.
     * @param hostRows Host absolute rows, required only for HostAbsoluteRows.
     * @param snapshot Optional existing captured scalar-root binding.
     * @param backend Exact backend of the admitted device.
     * @param canonicalCount Sole cache-owned device position authority.
     * @param requestPositions Admitted stable request-position destination.
     * @param totalRows Physical token count in this transaction.
     * @param requests Independent request count.
     * @param rowsPerRequest Physical request stride.
     * @return Immutable upload/padding policy for the validated producer.
     * @throws std::invalid_argument For missing, competing, foreign or nonscalar snapshot authority.
     */
    static RequestInputPositionAdmission bind(const int *hostRows,
        std::optional<DeviceDecodePositionBinding> snapshot,IBackend *backend,
        const std::int32_t *canonicalCount,std::int32_t *requestPositions,
        int totalRows,int requests,int rowsPerRequest)
    {
        if(!backend || !requestPositions || totalRows<=0 || requests<=0 || rowsPerRequest<=0 ||
            std::int64_t(requests)*rowsPerRequest!=totalRows)
            throw std::invalid_argument("Request position admission requires exact native ownership and geometry");
        if(snapshot) {
            if(hostRows || totalRows!=1 || requests!=1 || rowsPerRequest!=1 ||
                !snapshot->valid() || snapshot->backend!=backend || !canonicalCount ||
                snapshot->cached_tokens!=canonicalCount || snapshot->position!=requestPositions)
                throw std::invalid_argument("Request decode position admission requires one exact canonical captured snapshot and no host position rows");
            return RequestInputPositionAdmission(Source::CapturedCanonicalKVSnapshot);
        }
        if(!hostRows)
            throw std::invalid_argument("Request position admission is missing its explicit host rows or captured snapshot");
        return RequestInputPositionAdmission(Source::HostAbsoluteRows);
    }
    /** @return The sole authenticated producer, independent of live device contents. */
    [[nodiscard]] Source source() const noexcept { return source_; }
    /** @return Whether admission owns a host-to-device position publication. */
    [[nodiscard]] bool uploadsPositions() const noexcept { return source_==Source::HostAbsoluteRows; }
    /** @return Whether a prefill graph may widen this input into an initialized activation bucket. */
    [[nodiscard]] bool allowsBucketPadding() const noexcept { return source_==Source::HostAbsoluteRows; }
private:
    /** @brief Construct only after the complete ownership proof. */
    explicit RequestInputPositionAdmission(Source source):source_(source) {}
    Source source_;
};
} // namespace llaminar2
