/**
 * @file PooledImmutableLocalHotNativePlan.h
 * @brief Immutable local-cache native bindings derived from the genuine existing pooled source.
 *
 * This declaration retains the original service-issued source; it neither
 * constructs a provider nor turns equal geometry into materialization authority.
 */
#pragma once
#include "PooledImmutableLocalHotGeometry.h"
#include "tensors/TensorType.h"
#include <memory>
namespace llaminar2::qwen4exp {
class PooledImmutableSourcePreparedStorage;
/** @brief Checked static geometry plus original table/source lifetime identity. */
class PooledImmutableLocalHotNativePlan final {
public:
    /** @brief Bind the separate local geometry to the same original prepared source owner. */
    static PooledImmutableLocalHotNativePlan fromOriginalSource(PooledImmutableLocalHotGeometry geometry,
        std::shared_ptr<PooledImmutableSourcePreparedStorage> source);
    /** @return Same immutable local declaration consumed by planning. */
    const PooledImmutableLocalHotGeometry& geometry()const noexcept{return geometry_;}
    /** @return Original service-issued table identity, distinct from local slot generations. */
    std::uint64_t table()const noexcept{return table_;}
    /** @return Original immutable source storage generation. */
    std::uint64_t sourceStorageGeneration()const noexcept{return source_generation_;}
    /** @return Checked original source format for the existing native decoder. */
    TensorType nativeFormat()const noexcept{return format_;}
    /** @return Strong original source lifetime; no additional source or physical bytes are created. */
    const std::shared_ptr<PooledImmutableSourcePreparedStorage>& originalSource()const noexcept{return source_;}
    /** @return Complete immutable geometry/table/source association for capture identity. */
    std::uint64_t captureIdentity()const noexcept{return identity_;}
private:
    /** @brief Construct only after authenticating the original source association. */
    PooledImmutableLocalHotNativePlan(PooledImmutableLocalHotGeometry geometry,
        std::shared_ptr<PooledImmutableSourcePreparedStorage> source,std::uint64_t table,
        std::uint64_t generation,TensorType format,std::uint64_t identity)
        :geometry_(std::move(geometry)),source_(std::move(source)),table_(table),
          source_generation_(generation),format_(format),identity_(identity){}
    PooledImmutableLocalHotGeometry geometry_;
    std::shared_ptr<PooledImmutableSourcePreparedStorage> source_;
    std::uint64_t table_,source_generation_;
    TensorType format_;
    std::uint64_t identity_;
};
} // namespace llaminar2::qwen4exp
