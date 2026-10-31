/**
 * @file PooledImmutableNativeSourceFixture.h
 * @brief Diagnostic-only directed observation and original-native source producers.
 *
 * These producers live only in explicit native tests. They copy actual plan,
 * pin and grant observations and admitted original rows without any host count
 * observation. They authenticate no production transport or RAM/SSD service.
 */
#pragma once
#include "kernels/qwen4exp/PooledImmutableNativeTypes.h"
namespace llaminar2::test::pooled_source_fixture {
namespace n=qwen4exp::pooled_native;
/** @brief Nominal recorded-boundary operation; unsupported source-only decisions fail the component trace. */
enum class Action:std::uint32_t {PoisonCache,DeliverPins,DeliverFree,DeliverVictims,SourceHeader,DeliverSource,ReadRows};
/** @brief Exact private persistent diagnostic staging, separately admitted to the fixture PMA. */
struct IO {
    std::size_t header=0,record_bytes=64,payload_bytes=72,records=128,payload=0,bytes=0;
};
/** @brief Original checked declarations and addresses; no mutable execution values are projected to the host. */
struct Args {
    n::OwnerParameters owner;
    n::CoordinatorParameters coordinator;
    std::byte *cache,*owner_frame,*owner_native,*coordinator_frame,*coordinator_native,*io,*trace,*output;
    const std::byte* original_source;
    IO offsets;
    std::uint32_t source_rows,maximum_original;
};
/** @brief Submit one diagnostic producer on its exact non-default original CUDA stream. */
void launchCUDA(Action,const Args&,void*);
/** @brief Submit the identical diagnostic producer on its exact original HIP stream. */
void launchROCm(Action,const Args&,void*);
}
