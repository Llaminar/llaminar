/**
 * @file DensePrefillTiming.h
 * @brief Shared native-graph timing contract for dense GPU prefill tournaments.
 *
 * Serving replays retained graphs. A timing sample therefore replays one graph
 * containing sixteen complete projection operations with persistent storage.
 * Normalized latency includes activation quantization and ordered reducers,
 * while amortizing the one host graph submission and outer timing events.
 * The raw corpus records this count and authenticates it before installation.
 */
#pragma once
namespace llaminar2::test
{
    /** @brief Complete projection operations in one native timing replay. */
    inline constexpr int kDensePrefillCapturedOperations = 16;
}
