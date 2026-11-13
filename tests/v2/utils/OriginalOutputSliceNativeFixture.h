/**
 * @file OriginalOutputSliceNativeFixture.h
 * @brief Entry point for captured original-source K/V slice qualification.
 *
 * The shared implementation has its own translation unit so fixture changes do
 * not rebuild the large, unrelated CUDA and ROCm regression suite bodies.
 * Root-owned native tests invoke this on an explicit participant and retain the
 * real prepared-source, graph-capture and stream-retirement contracts.
 */
#pragma once
#include "backends/DeviceId.h"

namespace llaminar2::test::original_output_slice_native
{
/**
 * @brief Prove ordinary/grouped captured original-source slices across native formats.
 * @param device Exact GPU endpoint; unsupported or missing backends fail.
 * @throws std::runtime_error For a missing contract or byte/extent mismatch.
 */
void prove(DeviceId device);
}
