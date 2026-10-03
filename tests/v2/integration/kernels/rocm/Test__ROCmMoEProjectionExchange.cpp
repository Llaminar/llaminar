/**
 * @file Test__ROCmMoEProjectionExchange.cpp
 * @brief Native two-GPU replay proof for the exact expert projection boundary.
 *
 * Twenty changing-owner transactions include short/empty/reset sentinels. Each
 * output must equal the complete expert pipeline; both the existing route
 * fabric and lossless exchange run. Timing is disabled in this functional gate.
 */
#include <gtest/gtest.h>
#include "../../../utils/ROCmMoEProjectionExchangeFixture.h"

TEST(ROCmMoEProjectionExchange, NativeCollectiveReplaysChangingOwnersExactly)
{
    llaminar2::test::projection_exchange::run({});
}

TEST(ROCmMoEProjectionExchange, IndependentRoutesRetainDecodeAndPrefillBytes)
{
    using namespace llaminar2::test::projection_exchange;
    // Tiny rows use route-owned down dots; larger rows also exercise the
    // expert-tiled family. Neither can reorder the final top-k accumulation.
    for (const int rows : {1, 16, 65})
    {
        SCOPED_TRACE("rows=" + std::to_string(rows));
        run({.rows = rows, .down_publication = DownPublication::IndependentRoutes});
        run({.rows = rows, .down_publication = DownPublication::IndependentRoutes,
            .gate_up_format = "IQ2_S", .down_format = "IQ4_NL"});
    }
}
