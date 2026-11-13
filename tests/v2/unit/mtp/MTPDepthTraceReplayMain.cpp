/** @file MTPDepthTraceReplayMain.cpp
 * @brief Device-free CLI dispatch; ordinary GoogleTest behavior remains the default.
 */
#include "MTPDepthTraceReplay.h"
#include <gtest/gtest.h>
#include <string_view>
/** @param argc Exact argument count. @param argv CLI or ordinary GoogleTest arguments.
 * @return Explicit replay result or the standard GoogleTest result.
 */
int main(int argc,char **argv)
{
    if(argc>1&&std::string_view(argv[1])=="--replay-depth-trace")
        return llaminar2::test::mtp_trace_replay::runCLI(argc,argv);
    ::testing::InitGoogleTest(&argc,argv);return RUN_ALL_TESTS();
}
