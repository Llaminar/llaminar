/**
 * @file GroupedFFNNativeArchitectureReplay.inc
 * @brief Select the real native counted verifier on each supported AMD ISA.
 *
 * Generated GGUF source fixtures use the inherited original source admission
 * and an independent captured M1 row oracle. Each execution is one participant
 * and cannot start a collective across architectures.
 */

namespace llaminar2::test {
namespace {

struct GroupedFFNArchitectureCase {
    const char* label;
    const char* source;
    GroupedFFNFixtureSchedule schedule;
    int rows;
    bool deferred;
};

class GroupedFFNNativeArchitecture : public ::testing::TestWithParam<GroupedFFNArchitectureCase> {};

/** @test Exercise the production graph, dirty reset and accepted-row counters on one physical ISA. */
TEST_P(GroupedFFNNativeArchitecture, CountedReplay)
{
    const auto& selection=GetParam();
    const auto sources=groupedM4Cases();
    const auto source=std::ranges::find_if(sources,[&](const auto& item){return item.name==selection.source;});
    ASSERT_NE(source,sources.end()) << selection.source;
    runGroupedM4Source(*source,selection.deferred,false,selection.rows,selection.schedule);
}

const auto architectureCases=[](GroupedFFNFixtureSchedule schedule) {
    return std::array{
        GroupedFFNArchitectureCase{"Q4_Q51_F128_M4_Static","Q4_K_Q5_1_Down_F128",schedule,4,false},
        GroupedFFNArchitectureCase{"Q4_Q51_F256_M6_Deferred","Q4_K_Q5_1_Down_F256",schedule,6,true},
        GroupedFFNArchitectureCase{"G15_M6_Deferred","G15_IQ3_XXS_wide_direct_tiles",schedule,6,true}};
};
const auto architectureCaseName=[](const auto& info){return info.param.label;};

INSTANTIATE_TEST_SUITE_P(Gfx906,GroupedFFNNativeArchitecture,
    ::testing::ValuesIn(architectureCases(GroupedFFNFixtureSchedule::TunedGfx906)),architectureCaseName);
INSTANTIATE_TEST_SUITE_P(Gfx1100,GroupedFFNNativeArchitecture,
    ::testing::ValuesIn(architectureCases(GroupedFFNFixtureSchedule::PortableGfx1100)),architectureCaseName);

} // namespace
} // namespace llaminar2::test
