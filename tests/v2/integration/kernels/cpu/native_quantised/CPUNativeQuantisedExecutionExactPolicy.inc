/**
 * @file CPUNativeQuantisedExecutionExactPolicy.inc
 * @brief Device-free execution-exact admission through the real fused planner.
 *
 * Immutable fixture entries replace only the compile-time lookup dependency;
 * the production Auto/class/ISA resolver remains the actual implementation.
 * Native replay separately validates installed generated rows and output bytes.
 */
namespace llaminar2::test {
namespace {
using namespace cpu::native_quantised;
#if LLAMINAR_COMPILED_WITH_AVX512
constexpr auto exact_test_isa = CPUExecutionExactISA::AVX512;
constexpr auto exact_test_path = ISAPath::AVX512;
constexpr auto exact_test_choice = CPUExecutionExactChoice::WideRows;
constexpr auto exact_test_policy = VerifierRowsPolicy::WideRows;
#else
constexpr auto exact_test_isa = CPUExecutionExactISA::AVX2;
constexpr auto exact_test_path = ISAPath::AVX2;
constexpr auto exact_test_choice = CPUExecutionExactChoice::Pairwise;
constexpr auto exact_test_policy = VerifierRowsPolicy::Pairwise;
#endif
constexpr CPUExecutionExactKey exact_test_key{
    .operation = CPUExecutionExactOperation::GPUAlignedExpertFusedMTPVerifier,
    .numerical_policy = CPUProjectionNumericalPolicy::GPUAlignedExpert,
    .source_codebook = 5, .source_superblock = 0, .prepared_encoding = 0, .asymmetric = 1,
    .build_isa = exact_test_isa, .runtime_isa = exact_test_isa, .workers = 1, .rows = 3,
    .physical_n = 64, .policy_n = 64, .k = 64, .k_tiles = 2, .ambient_n_block_chunks = 1,
    .layout = CPUExecutionExactLayout::ContiguousFP32Aligned64,
    .team_entry = CPUExecutionExactTeamEntry::Standalone,
    .bundle_class = CPUExecutionExactBundleClass::FilledKPartOutputTileLocal};

/** @brief One immutable full-key observation, never a global runtime override. */
bool exactTestLookup(const CPUExecutionExactKey &key, CPUExecutionExactChoice &choice)
{
    if (key != exact_test_key) return false;
    choice = exact_test_choice;
    return true;
}
/** @brief Preserve the existing selector for every input. */
bool exactTestMiss(const CPUExecutionExactKey &, CPUExecutionExactChoice &) { return false; }
/** @brief Prove unsupported generated actions fail in the real ISA resolver. */
bool exactTestInvalidWide(const CPUExecutionExactKey &, CPUExecutionExactChoice &choice)
{
    choice = CPUExecutionExactChoice::WideRows;
    return true;
}
/** @brief Reject artifact corruption instead of silently using a generic policy. */
bool exactTestInvalidChoice(const CPUExecutionExactKey &, CPUExecutionExactChoice &choice)
{
    choice = static_cast<CPUExecutionExactChoice>(99);
    return true;
}

/** @brief Authored table action for qualification of the appended choice. */
bool exactTestOrderedThree(const CPUExecutionExactKey &, CPUExecutionExactChoice &choice)
{
    choice=CPUExecutionExactChoice::OrderedThreeRows;
    return true;
}

/** @brief Metadata-only valid prepared geometry; no payload/kernel is used. */
struct ExactPlanFixture
{
    CPUNativeQuantisedPackedWeights packed;
    alignas(64) std::array<float, 3 * 64> output0{}, output1{};
    std::array<FusedVerifierRowsDesc, 2> descriptors;

    /** @brief Bind contiguous real addresses to the admitted source geometry. */
    ExactPlanFixture()
    {
        packed.N = packed.N_padded = 64; packed.K = 64; packed.blocks_per_row = 2;
        packed.codebook_id = 5; packed.is_asymmetric = true;
        packed.encoding = CPUNativeQuantisedEncoding::NibbleLUT;
        packed.numerical_policy = CPUProjectionNumericalPolicy::GPUAlignedExpert;
        packed.interleaved_block_stride = 1408;
        descriptors = {{
            {.packed = &packed, .output = output0.data(), .N = 64, .ldc = 64, .rows = 3,
                .purpose = ProjectionRowsPurpose::MTPVerifier, .policy_n = 64},
            {.packed = &packed, .output = output1.data(), .N = 64, .ldc = 64, .rows = 3,
                .purpose = ProjectionRowsPurpose::MTPVerifier, .policy_n = 64}}};
    }
};

/** @test Actual Auto planning consumes the complete immutable exact entry. */
TEST(CPUNativeQuantisedExecutionExactPolicy, AutoConsumesExactEntryBeforeGeneric)
{
    ExactPlanFixture fixture;
    const auto plan = planNativeQuantisedFusedRows<exactTestLookup>(fixture.descriptors, 3, 1, exact_test_path);
    ASSERT_EQ(plan.execution_exact_class, CPUExecutionExactBundleClass::FilledKPartOutputTileLocal);
    EXPECT_EQ(plan.partial_route, NativeQuantisedFusedPartialRoute::OutputTileLocal);
    for (size_t index = 0; index < fixture.descriptors.size(); ++index)
    {
        EXPECT_TRUE(plan.projections[index].execution_exact_hit);
        EXPECT_EQ(plan.projections[index].requested_verifier_schedule, VerifierRowsPolicy::Auto);
        EXPECT_EQ(plan.projections[index].effective_verifier_schedule, exact_test_policy);
        EXPECT_EQ(cpuExecutionExactKey(fixture.descriptors[index], plan.projections[index], 1,
            exact_test_isa == CPUExecutionExactISA::AVX512, plan.execution_exact_class), exact_test_key);
    }
}

/** @test Phase/source/layout misses preserve complete existing policy selection. */
TEST(CPUNativeQuantisedExecutionExactPolicy, MissAndExplicitPolicyRetainExistingSelection)
{
    ExactPlanFixture fixture;
    for (const auto purpose : {ProjectionRowsPurpose::Decode, ProjectionRowsPurpose::Prefill})
    {
        for (auto &descriptor : fixture.descriptors) descriptor.purpose = purpose;
        const auto baseline = planNativeQuantisedFusedRows<exactTestMiss>(fixture.descriptors, 3, 1, exact_test_path);
        const auto miss = planNativeQuantisedFusedRows<exactTestLookup>(fixture.descriptors, 3, 1, exact_test_path);
        EXPECT_FALSE(miss.projections[0].execution_exact_hit);
        EXPECT_EQ(miss.projections[0].effective_verifier_schedule, baseline.projections[0].effective_verifier_schedule);
        EXPECT_EQ(miss.partial_route, baseline.partial_route);
    }
    for (auto &descriptor : fixture.descriptors)
    {
        descriptor.purpose = ProjectionRowsPurpose::MTPVerifier;
        descriptor.verifier_schedule = VerifierRowsPolicy::Pairwise;
    }
    const auto forced = planNativeQuantisedFusedRows<exactTestLookup>(fixture.descriptors, 3, 1, exact_test_path);
    EXPECT_FALSE(forced.projections[0].execution_exact_hit);
    EXPECT_EQ(forced.projections[0].effective_verifier_schedule, VerifierRowsPolicy::Pairwise);
}

/** @test Route-class and layout admission cannot borrow another bundle's entry. */
TEST(CPUNativeQuantisedExecutionExactPolicy, UnderfilledAndPaddedBundlesDoNotBorrowExactEvidence)
{
    ExactPlanFixture fixture;
    const auto underfilled = planNativeQuantisedFusedRows<exactTestLookup>(fixture.descriptors, 3, 4, exact_test_path);
    EXPECT_EQ(underfilled.execution_exact_class, CPUExecutionExactBundleClass::None);
    EXPECT_FALSE(underfilled.projections[0].execution_exact_hit);
    fixture.descriptors[1].ldc = 65;
    const auto padded = planNativeQuantisedFusedRows<exactTestLookup>(fixture.descriptors, 3, 1, exact_test_path);
    EXPECT_EQ(padded.execution_exact_class, CPUExecutionExactBundleClass::None);
    EXPECT_FALSE(padded.projections[0].execution_exact_hit);
}

/** @test Binding real outputs cannot add unreserved shared partial storage. */
TEST(CPUNativeQuantisedExecutionExactPolicy, MaterialisedOutputsPreserveMetadataStorageDemand)
{
    ExactPlanFixture fixture;
    auto metadata = fixture.descriptors;
    for (auto &descriptor : metadata) descriptor.output = nullptr;
    const auto admitted = planNativeQuantisedFusedRows<exactTestLookup>(metadata, 3, 1, exact_test_path);
    const auto materialised = planNativeQuantisedFusedRows<exactTestLookup>(fixture.descriptors, 3, 1, exact_test_path);
    ASSERT_EQ(admitted.execution_exact_class, CPUExecutionExactBundleClass::None);
    ASSERT_EQ(materialised.execution_exact_class, CPUExecutionExactBundleClass::FilledKPartOutputTileLocal);
    EXPECT_EQ(admitted.partial_route, NativeQuantisedFusedPartialRoute::OutputTileLocal);
    EXPECT_EQ(materialised.partial_route, admitted.partial_route);
    EXPECT_EQ(admitted.sharedPartialFloats(), 0u);
    EXPECT_EQ(materialised.sharedPartialFloats(), admitted.sharedPartialFloats());
    for (size_t member = 0; member < fixture.descriptors.size(); ++member)
    {
        EXPECT_FALSE(admitted.projections[member].execution_exact_hit);
        EXPECT_TRUE(materialised.projections[member].execution_exact_hit);
        EXPECT_EQ(materialised.projections[member].k_tiles, admitted.projections[member].k_tiles);
    }
}

/** @test Full-width source/geometry/phase fields never alias a packed key. */
TEST(CPUNativeQuantisedExecutionExactPolicy, FullKeyAndSingletonMTPPurposeRemainDistinct)
{
    const auto scalar = exact_test_key.scalars();
    EXPECT_EQ(scalar.size(), 18u);
    auto changed = exact_test_key;
    changed.rows += 65536u; EXPECT_NE(changed.scalars(), scalar);
    changed = exact_test_key; changed.physical_n += 65536u; EXPECT_NE(changed.scalars(), scalar);
    changed = exact_test_key; changed.source_superblock = 1; EXPECT_NE(changed.scalars(), scalar);
    changed = exact_test_key; changed.operation = CPUExecutionExactOperation::GPUAlignedExpertFusedDecode;
    EXPECT_NE(changed.scalars(), scalar);
    ExactPlanFixture fixture;
    for (auto &descriptor : fixture.descriptors) descriptor.rows = 1;
    const auto singleton = planNativeQuantisedFusedRows<exactTestLookup>(fixture.descriptors, 1, 1, exact_test_path);
    EXPECT_FALSE(singleton.projections[0].execution_exact_hit);
    EXPECT_EQ(cpuExecutionExactKey(fixture.descriptors[0], singleton.projections[0], 1,
        exact_test_isa == CPUExecutionExactISA::AVX512, singleton.execution_exact_class).operation,
        CPUExecutionExactOperation::GPUAlignedExpertFusedMTPVerifier);
    EXPECT_STREQ(projectionRowsPurposeName(ProjectionRowsPurpose::MTPVerifier), "mtp_verifier");
}

/** @test Corrupt or ISA-ineligible exact actions fail rather than being substituted. */
TEST(CPUNativeQuantisedExecutionExactPolicy, InvalidArtifactChoiceAndUnsupportedISAReject)
{
    ExactPlanFixture fixture;
    EXPECT_THROW((void)planNativeQuantisedFusedRows<exactTestInvalidChoice>(fixture.descriptors, 3, 1, exact_test_path), std::runtime_error);
    EXPECT_THROW((void)planNativeQuantisedFusedRows<exactTestInvalidWide>(fixture.descriptors, 3, 1, ISAPath::AVX2), std::invalid_argument);
}
/** @test Native source-block admission rejects unsupported down aliases explicitly. */
TEST(CPUNativeQuantisedExecutionExactPolicy, SourceBlockShapeAdmissionDoesNotPadOrRequantize)
{
    for (const auto &format : quantizedVerifierFormats())
    {
        SCOPED_TRACE(format.label);
        EXPECT_NO_THROW(requireQuantizedVerifierSourceShape(format, 320, 2560));
        EXPECT_THROW(requireQuantizedVerifierSourceShape(format, 0, 2560), std::invalid_argument);
        EXPECT_THROW(requireQuantizedVerifierSourceShape(format, 320, 0), std::invalid_argument);
        for (const int k : {64, 96, 160, 320, 640})
        {
            if (format.source_is_superblock)
                EXPECT_THROW(requireQuantizedVerifierSourceShape(format, 2560, k), std::invalid_argument);
            else
                EXPECT_NO_THROW(requireQuantizedVerifierSourceShape(format, 2560, k));
        }
    }
}
/** @test Appended exact choice reaches actual Auto planning only in its canonical domain. */
TEST(CPUNativeQuantisedExecutionExactPolicy, OrderedThreeChoiceAutoDomainAndStorageContract)
{
    static_assert(static_cast<uint32_t>(CPUExecutionExactChoice::Pairwise)==0);
    static_assert(static_cast<uint32_t>(CPUExecutionExactChoice::WideRows)==1);
    static_assert(static_cast<uint32_t>(CPUExecutionExactChoice::OrderedThreeRows)==2);
    ExactPlanFixture fixture;
    for(auto& descriptor:fixture.descriptors)descriptor.purpose=ProjectionRowsPurpose::Prefill;
#if LLAMINAR_COMPILED_WITH_AVX512
    const auto plan=planNativeQuantisedFusedRows<exactTestOrderedThree>(fixture.descriptors,3,1,ISAPath::AVX512);
    EXPECT_EQ(plan.partial_route,NativeQuantisedFusedPartialRoute::OutputTileLocal);
    for(const auto& member:std::span(plan.projections).first(fixture.descriptors.size())) {
        EXPECT_TRUE(member.execution_exact_hit);
        EXPECT_EQ(member.requested_verifier_schedule,VerifierRowsPolicy::Auto);
        EXPECT_EQ(member.effective_verifier_schedule,VerifierRowsPolicy::OrderedThreeRows);
        EXPECT_EQ(member.verifier_resolution.physical_row_tile,3);
    }
    auto metadata=fixture.descriptors;
    for(auto& descriptor:metadata)descriptor.output=nullptr;
    const auto admitted=planNativeQuantisedFusedRows<exactTestOrderedThree>(metadata,3,1,ISAPath::AVX512);
    EXPECT_EQ(admitted.partial_route,plan.partial_route);
    EXPECT_EQ(admitted.sharedPartialFloats(),plan.sharedPartialFloats());
    for(auto& descriptor:fixture.descriptors)descriptor.purpose=ProjectionRowsPurpose::MTPVerifier;
    EXPECT_THROW((void)planNativeQuantisedFusedRows<exactTestOrderedThree>(fixture.descriptors,3,1,ISAPath::AVX512),std::invalid_argument);
#endif
    for(auto& descriptor:fixture.descriptors)descriptor.purpose=ProjectionRowsPurpose::Prefill;
    EXPECT_THROW((void)planNativeQuantisedFusedRows<exactTestOrderedThree>(fixture.descriptors,3,1,ISAPath::AVX2),std::invalid_argument);
}
} // namespace
} // namespace llaminar2::test
