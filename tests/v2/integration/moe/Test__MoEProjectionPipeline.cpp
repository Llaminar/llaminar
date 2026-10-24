/**
 * @file Test__MoEProjectionPipeline.cpp
 * @brief Functional-only native projection graph regressions in production preflight.
 *
 * The shared fixture covers every expert format, sparse runtime ownership and
 * adversarial request/graph lifecycle transitions. Performance evidence has a
 * separate explicit target; there is no timing gate in production preflight.
 * Fixed scalar publication is exercised under both semantic phase declarations;
 * existing wider-row and counted-row declarations retain their original layouts.
 */
#include <gtest/gtest.h>
#include "utils/MoEProjectionPipelineFixture.h"

namespace llaminar2::test
{
/**
 * @brief Sweep serial/MTP row geometries and every codebook through the decode sum window.
 * @param device Backend selector; the fixture reverses physical participant order.
 * @param exchange Explicit native or counted intermediate publication protocol.
 */
static void proveSharedFullRowsOverlap(DeviceId device, ProjectionExchangeProof exchange)
{
    ProjectionPipelineFixtureConfig config;
    config.rows.clear();
    for (int rows = 1; rows <= 16; ++rows) config.rows.push_back(rows);
    config.exchange = exchange;
    config.output = ProjectionOutputProof::SharedFullRowsOverlap;
    config.service_phase = MoEOverlayServicePhaseHint::Decode;
    runProjectionPipelineFixture(device, 2, config);
}

/** @brief Same rows/source/owners under each semantic phase, including retained empty/shrinking replays. */
static void proveCountedPhaseScheduling(DeviceId device, int participants)
{
    for (const auto phase : {MoEOverlayServicePhaseHint::Prefill, MoEOverlayServicePhaseHint::Decode,
            MoEOverlayServicePhaseHint::GroupedVerifier})
    {
        ProjectionPipelineFixtureConfig config;
        config.rows = {1, 16};
        config.format = "Q8_0";
        config.exchange = ProjectionExchangeProof::DeviceCounted;
        config.service_phase = phase;
        runProjectionPipelineFixture(device, participants, config);
        ASSERT_FALSE(::testing::Test::HasFailure());
    }
}

/**
 * @brief Prove fixed M1 identity publication under both phase labels and transports.
 * @param device Actual native backend; no device work enters Unit.
 * @param participants Native membership, including the production four-peer case.
 * @details Every run keeps the original wider row, full/empty/partial routes,
 * changed expert ownership, normal request reset and hard-retirement replay.
 */
static void proveScalarColumnPublication(DeviceId device, int participants)
{
    for (const auto phase : {MoEOverlayServicePhaseHint::Prefill, MoEOverlayServicePhaseHint::Decode})
    for (const auto exchange : {ProjectionExchangeProof::Native, ProjectionExchangeProof::DeviceCounted})
    {
        ProjectionPipelineFixtureConfig config;
        config.rows = {16, 1};
        config.format = "Q8_0";
        config.exchange = exchange;
        config.service_phase = phase;
        config.output = ProjectionOutputProof::SharedFullRowsOverlap;
        runProjectionPipelineFixture(device, participants, config);
        ASSERT_FALSE(::testing::Test::HasFailure());
    }
}

#ifdef HAVE_CUDA
/** @brief CUDA scalar publication retains exact output banks and native node proof. */
TEST(MoEProjectionScalarPublication, CUDA2) { proveScalarColumnPublication(DeviceId::cuda(0), 2); }
#endif
#ifdef HAVE_ROCM
/** @brief HIP two-peer publication keeps the same scalar identity and replay proof. */
TEST(MoEProjectionScalarPublication, ROCm2) { proveScalarColumnPublication(DeviceId::rocm(0), 2); }
/** @brief Production four-peer geometry proves participant/physical-order independence. */
TEST(MoEProjectionScalarPublication, ROCm4) { proveScalarColumnPublication(DeviceId::rocm(0), 4); }
#endif

#ifdef HAVE_CUDA
/** @brief Actual counted allgather captures individual prefill/verifier and batched decode at equal M. */
TEST(MoEProjectionPhaseScheduling, CUDA2) { proveCountedPhaseScheduling(DeviceId::cuda(0), 2); }
#endif
#ifdef HAVE_ROCM
/** @brief Symmetric HIP phase policy at equal source geometry. */
TEST(MoEProjectionPhaseScheduling, ROCm2) { proveCountedPhaseScheduling(DeviceId::rocm(0), 2); }
/** @brief P4 distinguishes three original endpoints from one by-value argument packet. */
TEST(MoEProjectionPhaseScheduling, ROCm4) { proveCountedPhaseScheduling(DeviceId::rocm(0), 4); }
#endif

#ifdef HAVE_CUDA
/** @brief CUDA's native packet collective precedes the overlapped full-row shared sum. */
TEST(MoEProjectionSharedFullRowsNative, CUDA)
{ proveSharedFullRowsOverlap(DeviceId::cuda(0), ProjectionExchangeProof::Native); }
/** @brief CUDA counted transfers retain the same sum order through twenty owner replays. */
TEST(MoEProjectionSharedFullRowsCounted, CUDA)
{ proveSharedFullRowsOverlap(DeviceId::cuda(0), ProjectionExchangeProof::DeviceCounted); }
/** @brief All formats, real captured transport, shared overlap and repeated owner changes. */
TEST(MoEProjectionCounted, CUDA)
{
    ProjectionPipelineFixtureConfig config;
    config.exchange = ProjectionExchangeProof::DeviceCounted;
    config.output = ProjectionOutputProof::SharedColumnCompletion;
    runProjectionPipelineFixture(DeviceId::cuda(0), 2, config);
}
TEST(MoEProjectionPipeline, CUDA) { runProjectionPipelineFixture(DeviceId::cuda(0), 2); }
/** @brief Every codebook retains full-row shared-gate arithmetic after column reduction. */
TEST(MoEProjectionSharedColumns, CUDA)
{
    ProjectionPipelineFixtureConfig config;
    config.output = ProjectionOutputProof::SharedColumnCompletion;
    runProjectionPipelineFixture(DeviceId::cuda(0), 2, config);
}
/** @brief Every format preserves accepted-only demand through retained CUDA graphs. */
TEST(MoEProjectionDemand, CUDA)
{
    ProjectionPipelineFixtureConfig config;
    config.history = ProjectionHistoryProof::AllDemandBoundaries;
    runProjectionPipelineFixture(DeviceId::cuda(0), 2, config);
}
#endif
#ifdef HAVE_ROCM
/** @brief Symmetric RCCL shared-sum proof over all floating and quantized experts. */
TEST(MoEProjectionSharedFullRowsNative, ROCm)
{ proveSharedFullRowsOverlap(DeviceId::rocm(0), ProjectionExchangeProof::Native); }
/** @brief HIP counted exchange overlaps only independently owned routed-down work. */
TEST(MoEProjectionSharedFullRowsCounted, ROCm)
{ proveSharedFullRowsOverlap(DeviceId::rocm(0), ProjectionExchangeProof::DeviceCounted); }
/** @brief Symmetric HIP proof of compact production lowering and independent shared work. */
TEST(MoEProjectionCounted, ROCm)
{
    ProjectionPipelineFixtureConfig config;
    config.exchange = ProjectionExchangeProof::DeviceCounted;
    config.output = ProjectionOutputProof::SharedColumnCompletion;
    runProjectionPipelineFixture(DeviceId::rocm(0), 2, config);
}
/** @brief Four producers exercise directed channels beyond a special two-peer protocol. */
TEST(MoEProjectionCounted, ROCm4)
{
    ProjectionPipelineFixtureConfig config;
    config.exchange = ProjectionExchangeProof::DeviceCounted;
    config.output = ProjectionOutputProof::SharedColumnCompletion;
    runProjectionPipelineFixture(DeviceId::rocm(0), 4, config);
}
TEST(MoEProjectionPipeline, ROCm) { runProjectionPipelineFixture(DeviceId::rocm(0), 2); }
TEST(MoEProjectionPipeline, ROCm4) { runProjectionPipelineFixture(DeviceId::rocm(0), 4); }
/** @brief ROCm uses the same all-format completion proof and adversarial row masks. */
TEST(MoEProjectionSharedColumns, ROCm)
{
    ProjectionPipelineFixtureConfig config;
    config.output = ProjectionOutputProof::SharedColumnCompletion;
    runProjectionPipelineFixture(DeviceId::rocm(0), 2, config);
}
/** @brief Four participants prove that column ownership is not a two-peer special case. */
TEST(MoEProjectionSharedColumns, ROCm4)
{
    ProjectionPipelineFixtureConfig config;
    config.output = ProjectionOutputProof::SharedColumnCompletion;
    runProjectionPipelineFixture(DeviceId::rocm(0), 4, config);
}
/** @brief HIP uses the same typed history policy and real accepted-row publisher. */
TEST(MoEProjectionDemand, ROCm)
{
    ProjectionPipelineFixtureConfig config;
    config.history = ProjectionHistoryProof::AllDemandBoundaries;
    runProjectionPipelineFixture(DeviceId::rocm(0), 2, config);
}
#endif
}
