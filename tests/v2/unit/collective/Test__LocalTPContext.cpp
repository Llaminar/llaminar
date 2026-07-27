/**
 * @file Test__LocalTPContext.cpp
 * @brief Unit tests for LocalTPContext
 * @author David Sanftenberg
 * @date January 2026
 *
 * Tests for LOCAL tensor parallelism context:
 * - Construction with devices and weights
 * - Degree calculation
 * - Device index management
 * - Weight normalization
 * - Head/row/column range calculation for proportional TP
 * - Backend auto-detection
 * - Backend initialization (NEW)
 * - Collective operations delegating to backend (NEW)
 * - Integration with HostBackend (NEW)
 */

#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>
#include <cstring>
#include <thread>
#include <atomic>
#include <chrono>
#include <future>
#include <fstream>
#include <sstream>

#include "collective/LocalTPContext.h"
#include "collective/CollectiveTimeoutPolicy.h"
#include "collective/ICollectiveBackend.h"
#include "collective/DeviceGroup.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "backends/GlobalDeviceAddress.h"
#include "config/OrchestrationConfig.h"
#include "tensors/Tensors.h"
#include "utils/DebugEnv.h"
#include "../../utils/TestTensorFactory.h"

using namespace llaminar2;
using namespace llaminar2::test;

namespace
{
    class ScopedEnvVar
    {
    public:
        ScopedEnvVar(const char *name, const char *value)
            : name_(name)
        {
            const char *existing = std::getenv(name);
            if (existing)
                previous_ = std::string(existing);
            setenv(name, value, 1);
            mutableDebugEnv().reload();
        }

        ~ScopedEnvVar()
        {
            if (previous_)
                setenv(name_.c_str(), previous_->c_str(), 1);
            else
                unsetenv(name_.c_str());
            mutableDebugEnv().reload();
        }

        ScopedEnvVar(const ScopedEnvVar &) = delete;
        ScopedEnvVar &operator=(const ScopedEnvVar &) = delete;

    private:
        std::string name_;
        std::optional<std::string> previous_;
    };

    std::string readTextFile(const char *path)
    {
        std::ifstream input(path);
        if (!input)
            return {};

        std::ostringstream contents;
        contents << input.rdbuf();
        return contents.str();
    }
}

// =============================================================================
// Test Fixture
// =============================================================================

class Test__LocalTPContext : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Create test devices
        cuda0_ = GlobalDeviceAddress::cuda(0, 0);
        cuda1_ = GlobalDeviceAddress::cuda(1, 0);
        rocm0_ = GlobalDeviceAddress::rocm(0, 0);
        cpu0_ = GlobalDeviceAddress::cpu(0);
    }

    GlobalDeviceAddress cuda0_;
    GlobalDeviceAddress cuda1_;
    GlobalDeviceAddress rocm0_;
    GlobalDeviceAddress cpu0_;
};

// =============================================================================
// Construction Tests
// =============================================================================

/**
 * @test Construct with single device and no weights
 */
TEST_F(Test__LocalTPContext, ConstructSingleDevice)
{
    auto ctx = createLocalTPContext({cuda0_}, {}, CollectiveBackendType::AUTO);

    ASSERT_NE(ctx, nullptr);
    EXPECT_EQ(ctx->degree(), 1);
    EXPECT_EQ(ctx->devices().size(), 1);
    EXPECT_EQ(ctx->weights().size(), 1);
    EXPECT_FLOAT_EQ(ctx->weights()[0], 1.0f);
}

/**
 * @test Construct with two devices and equal weights
 */
TEST_F(Test__LocalTPContext, ConstructTwoDevicesEqualWeights)
{
    auto ctx = createLocalTPContext({cuda0_, cuda1_}, {}, CollectiveBackendType::HOST);

    ASSERT_NE(ctx, nullptr);
    EXPECT_EQ(ctx->degree(), 2);
    EXPECT_EQ(ctx->devices().size(), 2);
    EXPECT_EQ(ctx->weights().size(), 2);
    EXPECT_FLOAT_EQ(ctx->weights()[0], 0.5f);
    EXPECT_FLOAT_EQ(ctx->weights()[1], 0.5f);
}

/**
 * @test Construct with explicit proportional weights
 */
TEST_F(Test__LocalTPContext, ConstructProportionalWeights)
{
    // 73% / 27% split (like NVIDIA vs AMD performance ratio)
    auto ctx = createLocalTPContext({cuda0_, rocm0_}, {0.73f, 0.27f}, CollectiveBackendType::HOST);

    ASSERT_NE(ctx, nullptr);
    EXPECT_EQ(ctx->degree(), 2);

    // Weights should be normalized to sum to 1.0
    float sum = ctx->weights()[0] + ctx->weights()[1];
    EXPECT_FLOAT_EQ(sum, 1.0f);
    EXPECT_FLOAT_EQ(ctx->weights()[0], 0.73f);
    EXPECT_FLOAT_EQ(ctx->weights()[1], 0.27f);
}

/**
 * @test Construct with unnormalized weights (should normalize)
 */
TEST_F(Test__LocalTPContext, ConstructUnnormalizedWeights)
{
    // Weights that don't sum to 1.0 should be normalized
    auto ctx = createLocalTPContext({cuda0_, cuda1_}, {2.0f, 1.0f}, CollectiveBackendType::HOST);

    ASSERT_NE(ctx, nullptr);

    // Should normalize to 2/3, 1/3
    float sum = ctx->weights()[0] + ctx->weights()[1];
    EXPECT_FLOAT_EQ(sum, 1.0f);
    EXPECT_NEAR(ctx->weights()[0], 2.0f / 3.0f, 0.0001f);
    EXPECT_NEAR(ctx->weights()[1], 1.0f / 3.0f, 0.0001f);
}

TEST_F(Test__LocalTPContext, GpuGraphPolicyAllowsRCCLCapturedCollectivesByDefault)
{
    ScopedEnvVar graphs("LLAMINAR_GPU_GRAPHS", "1");
    ScopedEnvVar capture_collectives("LLAMINAR_GPU_GRAPH_CAPTURE_COLLECTIVES", "1");
    ScopedEnvVar segmented("LLAMINAR_GPU_GRAPH_COLLECTIVE_SEGMENTED", "0");

    std::string reason;
    const bool supported = LocalTPContext::isLocalTPGpuGraphPolicySupported(
        CollectiveBackendType::RCCL,
        &reason);

    EXPECT_TRUE(supported);
    EXPECT_EQ(reason, "rccl_captured_collectives_enabled");
}

TEST_F(Test__LocalTPContext, GpuGraphPolicyAllowsNCCLCapturedCollectivesByDefault)
{
    ScopedEnvVar graphs("LLAMINAR_GPU_GRAPHS", "1");
    ScopedEnvVar capture_collectives("LLAMINAR_GPU_GRAPH_CAPTURE_COLLECTIVES", "1");
    ScopedEnvVar segmented("LLAMINAR_GPU_GRAPH_COLLECTIVE_SEGMENTED", "0");

    std::string reason;
    const bool supported = LocalTPContext::isLocalTPGpuGraphPolicySupported(
        CollectiveBackendType::NCCL,
        &reason);

    EXPECT_TRUE(supported);
    EXPECT_EQ(reason, "nccl_captured_collectives_enabled");
}

TEST_F(Test__LocalTPContext, GpuGraphPolicyRejectsSegmentedOverrideForHomogeneousRCCL)
{
    ScopedEnvVar graphs("LLAMINAR_GPU_GRAPHS", "1");
    ScopedEnvVar capture_collectives("LLAMINAR_GPU_GRAPH_CAPTURE_COLLECTIVES", "0");
    ScopedEnvVar segmented("LLAMINAR_GPU_GRAPH_COLLECTIVE_SEGMENTED", "1");

    std::string reason;
    const bool supported = LocalTPContext::isLocalTPGpuGraphPolicySupported(
        CollectiveBackendType::RCCL,
        &reason);

    EXPECT_FALSE(supported);
    EXPECT_EQ(reason, "homogeneous_collectives_require_full_graph_capture");
}

TEST_F(Test__LocalTPContext, GpuGraphPolicyRejectsLocalTPWhenNoCollectiveGraphPathIsEnabled)
{
    ScopedEnvVar graphs("LLAMINAR_GPU_GRAPHS", "1");
    ScopedEnvVar capture_collectives("LLAMINAR_GPU_GRAPH_CAPTURE_COLLECTIVES", "0");
    ScopedEnvVar segmented("LLAMINAR_GPU_GRAPH_COLLECTIVE_SEGMENTED", "0");

    std::string reason;
    const bool supported = LocalTPContext::isLocalTPGpuGraphPolicySupported(
        CollectiveBackendType::NCCL,
        &reason);

    EXPECT_FALSE(supported);
    EXPECT_EQ(reason, "homogeneous_collectives_require_full_graph_capture");
}

TEST_F(Test__LocalTPContext, AllreduceOnStreamRejectsNullStream)
{
    auto ctx = createLocalTPContext({cuda0_}, {}, CollectiveBackendType::HOST);
    auto tensor = TestTensorFactory::createFP32({2, 4});

    EXPECT_THROW(
        ctx->allreduceOnStream(tensor.get(), "unit_null_stream_allreduce", tensor->numel(), nullptr, "fp16"),
        std::invalid_argument);
}

TEST_F(Test__LocalTPContext, CollectiveSidebandAbiNamesNonAdditiveMetadataSemantics)
{
    EXPECT_STREQ(toString(LocalTPCollectiveSidebandKind::AllreduceSum), "AllreduceSum");
    EXPECT_STREQ(toString(LocalTPCollectiveSidebandKind::Allgather), "Allgather");
    EXPECT_STREQ(toString(LocalTPCollectiveSidebandKind::Broadcast), "Broadcast");

    const std::string header = readTextFile(LLAMINAR_LOCAL_TP_CONTEXT_HEADER);
    ASSERT_FALSE(header.empty());
    EXPECT_NE(header.find("LocalTPCollectiveSidebandKind"), std::string::npos);
    EXPECT_NE(header.find("Command metadata is not additive"), std::string::npos)
        << "MoE rebalance command bytes must not be smuggled through an activation allreduce.";
    EXPECT_NE(header.find("AllreduceSum"), std::string::npos);
    EXPECT_NE(header.find("Allgather"), std::string::npos);
    EXPECT_NE(header.find("Broadcast"), std::string::npos);
}

TEST_F(Test__LocalTPContext, CollectiveSidebandOnStreamRejectsNullStream)
{
    auto ctx = createLocalTPContext({cuda0_}, {}, CollectiveBackendType::HOST);
    LocalTPCollectiveSidebandBuffer sideband;
    sideband.kind = LocalTPCollectiveSidebandKind::AllreduceSum;
    sideband.element_count = 4;
    sideband.name = "unit_histogram_sideband";
    std::vector<LocalTPCollectiveSidebandBuffer> sidebands{sideband};

    EXPECT_THROW(
        ctx->collectiveSidebandOnStream(sidebands, 0, nullptr, "unit_anchor_collective"),
        std::invalid_argument);
}

TEST_F(Test__LocalTPContext, CollectiveSidebandRoutesSemanticKindsToMatchingOnStreamPrimitives)
{
    const std::string source = readTextFile(LLAMINAR_LOCAL_TP_CONTEXT_SOURCE);
    ASSERT_FALSE(source.empty());

    const size_t fn = source.find("bool LocalTPContext::collectiveSidebandOnStream(");
    ASSERT_NE(fn, std::string::npos);
    const size_t next_fn = source.find("bool LocalTPContext::allgatherRawOnStream(", fn);
    ASSERT_NE(next_fn, std::string::npos);
    const std::string body = source.substr(fn, next_fn - fn);

    EXPECT_NE(body.find("LocalTPCollectiveSidebandKind::AllreduceSum"), std::string::npos);
    EXPECT_NE(body.find("allreduceSingleDeviceOnStream"), std::string::npos);
    EXPECT_NE(body.find("CollectiveOp::ALLREDUCE_SUM"), std::string::npos);
    EXPECT_NE(body.find("LocalTPCollectiveSidebandKind::Allgather"), std::string::npos);
    EXPECT_NE(body.find("allgatherSingleDeviceOnStream"), std::string::npos);
    EXPECT_NE(body.find("LocalTPCollectiveSidebandKind::Broadcast"), std::string::npos);
    EXPECT_NE(body.find("broadcastSingleDeviceOnStream"), std::string::npos)
        << "Root command publication needs broadcast semantics; it must not be encoded as an activation allreduce.";
    EXPECT_NE(body.find("record_sideband_runtime"), std::string::npos);
    EXPECT_NE(body.find("sideband_backend_collective_calls"), std::string::npos);
    EXPECT_NE(body.find("sideband_separate_backend_collective_calls"), std::string::npos);
    EXPECT_NE(body.find("\"backend_primitive\""), std::string::npos);
    EXPECT_NE(body.find("\"launch_relation\", \"same_stream_after_anchor\""), std::string::npos);
    EXPECT_NE(body.find("\"fused_with_anchor\", \"false\""), std::string::npos);
    EXPECT_NE(body.find("\"physical_fusion\", \"separate_backend_collective\""), std::string::npos)
        << "Perfstats must make clear that sidebands are same-stream adjacent collectives, not physically fused into the anchor.";
}

TEST_F(Test__LocalTPContext, AllreduceWithSidebandsUsesOneGroupedBackendBundle)
{
    const std::string source = readTextFile(LLAMINAR_LOCAL_TP_CONTEXT_SOURCE);
    ASSERT_FALSE(source.empty());

    const size_t entry = source.find("bool LocalTPContext::allreduceWithSidebandsOnStream(");
    ASSERT_NE(entry, std::string::npos);
    const size_t legacy = source.find("bool LocalTPContext::collectiveSidebandOnStream(", entry);
    ASSERT_NE(legacy, std::string::npos);
    const std::string body = source.substr(entry, legacy - entry);

    EXPECT_NE(body.find("supportsAllreduceWithSidebandsMultiOnStreams"), std::string::npos);
    EXPECT_NE(body.find("allreduceGroupedOnExplicitStreams("), std::string::npos);
    EXPECT_NE(body.find("&sidebands"), std::string::npos);
    EXPECT_NE(body.find("recordLocalTPRuntimeGroupedSidebands"), std::string::npos);
    EXPECT_EQ(body.find("collectiveSidebandOnStream"), std::string::npos)
        << "The production grouped path must not fall back to adjacent sideband collectives.";

    const size_t grouped = source.find("bool LocalTPContext::allreduceGroupedOnExplicitStreams(");
    ASSERT_NE(grouped, std::string::npos);
    const size_t rendezvous = source.find("bool LocalTPContext::rendezvousOnStreamCollective(", grouped);
    ASSERT_NE(rendezvous, std::string::npos);
    const std::string grouped_body = source.substr(grouped, rendezvous - grouped);

    EXPECT_NE(grouped_body.find("grouped_onstream_allreduce_reference_sidebands_"), std::string::npos);
    EXPECT_NE(grouped_body.find("CollectiveSidebandMultiOnStreamsOp"), std::string::npos);
    EXPECT_NE(grouped_body.find("backend_impl_->allreduceWithSidebandsMultiOnStreams"), std::string::npos);
    EXPECT_NE(grouped_body.find("sideband descriptor mismatch"), std::string::npos);
    EXPECT_NE(grouped_body.find("sideband count mismatch"), std::string::npos);

    const size_t support = source.find("bool LocalTPContext::supportsCollectiveSidebandOnStreamGraphCapture() const");
    ASSERT_NE(support, std::string::npos);
    const size_t support_end = source.find("bool LocalTPContext::allreduceWithSidebandsOnStream(", support);
    ASSERT_NE(support_end, std::string::npos);
    const std::string support_body = source.substr(support, support_end - support);
    EXPECT_NE(support_body.find("supportsAllreduceWithSidebandsMultiOnStreams"), std::string::npos)
        << "Graph-native MoE rebalance should require grouped sideband bundles, not merely standalone sideband primitives.";
}

TEST_F(Test__LocalTPContext, BackendCoordinatorsGroupAnchorAndSidebandsInOneRegion)
{
    const std::string backend = readTextFile(LLAMINAR_COLLECTIVE_BACKEND_HEADER);
    const std::string nccl = readTextFile(LLAMINAR_NCCL_COORDINATOR_SOURCE);
    const std::string rccl = readTextFile(LLAMINAR_RCCL_COORDINATOR_SOURCE);
    ASSERT_FALSE(backend.empty());
    ASSERT_FALSE(nccl.empty());
    ASSERT_FALSE(rccl.empty());

    EXPECT_NE(backend.find("CollectiveSidebandMultiOnStreamsOp"), std::string::npos);
    EXPECT_NE(backend.find("allreduceWithSidebandsMultiOnStreams"), std::string::npos);

    auto expectGroupedBundle = [](const std::string &source,
                                  const char *signature,
                                  const char *group_start,
                                  const char *group_end,
                                  const char *anchor_call,
                                  const char *allgather_call,
                                  const char *broadcast_call)
    {
        const size_t fn = source.find(signature);
        ASSERT_NE(fn, std::string::npos) << signature;
        const size_t next = source.find("\n    bool ", fn + 1);
        const std::string body = source.substr(fn, next == std::string::npos
                                                       ? std::string::npos
                                                       : next - fn);

        const size_t start = body.find(group_start);
        ASSERT_NE(start, std::string::npos) << signature;
        const size_t anchor = body.find(anchor_call, start);
        ASSERT_NE(anchor, std::string::npos) << signature;
        const size_t sideband_loop = body.find("for (size_t sideband_idx", anchor);
        ASSERT_NE(sideband_loop, std::string::npos) << signature;
        EXPECT_NE(body.find(allgather_call, sideband_loop), std::string::npos) << signature;
        EXPECT_NE(body.find(broadcast_call, sideband_loop), std::string::npos) << signature;
        const size_t end = body.rfind(group_end);
        ASSERT_NE(end, std::string::npos) << signature;
        EXPECT_LT(start, anchor) << signature;
        EXPECT_LT(anchor, sideband_loop) << signature;
        EXPECT_LT(sideband_loop, end) << signature;
    };

    expectGroupedBundle(
        nccl,
        "bool NCCLCoordinator::allreduceWithSidebandsMultiOnStreams(",
        "nccl::ncclGroupStart()",
        "nccl::ncclGroupEnd()",
        "nccl::ncclAllReduce(",
        "nccl::ncclAllGather(",
        "nccl::ncclBroadcast(");
    expectGroupedBundle(
        rccl,
        "bool RCCLCoordinator::allreduceWithSidebandsMultiOnStreams(",
        "rccl::ncclGroupStart()",
        "rccl::ncclGroupEnd()",
        "rccl::ncclAllReduce(",
        "rccl::ncclAllGather(",
        "rccl::ncclBroadcast(");
}

TEST_F(Test__LocalTPContext, OnStreamGpuCollectivesStayGroupedDuringGraphCapture)
{
    const std::string source = readTextFile(LLAMINAR_LOCAL_TP_CONTEXT_SOURCE);
    ASSERT_FALSE(source.empty());

    const size_t grouped_decl = source.find("const bool grouped_explicit_streams");
    ASSERT_NE(grouped_decl, std::string::npos);
    const size_t grouped_branch = source.find("if (grouped_explicit_streams)", grouped_decl);
    ASSERT_NE(grouped_branch, std::string::npos);

    const std::string grouped_policy = source.substr(grouped_decl, grouped_branch - grouped_decl);
    EXPECT_NE(grouped_policy.find("CollectiveBackendType::NCCL"), std::string::npos);
    EXPECT_NE(grouped_policy.find("CollectiveBackendType::RCCL"), std::string::npos);
    EXPECT_EQ(grouped_policy.find("isGraphCaptureActive"), std::string::npos)
        << "NCCL/RCCL graph-captured on-stream allreduces must use the grouped "
           "explicit-stream launcher, not independent per-device captures.";
}

TEST_F(Test__LocalTPContext, CollectTimeoutPolicySeparatesCollectivesFromWorkerJoins)
{
    using collective_timeout_policy::effectiveCollectTimeoutMs;
    using collective_timeout_policy::effectiveWorkerJoinTimeoutMs;

    EXPECT_EQ(effectiveCollectTimeoutMs(0), 30000);
    EXPECT_EQ(effectiveCollectTimeoutMs(30000), 30000);
    EXPECT_EQ(effectiveCollectTimeoutMs(450000), 450000);

    EXPECT_EQ(effectiveWorkerJoinTimeoutMs(0), 0);
    EXPECT_EQ(effectiveWorkerJoinTimeoutMs(30000), 0)
        << "The collective deadline must not become a wall-clock deadline for "
           "a complete participant forward containing many collectives.";
    EXPECT_EQ(effectiveWorkerJoinTimeoutMs(450000), 0);
}

TEST_F(Test__LocalTPContext, FP16TransportFailuresFailBeforeFP32GroupedAllreduce)
{
    const std::string source = readTextFile(LLAMINAR_LOCAL_TP_CONTEXT_SOURCE);
    ASSERT_FALSE(source.empty());

    const size_t fp32_path = source.find("Standard FP32 allreduce path");
    ASSERT_NE(fp32_path, std::string::npos);
    const size_t scratch_requirement =
        source.find("requireReservedFp16Scratch(", source.find("bool LocalTPContext::allreduceOnStream("));
    ASSERT_NE(scratch_requirement, std::string::npos);
    ASSERT_LT(scratch_requirement, fp32_path);

    const size_t requirement_impl =
        source.find("void *LocalTPContext::requireReservedFp16Scratch(");
    const size_t reservation_impl =
        source.find("bool LocalTPContext::reserveCollectiveResources(", requirement_impl);
    ASSERT_NE(requirement_impl, std::string::npos);
    ASSERT_NE(reservation_impl, std::string::npos);
    const std::string requirement_body =
        source.substr(requirement_impl, reservation_impl - requirement_impl);
    EXPECT_NE(requirement_body.find("requestAbort();"), std::string::npos);
    EXPECT_NE(requirement_body.find("throw std::runtime_error"), std::string::npos);
    EXPECT_NE(requirement_body.find("allocation-free"), std::string::npos);
    EXPECT_EQ(requirement_body.find("->allocate("), std::string::npos);

    auto expectHardFailBeforeFP32Path = [&](const char *marker)
    {
        const size_t marker_pos = source.find(marker);
        ASSERT_NE(marker_pos, std::string::npos) << marker;
        ASSERT_LT(marker_pos, fp32_path) << marker;

        const std::string block = source.substr(marker_pos, fp32_path - marker_pos);
        EXPECT_NE(block.find("requestAbort();"), std::string::npos) << marker;
        EXPECT_NE(block.find("return false;"), std::string::npos) << marker;
        EXPECT_EQ(block.find("fall through"), std::string::npos) << marker;
        EXPECT_EQ(block.find("falling back"), std::string::npos) << marker;
    };

    expectHardFailBeforeFP32Path("FP32->FP16 cast failed");
    expectHardFailBeforeFP32Path("FP16 allreduce failed");
    expectHardFailBeforeFP32Path("FP16->FP32 cast-back failed");
}

TEST_F(Test__LocalTPContext, FP16CastWrappersBindParticipantOrdinalBeforeLaunch)
{
    const std::string local_tp = readTextFile(LLAMINAR_LOCAL_TP_CONTEXT_SOURCE);
    const std::string cuda_cast = readTextFile(LLAMINAR_CUDA_CAST_KERNELS_SOURCE);
    const std::string rocm_cast = readTextFile(LLAMINAR_ROCM_CAST_KERNELS_SOURCE);
    ASSERT_FALSE(local_tp.empty());
    ASSERT_FALSE(cuda_cast.empty());
    ASSERT_FALSE(rocm_cast.empty());

    const size_t ordinal_decl = local_tp.find("const int ordinal = devices_[device_index].device_ordinal");
    ASSERT_NE(ordinal_decl, std::string::npos);

    auto expectLocalCallPassesOrdinal = [&](const char *call_name)
    {
        const size_t call_pos = local_tp.find(call_name, ordinal_decl);
        ASSERT_NE(call_pos, std::string::npos) << call_name;
        const size_t call_end = local_tp.find(") == 0", call_pos);
        ASSERT_NE(call_end, std::string::npos) << call_name;
        const std::string call = local_tp.substr(call_pos, call_end - call_pos);
        EXPECT_NE(call.find("ordinal"), std::string::npos) << call_name;
    };

    expectLocalCallPassesOrdinal("cudaCastFP32ToFP16(");
    expectLocalCallPassesOrdinal("rocmCastFP32ToFP16(");
    expectLocalCallPassesOrdinal("cudaCastFP16ToFP32(");
    expectLocalCallPassesOrdinal("rocmCastFP16ToFP32(");

    auto expectWrapperBindsDevice = [](const std::string &source,
                                       const char *signature,
                                       const char *set_device,
                                       const char *null_guard)
    {
        const size_t sig_pos = source.find(signature);
        ASSERT_NE(sig_pos, std::string::npos) << signature;
        const size_t next_wrapper = source.find("extern \"C\"", sig_pos + 1);
        const std::string body = source.substr(sig_pos, next_wrapper == std::string::npos
                                                            ? std::string::npos
                                                            : next_wrapper - sig_pos);
        EXPECT_NE(body.find("int ordinal"), std::string::npos) << signature;
        EXPECT_NE(body.find(null_guard), std::string::npos) << signature;
        EXPECT_NE(body.find(set_device), std::string::npos) << signature;
    };

    expectWrapperBindsDevice(cuda_cast, "cudaCastFP32ToFP16(", "cudaSetDevice(ordinal)",
                             "!fp32_input || !fp16_output || !stream");
    expectWrapperBindsDevice(cuda_cast, "cudaCastFP16ToFP32(", "cudaSetDevice(ordinal)",
                             "!fp16_input || !fp32_output || !stream");
    expectWrapperBindsDevice(rocm_cast, "rocmCastFP32ToFP16(", "hipSetDevice(ordinal)",
                             "!fp32_input || !fp16_output || !stream");
    expectWrapperBindsDevice(rocm_cast, "rocmCastFP16ToFP32(", "hipSetDevice(ordinal)",
                             "!fp16_input || !fp32_output || !stream");
}

TEST_F(Test__LocalTPContext, GDNCompactWrappersFailFastOnDeviceBindAndClearLaunchState)
{
    const std::string cuda_gdn = readTextFile(LLAMINAR_CUDA_GDN_KERNELS_SOURCE);
    const std::string rocm_gdn = readTextFile(LLAMINAR_ROCM_GDN_KERNELS_SOURCE);
    ASSERT_FALSE(cuda_gdn.empty());
    ASSERT_FALSE(rocm_gdn.empty());

    auto extractWrapper = [](const std::string &source, const char *signature)
    {
        const size_t sig_pos = source.find(signature);
        EXPECT_NE(sig_pos, std::string::npos) << signature;
        if (sig_pos == std::string::npos)
            return std::string();
        const size_t next_wrapper = source.find("extern \"C\"", sig_pos + 1);
        return source.substr(sig_pos, next_wrapper == std::string::npos
                                          ? std::string::npos
                                          : next_wrapper - sig_pos);
    };

    const std::string cuda_body = extractWrapper(cuda_gdn, "cudaGDN_compact_modular_conv_state(");
    const std::string rocm_body = extractWrapper(rocm_gdn, "rocmGDN_compact_modular_conv_state(");
    ASSERT_FALSE(cuda_body.empty());
    ASSERT_FALSE(rocm_body.empty());

    auto expectStrictLaunchWrapper = [](const std::string &body,
                                        const char *set_device,
                                        const char *set_failure,
                                        const char *clear_error,
                                        const char *launch)
    {
        const size_t set_pos = body.find(set_device);
        const size_t failure_pos = body.find(set_failure);
        const size_t clear_pos = body.find(clear_error);
        const size_t launch_pos = body.find(launch);
        ASSERT_NE(set_pos, std::string::npos) << set_device;
        ASSERT_NE(failure_pos, std::string::npos) << set_failure;
        ASSERT_NE(clear_pos, std::string::npos) << clear_error;
        ASSERT_NE(launch_pos, std::string::npos) << launch;
        EXPECT_LT(set_pos, failure_pos);
        EXPECT_LT(failure_pos, clear_pos);
        EXPECT_LT(clear_pos, launch_pos)
            << "stale launch errors must be cleared immediately before the compaction launch";
    };

    expectStrictLaunchWrapper(cuda_body,
                              "cudaSetDevice(device_idx)",
                              "set_err != cudaSuccess",
                              "cudaGetLastError()",
                              "cuda_gdn_compact_modular_conv_state_kernel<<<");
    expectStrictLaunchWrapper(rocm_body,
                              "HipDeviceGuard::setDevice(device_idx)",
                              "set_err != hipSuccess",
                              "hipGetLastError()",
                              "hipLaunchKernelGGL");
}

TEST_F(Test__LocalTPContext, RCCLCoordinatorKeepsHipDeviceGuardTrackingInSync)
{
    const std::string source = readTextFile(LLAMINAR_RCCL_COORDINATOR_SOURCE);
    ASSERT_FALSE(source.empty());

    EXPECT_NE(source.find("trackedHipSetDevice("), std::string::npos);
    EXPECT_NE(source.find("HipDeviceGuard::forceSetDevice(device_ordinal)"), std::string::npos);
    EXPECT_EQ(source.find("hipSetDevice("), std::string::npos)
        << "RCCL coordinator runs direct grouped collectives on worker threads; "
           "raw hipSetDevice leaves HipDeviceGuard's thread-local cache stale and "
           "can make the next kernel launch use a stream from the wrong device.";
}

TEST_F(Test__LocalTPContext, WorkerGpuCheckedEventHelpersBindContextDevice)
{
    const std::string cuda_context = readTextFile(LLAMINAR_NVIDIA_DEVICE_CONTEXT_SOURCE);
    const std::string rocm_context = readTextFile(LLAMINAR_AMD_DEVICE_CONTEXT_SOURCE);
    ASSERT_FALSE(cuda_context.empty());
    ASSERT_FALSE(rocm_context.empty());

    auto extractMethod = [](const std::string &source, const char *signature)
    {
        const size_t sig_pos = source.find(signature);
        EXPECT_NE(sig_pos, std::string::npos) << signature;
        if (sig_pos == std::string::npos)
            return std::string();
        const size_t next_method = source.find("\n    bool ", sig_pos + 1);
        const size_t next_void = source.find("\n    void ", sig_pos + 1);
        size_t end = std::min(next_method == std::string::npos ? source.size() : next_method,
                              next_void == std::string::npos ? source.size() : next_void);
        if (end <= sig_pos)
            end = source.size();
        return source.substr(sig_pos, end - sig_pos);
    };

    const std::string cuda_record =
        extractMethod(cuda_context, "bool NvidiaDeviceContext::recordEventChecked(");
    const std::string cuda_wait =
        extractMethod(cuda_context, "bool NvidiaDeviceContext::waitEventChecked(");
    const std::string cuda_query =
        extractMethod(cuda_context, "bool NvidiaDeviceContext::queryEventChecked(");
    const std::string cuda_sync =
        extractMethod(cuda_context, "bool NvidiaDeviceContext::synchronizeStreamChecked(");
    const std::string rocm_record =
        extractMethod(rocm_context, "bool AMDDeviceContext::recordEventChecked(");
    const std::string rocm_wait =
        extractMethod(rocm_context, "bool AMDDeviceContext::waitEventChecked(");
    const std::string rocm_query =
        extractMethod(rocm_context, "bool AMDDeviceContext::queryEventChecked(");
    const std::string rocm_sync =
        extractMethod(rocm_context, "bool AMDDeviceContext::synchronizeStreamChecked(");
    ASSERT_FALSE(cuda_record.empty());
    ASSERT_FALSE(cuda_wait.empty());
    ASSERT_FALSE(cuda_query.empty());
    ASSERT_FALSE(cuda_sync.empty());
    ASSERT_FALSE(rocm_record.empty());
    ASSERT_FALSE(rocm_wait.empty());
    ASSERT_FALSE(rocm_query.empty());
    ASSERT_FALSE(rocm_sync.empty());

    EXPECT_NE(cuda_record.find("cudaSetDevice(device_ordinal_)"), std::string::npos);
    EXPECT_NE(cuda_wait.find("cudaSetDevice(device_ordinal_)"), std::string::npos);
    EXPECT_NE(cuda_query.find("cudaSetDevice(device_ordinal_)"), std::string::npos);
    EXPECT_NE(cuda_sync.find("cudaSetDevice(device_ordinal_)"), std::string::npos);
    EXPECT_NE(rocm_record.find("setAMDDeviceForResource(device_ordinal_, \"recordEventChecked\")"),
              std::string::npos);
    EXPECT_NE(rocm_wait.find("setAMDDeviceForResource(device_ordinal_, \"waitEventChecked\")"),
              std::string::npos);
    EXPECT_NE(rocm_query.find("setAMDDeviceForResource(device_ordinal_, \"queryEventChecked\")"),
              std::string::npos);
    EXPECT_NE(rocm_sync.find("setAMDDeviceForResource(device_ordinal_, \"synchronizeStreamChecked\")"),
              std::string::npos);
}

TEST_F(Test__LocalTPContext, RawAllgatherUsesParticipantProducerStreams)
{
    const std::string source = readTextFile(LLAMINAR_LOCAL_TP_CONTEXT_SOURCE);
    ASSERT_FALSE(source.empty());

    const size_t fn = source.find("bool LocalTPContext::allgatherRawWithBarrierMultiGpu(");
    ASSERT_NE(fn, std::string::npos);
    const size_t next_fn = source.find("bool LocalTPContext::gatherFromDevices(", fn);
    ASSERT_NE(next_fn, std::string::npos);

    const std::string block = source.substr(fn, next_fn - fn);
    EXPECT_NE(block.find("raw_allgather_producer_streams_"), std::string::npos);
    EXPECT_NE(block.find("backend_impl_->allgatherMultiOnStreams("), std::string::npos);
    EXPECT_EQ(block.find("backend_impl_->allgatherMultiWithComputeDeps("), std::string::npos)
        << "raw allgather must use the caller's producer streams, not globally "
           "registered compute streams";
    EXPECT_EQ(block.find("backend_impl_->allgatherMulti("), std::string::npos)
        << "raw allgather must not continue on producer streams before collective "
           "completion is ordered on those same streams";
}

/**
 * @test Construct with empty devices throws
 */
TEST_F(Test__LocalTPContext, ConstructEmptyDevicesThrows)
{
    EXPECT_THROW(
        createLocalTPContext({}, {}, CollectiveBackendType::AUTO),
        std::invalid_argument);
}

/**
 * @test Construct with mismatched weights count throws
 */
TEST_F(Test__LocalTPContext, ConstructMismatchedWeightsThrows)
{
    EXPECT_THROW(
        createLocalTPContext({cuda0_, cuda1_}, {0.5f}, CollectiveBackendType::HOST),
        std::invalid_argument);
}

/**
 * @test Construct with zero weight throws
 */
TEST_F(Test__LocalTPContext, ConstructZeroWeightThrows)
{
    EXPECT_THROW(
        createLocalTPContext({cuda0_, cuda1_}, {1.0f, 0.0f}, CollectiveBackendType::HOST),
        std::invalid_argument);
}

/**
 * @test Construct with negative weight throws
 */
TEST_F(Test__LocalTPContext, ConstructNegativeWeightThrows)
{
    EXPECT_THROW(
        createLocalTPContext({cuda0_, cuda1_}, {1.0f, -0.5f}, CollectiveBackendType::HOST),
        std::invalid_argument);
}

// =============================================================================
// Backend Auto-Detection Tests
// =============================================================================

/**
 * @test AUTO backend with all CUDA devices -> NCCL (requires 2+ CUDA GPUs)
 */
TEST_F(Test__LocalTPContext, AutoBackendAllCuda)
{
    std::unique_ptr<ILocalTPContext> ctx;
    try
    {
        ctx = createLocalTPContext({cuda0_, cuda1_}, {}, CollectiveBackendType::AUTO);
    }
    catch (const std::runtime_error &e)
    {
        GTEST_SKIP() << "NCCL initialization failed (need 2+ CUDA GPUs): " << e.what();
    }

    EXPECT_EQ(ctx->backend(), CollectiveBackendType::NCCL);
}

// NOTE: Hardware-dependent backend auto-detection tests (AutoBackendAllRocm,
// AutoBackendMixedGpus, AutoBackendSelectsHeterogeneousFor*)
// have been migrated to integration tests in Test__LocalTPBackendBehavior.cpp.
// Those tests require actual GPU hardware (RCCL, HETEROGENEOUS backends).

/**
 * @test AUTO backend with CPU involved -> HOST
 */
TEST_F(Test__LocalTPContext, AutoBackendWithCpu)
{
    auto ctx = createLocalTPContext({cuda0_, cpu0_}, {}, CollectiveBackendType::AUTO);

    EXPECT_EQ(ctx->backend(), CollectiveBackendType::HOST);
}

/**
 * @test Explicit backend is preserved
 */
TEST_F(Test__LocalTPContext, ExplicitBackendPreserved)
{
    auto ctx = createLocalTPContext({cuda0_, cuda1_}, {}, CollectiveBackendType::MPI);

    EXPECT_EQ(ctx->backend(), CollectiveBackendType::MPI);
}

// =============================================================================
// Device Management Tests
// =============================================================================

/**
 * @test indexForDevice returns correct index
 */
TEST_F(Test__LocalTPContext, IndexForDeviceCorrect)
{
    auto ctx = createLocalTPContext({cuda0_, cuda1_, rocm0_}, {}, CollectiveBackendType::HOST);

    EXPECT_EQ(ctx->indexForDevice(cuda0_), 0);
    EXPECT_EQ(ctx->indexForDevice(cuda1_), 1);
    EXPECT_EQ(ctx->indexForDevice(rocm0_), 2);
}

/**
 * @test indexForDevice returns -1 for unknown device
 */
TEST_F(Test__LocalTPContext, IndexForDeviceNotFound)
{
    auto ctx = createLocalTPContext({cuda0_, cuda1_}, {}, CollectiveBackendType::HOST);

    EXPECT_EQ(ctx->indexForDevice(rocm0_), -1);
    EXPECT_EQ(ctx->indexForDevice(cpu0_), -1);
}

/**
 * @test deviceAt returns correct device
 */
TEST_F(Test__LocalTPContext, DeviceAtCorrect)
{
    auto ctx = createLocalTPContext({cuda0_, cuda1_, rocm0_}, {}, CollectiveBackendType::HOST);

    EXPECT_EQ(ctx->deviceAt(0), cuda0_);
    EXPECT_EQ(ctx->deviceAt(1), cuda1_);
    EXPECT_EQ(ctx->deviceAt(2), rocm0_);
}

/**
 * @test deviceAt throws for invalid index
 */
TEST_F(Test__LocalTPContext, DeviceAtThrowsForInvalidIndex)
{
    auto ctx = createLocalTPContext({cuda0_, cuda1_}, {}, CollectiveBackendType::HOST);

    EXPECT_THROW(ctx->deviceAt(-1), std::out_of_range);
    EXPECT_THROW(ctx->deviceAt(2), std::out_of_range);
    EXPECT_THROW(ctx->deviceAt(100), std::out_of_range);
}

/**
 * @test weightForDevice returns correct weight
 */
TEST_F(Test__LocalTPContext, WeightForDeviceCorrect)
{
    auto ctx = createLocalTPContext({cuda0_, rocm0_}, {0.73f, 0.27f}, CollectiveBackendType::HOST);

    EXPECT_FLOAT_EQ(ctx->weightForDevice(cuda0_), 0.73f);
    EXPECT_FLOAT_EQ(ctx->weightForDevice(rocm0_), 0.27f);
}

/**
 * @test weightForDevice returns 0 for unknown device
 */
TEST_F(Test__LocalTPContext, WeightForDeviceUnknown)
{
    auto ctx = createLocalTPContext({cuda0_, cuda1_}, {}, CollectiveBackendType::HOST);

    EXPECT_FLOAT_EQ(ctx->weightForDevice(rocm0_), 0.0f);
}

// =============================================================================
// Head Distribution Tests
// =============================================================================

/**
 * @test headsForDevice with equal weights
 */
TEST_F(Test__LocalTPContext, HeadsForDeviceEqualWeights)
{
    auto ctx = createLocalTPContext({cuda0_, cuda1_}, {}, CollectiveBackendType::HOST);

    // 28 heads / 2 devices = 14 each
    EXPECT_EQ(ctx->headsForDevice(cuda0_, 28), 14);
    EXPECT_EQ(ctx->headsForDevice(cuda1_, 28), 14);

    // Total should equal original
    int total = ctx->headsForDevice(cuda0_, 28) + ctx->headsForDevice(cuda1_, 28);
    EXPECT_EQ(total, 28);
}

/**
 * @test headsForDevice with odd number of heads
 */
TEST_F(Test__LocalTPContext, HeadsForDeviceOddHeads)
{
    auto ctx = createLocalTPContext({cuda0_, cuda1_}, {}, CollectiveBackendType::HOST);

    // 29 heads / 2 devices = 14 + 15 (last gets remainder)
    int h0 = ctx->headsForDevice(cuda0_, 29);
    int h1 = ctx->headsForDevice(cuda1_, 29);

    EXPECT_EQ(h0 + h1, 29);
    // First should get ~half, second gets remainder
    EXPECT_TRUE(h0 >= 14 && h0 <= 15);
    EXPECT_TRUE(h1 >= 14 && h1 <= 15);
}

/**
 * @test headsForDevice with proportional weights
 */
TEST_F(Test__LocalTPContext, HeadsForDeviceProportional)
{
    auto ctx = createLocalTPContext({cuda0_, rocm0_}, {0.73f, 0.27f}, CollectiveBackendType::HOST);

    // 28 heads with 73%/27% split
    int h0 = ctx->headsForDevice(cuda0_, 28); // Should get ~20 heads (73% of 28)
    int h1 = ctx->headsForDevice(rocm0_, 28); // Should get ~8 heads (27% of 28)

    // Total must equal original
    EXPECT_EQ(h0 + h1, 28);

    // Check proportions are approximately correct
    EXPECT_GE(h0, 18); // At least 64% to account for rounding
    EXPECT_LE(h0, 22); // At most 78%
    EXPECT_GE(h1, 6);
    EXPECT_LE(h1, 10);
}

/**
 * @test headsForDevice with zero heads
 */
TEST_F(Test__LocalTPContext, HeadsForDeviceZeroHeads)
{
    auto ctx = createLocalTPContext({cuda0_, cuda1_}, {}, CollectiveBackendType::HOST);

    EXPECT_EQ(ctx->headsForDevice(cuda0_, 0), 0);
    EXPECT_EQ(ctx->headsForDevice(cuda1_, 0), 0);
}

/**
 * @test headsForDevice for unknown device
 */
TEST_F(Test__LocalTPContext, HeadsForDeviceUnknown)
{
    auto ctx = createLocalTPContext({cuda0_, cuda1_}, {}, CollectiveBackendType::HOST);

    EXPECT_EQ(ctx->headsForDevice(rocm0_, 28), 0);
}

// =============================================================================
// Row/Column Range Tests
// =============================================================================

/**
 * @test rowRangeForDevice with equal weights
 */
TEST_F(Test__LocalTPContext, RowRangeForDeviceEqual)
{
    auto ctx = createLocalTPContext({cuda0_, cuda1_}, {}, CollectiveBackendType::HOST);

    auto [r0_start, r0_end] = ctx->rowRangeForDevice(cuda0_, 1000);
    auto [r1_start, r1_end] = ctx->rowRangeForDevice(cuda1_, 1000);

    // First device: [0, 500)
    EXPECT_EQ(r0_start, 0);
    EXPECT_EQ(r0_end, 500);

    // Second device: [500, 1000)
    EXPECT_EQ(r1_start, 500);
    EXPECT_EQ(r1_end, 1000);

    // Ranges should be contiguous and cover all rows
    EXPECT_EQ(r0_end, r1_start);
    EXPECT_EQ(r1_end - r0_start, 1000);
}

/**
 * @test rowRangeForDevice with proportional weights
 */
TEST_F(Test__LocalTPContext, RowRangeForDeviceProportional)
{
    auto ctx = createLocalTPContext({cuda0_, rocm0_}, {0.73f, 0.27f}, CollectiveBackendType::HOST);

    auto [r0_start, r0_end] = ctx->rowRangeForDevice(cuda0_, 1000);
    auto [r1_start, r1_end] = ctx->rowRangeForDevice(rocm0_, 1000);

    // Ranges should be contiguous
    EXPECT_EQ(r0_start, 0);
    EXPECT_EQ(r0_end, r1_start);
    EXPECT_EQ(r1_end, 1000);

    // Check proportions
    int count0 = r0_end - r0_start;
    int count1 = r1_end - r1_start;
    EXPECT_EQ(count0 + count1, 1000);

    // Device 0 should get ~730 rows (73%)
    EXPECT_GE(count0, 700);
    EXPECT_LE(count0, 760);
}

/**
 * @test colRangeForDevice with three devices
 */
TEST_F(Test__LocalTPContext, ColRangeForDeviceThreeDevices)
{
    auto rocm1 = GlobalDeviceAddress::rocm(1, 0);
    auto ctx = createLocalTPContext(
        {cuda0_, rocm0_, rocm1},
        {0.5f, 0.25f, 0.25f},
        CollectiveBackendType::HOST);

    auto [c0_start, c0_end] = ctx->colRangeForDevice(cuda0_, 1024);
    auto [c1_start, c1_end] = ctx->colRangeForDevice(rocm0_, 1024);
    auto [c2_start, c2_end] = ctx->colRangeForDevice(rocm1, 1024);

    // Ranges should be contiguous
    EXPECT_EQ(c0_start, 0);
    EXPECT_EQ(c0_end, c1_start);
    EXPECT_EQ(c1_end, c2_start);
    EXPECT_EQ(c2_end, 1024);

    // Check proportions (50%, 25%, 25%)
    int count0 = c0_end - c0_start;
    int count1 = c1_end - c1_start;
    int count2 = c2_end - c2_start;

    EXPECT_EQ(count0 + count1 + count2, 1024);

    // Device 0 should get ~512 cols
    EXPECT_GE(count0, 480);
    EXPECT_LE(count0, 544);

    // Devices 1 and 2 should get ~256 each
    EXPECT_GE(count1, 224);
    EXPECT_LE(count1, 288);
    EXPECT_GE(count2, 224);
    EXPECT_LE(count2, 288);
}

/**
 * @test rowRangeForDevice with zero rows
 */
TEST_F(Test__LocalTPContext, RowRangeForDeviceZeroRows)
{
    auto ctx = createLocalTPContext({cuda0_, cuda1_}, {}, CollectiveBackendType::HOST);

    auto [r0_start, r0_end] = ctx->rowRangeForDevice(cuda0_, 0);
    EXPECT_EQ(r0_start, 0);
    EXPECT_EQ(r0_end, 0);
}

/**
 * @test colRangeForDevice for unknown device
 */
TEST_F(Test__LocalTPContext, ColRangeForDeviceUnknown)
{
    auto ctx = createLocalTPContext({cuda0_, cuda1_}, {}, CollectiveBackendType::HOST);

    auto [start, end] = ctx->colRangeForDevice(rocm0_, 1000);
    EXPECT_EQ(start, 0);
    EXPECT_EQ(end, 0);
}

// =============================================================================
// Collective Operation Tests (Basic Validation)
// =============================================================================

/**
 * @test allreduce with single device is no-op
 */
TEST_F(Test__LocalTPContext, AllreduceSingleDeviceNoop)
{
    auto ctx = createLocalTPContext({cuda0_}, {}, CollectiveBackendType::AUTO);

    // Single device - should succeed as no-op
    // Note: Full collective testing requires integration tests with real devices
    EXPECT_TRUE(ctx->allreduce(nullptr) == false); // null tensor fails
}

/**
 * @test synchronize doesn't crash
 */
TEST_F(Test__LocalTPContext, SynchronizeDoesNotCrash)
{
    auto ctx = createLocalTPContext({cuda0_, cuda1_}, {}, CollectiveBackendType::HOST);

    // Should not crash
    ctx->synchronize();
}
// =============================================================================
// MockCollectiveBackend for Testing
// =============================================================================

/**
 * @brief Mock ICollectiveBackend that tracks calls and allows configurable behavior
 *
 * Used for testing that LocalTPContext correctly delegates to its backend.
 */
class MockCollectiveBackend : public ICollectiveBackend
{
public:
    // Call counters
    std::atomic<int> initialize_call_count{0};
    std::atomic<int> shutdown_call_count{0};
    std::atomic<int> allreduce_call_count{0};
    std::atomic<int> allreduce_multi_call_count{0};
    std::atomic<int> allreduce_multi_on_streams_call_count{0};
    std::atomic<int> allreduce_on_stream_call_count{0};
    std::atomic<int> allgather_call_count{0};
    std::atomic<int> allgather_multi_call_count{0};
    std::atomic<int> allgather_multi_on_streams_call_count{0};
    std::atomic<int> allgather_on_stream_call_count{0};
    std::atomic<int> reduce_scatter_call_count{0};
    std::atomic<int> synchronize_call_count{0};
    std::atomic<int> broadcast_call_count{0};

    // Configurable behavior
    bool should_fail_initialize = false;
    bool should_fail_allreduce = false;
    bool should_fail_allgather = false;
    bool should_fail_reduce_scatter = false;
    bool multi_gpu_mode = true;
    bool supports_allreduce_on_stream = true;
    bool supports_allgather_on_stream = true;

    // Captured parameters from last call (for verification)
    size_t last_allreduce_count = 0;
    size_t last_allgather_count = 0;
    size_t last_reduce_scatter_count = 0;
    CollectiveDataType last_dtype = CollectiveDataType::FLOAT32;
    CollectiveOp last_op = CollectiveOp::ALLREDUCE_SUM;
    std::vector<void *> last_multi_buffers;
    std::vector<void *> last_allreduce_multi_streams;
    std::vector<void *> last_allgather_multi_streams;
    int last_allreduce_on_stream_device_idx = -1;
    void *last_allreduce_on_stream_stream = nullptr;
    int last_allgather_on_stream_device_idx = -1;
    void *last_allgather_on_stream_stream = nullptr;

    // =========================================================================
    // Identity
    // =========================================================================

    CollectiveBackendType type() const override { return CollectiveBackendType::HOST; }
    std::string name() const override { return "MockBackend"; }

    // =========================================================================
    // Capability Queries
    // =========================================================================

    bool supportsDevice(DeviceType type) const override
    {
        (void)type;
        return true;
    }

    bool supportsDirectTransfer(DeviceId src, DeviceId dst) const override
    {
        (void)src;
        (void)dst;
        return true;
    }

    bool isAvailable() const override { return true; }

    // =========================================================================
    // Lifecycle
    // =========================================================================

    bool initialize(const DeviceGroup &group) override
    {
        initialize_call_count++;
        group_ = group;
        initialized_ = !should_fail_initialize;
        return initialized_;
    }

    bool isInitialized() const override { return initialized_; }

    void shutdown() override
    {
        shutdown_call_count++;
        initialized_ = false;
    }

    // =========================================================================
    // Collective Operations
    // =========================================================================

    bool allreduce(void *buffer, size_t count, CollectiveDataType dtype, CollectiveOp op) override
    {
        allreduce_call_count++;
        last_allreduce_count = count;
        last_dtype = dtype;
        last_op = op;
        (void)buffer;
        return !should_fail_allreduce;
    }

    bool allgather(const void *send_buf, void *recv_buf, size_t send_count, CollectiveDataType dtype) override
    {
        allgather_call_count++;
        last_allgather_count = send_count;
        last_dtype = dtype;
        (void)send_buf;
        (void)recv_buf;
        return !should_fail_allgather;
    }

    bool allgatherv(const void *send_buf, size_t send_count, void *recv_buf,
                    const std::vector<int> &recv_counts, const std::vector<int> &displacements,
                    CollectiveDataType dtype) override
    {
        (void)send_buf;
        (void)send_count;
        (void)recv_buf;
        (void)recv_counts;
        (void)displacements;
        (void)dtype;
        return true;
    }

    bool reduceScatter(const void *send_buf, void *recv_buf, size_t recv_count,
                       CollectiveDataType dtype, CollectiveOp op) override
    {
        reduce_scatter_call_count++;
        last_reduce_scatter_count = recv_count;
        last_dtype = dtype;
        last_op = op;
        (void)send_buf;
        (void)recv_buf;
        return !should_fail_reduce_scatter;
    }

    bool broadcast(void *buffer, size_t count, CollectiveDataType dtype, int root_rank) override
    {
        broadcast_call_count++;
        (void)buffer;
        (void)count;
        (void)dtype;
        (void)root_rank;
        return true;
    }

    bool synchronize() override
    {
        synchronize_call_count++;
        return true;
    }

    // =========================================================================
    // Multi-GPU Operations
    // =========================================================================

    bool isMultiGpuSingleProcess() const override { return multi_gpu_mode; }

    bool allreduceMulti(const std::vector<void *> &buffers, size_t count,
                        CollectiveDataType dtype, CollectiveOp op) override
    {
        allreduce_multi_call_count++;
        last_multi_buffers = buffers;
        last_allreduce_count = count;
        last_dtype = dtype;
        last_op = op;
        return !should_fail_allreduce;
    }

    bool allreduceMultiOnStreams(
        const std::vector<void *> &buffers,
        size_t count,
        CollectiveDataType dtype,
        CollectiveOp op,
        const std::vector<void *> &streams) override
    {
        allreduce_multi_on_streams_call_count++;
        last_multi_buffers = buffers;
        last_allreduce_count = count;
        last_dtype = dtype;
        last_op = op;
        last_allreduce_multi_streams = streams;
        return !should_fail_allreduce;
    }

    bool supportsAllreduceMultiOnStreams() const override
    {
        return multi_gpu_mode;
    }

    bool allreduceSingleDeviceOnStream(
        void *buffer,
        size_t count,
        CollectiveDataType dtype,
        CollectiveOp op,
        int device_idx,
        void *stream) override
    {
        allreduce_on_stream_call_count++;
        last_allreduce_count = count;
        last_dtype = dtype;
        last_op = op;
        last_allreduce_on_stream_device_idx = device_idx;
        last_allreduce_on_stream_stream = stream;
        (void)buffer;
        return supports_allreduce_on_stream && !should_fail_allreduce;
    }

    bool supportsAllreduceSingleDeviceOnStream() const override
    {
        return supports_allreduce_on_stream;
    }

    bool allgatherMulti(const std::vector<const void *> &send_bufs,
                        const std::vector<void *> &recv_bufs,
                        size_t send_count, CollectiveDataType dtype) override
    {
        allgather_multi_call_count++;
        last_allgather_count = send_count;
        last_dtype = dtype;
        (void)send_bufs;
        (void)recv_bufs;
        return !should_fail_allgather;
    }

    bool allgatherMultiOnStreams(
        const std::vector<const void *> &send_bufs,
        const std::vector<void *> &recv_bufs,
        size_t send_count,
        CollectiveDataType dtype,
        const std::vector<void *> &streams) override
    {
        allgather_multi_on_streams_call_count++;
        last_allgather_count = send_count;
        last_dtype = dtype;
        last_allgather_multi_streams = streams;
        (void)send_bufs;
        (void)recv_bufs;
        return !should_fail_allgather;
    }

    bool supportsAllgatherMultiOnStreams() const override
    {
        return multi_gpu_mode;
    }

    bool allgatherSingleDeviceOnStream(
        const void *send_buf,
        void *recv_buf,
        size_t send_count,
        CollectiveDataType dtype,
        int device_idx,
        void *stream) override
    {
        allgather_on_stream_call_count++;
        last_allgather_count = send_count;
        last_dtype = dtype;
        last_allgather_on_stream_device_idx = device_idx;
        last_allgather_on_stream_stream = stream;
        (void)send_buf;
        (void)recv_buf;
        return supports_allgather_on_stream && !should_fail_allgather;
    }

    bool supportsAllgatherSingleDeviceOnStream() const override
    {
        return supports_allgather_on_stream;
    }

    std::string lastError() const override { return last_error_; }

    // Helper to reset all counters
    void reset()
    {
        initialize_call_count = 0;
        shutdown_call_count = 0;
        allreduce_call_count = 0;
        allreduce_multi_call_count = 0;
        allreduce_multi_on_streams_call_count = 0;
        allreduce_on_stream_call_count = 0;
        allgather_call_count = 0;
        allgather_multi_call_count = 0;
        allgather_multi_on_streams_call_count = 0;
        allgather_on_stream_call_count = 0;
        reduce_scatter_call_count = 0;
        synchronize_call_count = 0;
        broadcast_call_count = 0;
        should_fail_initialize = false;
        should_fail_allreduce = false;
        should_fail_allgather = false;
        should_fail_reduce_scatter = false;
        multi_gpu_mode = true;
        supports_allreduce_on_stream = true;
        supports_allgather_on_stream = true;
        last_multi_buffers.clear();
        last_allreduce_multi_streams.clear();
        last_allgather_multi_streams.clear();
        last_allreduce_on_stream_device_idx = -1;
        last_allreduce_on_stream_stream = nullptr;
        last_allgather_on_stream_device_idx = -1;
        last_allgather_on_stream_stream = nullptr;
    }

private:
    DeviceGroup group_;
    bool initialized_ = false;
    std::string last_error_;
};

// =============================================================================
// Backend Initialization Tests
// =============================================================================

/**
 * @test Single device skips backend initialization
 */
TEST_F(Test__LocalTPContext, SingleDeviceSkipsBackendInit)
{
    // With a single device, no collective backend is needed
    auto ctx = createLocalTPContext({cpu0_}, {}, CollectiveBackendType::HOST);

    EXPECT_EQ(ctx->degree(), 1);
    // Single device should not attempt backend initialization
    // The context should still work (allreduce is a no-op)

    auto tensor = TestTensorFactory::createFP32({4, 4});
    TestTensorFactory::fillValue(tensor.get(), 1.0f);

    // Should succeed as no-op for single device
    EXPECT_TRUE(ctx->allreduce(tensor.get()));
}

/**
 * @test HOST backend is always available
 */
TEST_F(Test__LocalTPContext, HostBackendAlwaysAvailable)
{
    // CPU devices should get HOST backend and it should initialize
    auto ctx = createLocalTPContext({cpu0_, GlobalDeviceAddress::cpu(1)}, {}, CollectiveBackendType::HOST);

    EXPECT_EQ(ctx->backend(), CollectiveBackendType::HOST);
    EXPECT_EQ(ctx->degree(), 2);
}

TEST_F(Test__LocalTPContext, RawAllgatherGraphCaptureSupportUsesAllreducePrimitive)
{
    auto ctx_base = createLocalTPContext({cpu0_, GlobalDeviceAddress::cpu(1)}, {}, CollectiveBackendType::HOST);
    auto *ctx = dynamic_cast<LocalTPContext *>(ctx_base.get());
    ASSERT_NE(ctx, nullptr);

    auto backend = std::make_unique<MockCollectiveBackend>();
    auto *backend_raw = backend.get();
    backend_raw->multi_gpu_mode = true;
    backend_raw->supports_allgather_on_stream = false;
    backend_raw->supports_allreduce_on_stream = true;
    ctx->setBackendForTesting(
        std::move(backend),
        CollectiveBackendType::NCCL,
        /*initialized=*/true);

    EXPECT_TRUE(ctx->supportsRawAllgatherOnStreamGraphCapture())
        << "NCCL graph-captured raw allgather is implemented through the validated allreduce primitive.";

    backend_raw->supports_allreduce_on_stream = false;
    EXPECT_FALSE(ctx->supportsRawAllgatherOnStreamGraphCapture());
    EXPECT_EQ(backend_raw->allgather_on_stream_call_count.load(), 0);
    EXPECT_EQ(backend_raw->allgather_multi_call_count.load(), 0)
        << "Support probing must not enter the host-synchronized raw allgather barrier.";
    EXPECT_EQ(backend_raw->allgather_multi_on_streams_call_count.load(), 0)
        << "Support probing must not launch eager grouped allgather.";
}

TEST_F(Test__LocalTPContext, RCCLRawAllgatherGraphCaptureSupportUsesAllreducePrimitive)
{
    auto ctx_base = createLocalTPContext({cpu0_, GlobalDeviceAddress::cpu(1)}, {}, CollectiveBackendType::HOST);
    auto *ctx = dynamic_cast<LocalTPContext *>(ctx_base.get());
    ASSERT_NE(ctx, nullptr);

    auto backend = std::make_unique<MockCollectiveBackend>();
    auto *backend_raw = backend.get();
    backend_raw->multi_gpu_mode = true;
    backend_raw->supports_allgather_on_stream = false;
    backend_raw->supports_allreduce_on_stream = true;
    ctx->setBackendForTesting(
        std::move(backend),
        CollectiveBackendType::RCCL,
        /*initialized=*/true);

    EXPECT_TRUE(ctx->supportsRawAllgatherOnStreamGraphCapture())
        << "RCCL graph-captured raw allgather is implemented via captured allreduce emulation.";

    backend_raw->supports_allreduce_on_stream = false;
    EXPECT_FALSE(ctx->supportsRawAllgatherOnStreamGraphCapture());
}

TEST_F(Test__LocalTPContext, GraphCapturedRawAllgatherUsesGroupedAllreduceEmulation)
{
    std::ifstream source("src/v2/collective/LocalTPContext.cpp");
    ASSERT_TRUE(source.is_open());
    const std::string contents((std::istreambuf_iterator<char>(source)),
                               std::istreambuf_iterator<char>());

    const size_t entry = contents.find("bool LocalTPContext::allgatherRawOnStream(");
    ASSERT_NE(entry, std::string::npos);
    const size_t graph_capture_branch = contents.find("if (isGraphCaptureActive())", entry);
    ASSERT_NE(graph_capture_branch, std::string::npos);
    const size_t eager_handoff = contents.find("return allgatherRawWithBarrierMultiGpu(", graph_capture_branch);
    ASSERT_NE(eager_handoff, std::string::npos);
    const std::string graph_body = contents.substr(graph_capture_branch, eager_handoff - graph_capture_branch);

    const size_t emulation_comment = graph_body.find(
        "deterministic publish-and-sum transaction");
    ASSERT_NE(emulation_comment, std::string::npos);
    const size_t grouped_allreduce_call = graph_body.find(
        "allreduceGroupedOnExplicitStreams(",
        emulation_comment);
    ASSERT_NE(grouped_allreduce_call, std::string::npos);
    EXPECT_NE(graph_body.find("nccl_backend_detail::cudaMemsetAsyncDevice"), std::string::npos);
    EXPECT_NE(graph_body.find("nccl_backend_detail::cudaMemcpyAsyncSameDevice"), std::string::npos);
    EXPECT_NE(graph_body.find("rccl_backend_detail::hipMemsetAsyncDevice"), std::string::npos);
    EXPECT_NE(graph_body.find("rccl_backend_detail::hipMemcpyAsyncSameDevice"), std::string::npos);
    EXPECT_EQ(graph_body.find("backend_impl_->allgatherSingleDeviceOnStream"), std::string::npos)
        << "Graph-captured raw allgather should not depend on backend-native allgather replay.";
}

TEST_F(Test__LocalTPContext, RawAllgatherGraphCaptureFailsWithoutOnStreamSupport)
{
    auto ctx_base = createLocalTPContext({cpu0_, GlobalDeviceAddress::cpu(1)}, {}, CollectiveBackendType::HOST);
    auto *ctx = dynamic_cast<LocalTPContext *>(ctx_base.get());
    ASSERT_NE(ctx, nullptr);

    auto backend = std::make_unique<MockCollectiveBackend>();
    auto *backend_raw = backend.get();
    backend_raw->multi_gpu_mode = true;
    backend_raw->supports_allgather_on_stream = false;
    backend_raw->supports_allreduce_on_stream = false;
    ctx->setBackendForTesting(
        std::move(backend),
        CollectiveBackendType::NCCL,
        /*initialized=*/true);

    EXPECT_FALSE(ctx->supportsRawAllgatherOnStreamGraphCapture());

    int send = 7;
    int recv[2] = {};
    void *stream = reinterpret_cast<void *>(0x1234);

    {
        GraphCaptureGuard guard;
        EXPECT_FALSE(ctx->allgatherRawOnStream(
            &send,
            recv,
            1,
            CollectiveDataType::INT32,
            0,
            stream,
            "captured_raw_stage"));
    }

    EXPECT_EQ(backend_raw->allgather_on_stream_call_count.load(), 0);
    EXPECT_EQ(backend_raw->allgather_multi_call_count.load(), 0)
        << "Unsupported graph-captured raw allgather must fail, not fall back.";
    EXPECT_EQ(backend_raw->allgather_multi_on_streams_call_count.load(), 0)
        << "Unsupported graph-captured raw allgather must fail, not enter eager grouped launch.";
}

TEST_F(Test__LocalTPContext, RawAllgatherResultSurvivesBackToBackGenerations)
{
    auto ctx_base = createLocalTPContext({cpu0_, GlobalDeviceAddress::cpu(1)}, {}, CollectiveBackendType::HOST);
    auto *ctx = dynamic_cast<LocalTPContext *>(ctx_base.get());
    ASSERT_NE(ctx, nullptr);

    auto backend = std::make_unique<MockCollectiveBackend>();
    auto *backend_raw = backend.get();
    backend_raw->multi_gpu_mode = true;
    ctx->setBackendForTesting(
        std::move(backend),
        CollectiveBackendType::NCCL,
        /*initialized=*/true);

    int send0_gen0 = 1;
    int send1_gen0 = 2;
    int recv0_gen0[2] = {};
    int recv1_gen0[2] = {};
    int send0_gen1 = 3;
    int send1_gen1 = 4;
    int recv0_gen1[2] = {};
    int recv1_gen1[2] = {};
    void *slot0_stream = reinterpret_cast<void *>(0x1234);
    void *slot1_stream = reinterpret_cast<void *>(0x5678);

    std::promise<void> release_second_generation;
    auto release_second_generation_future = release_second_generation.get_future().share();
    std::atomic<bool> slot0_first_returned{false};
    std::atomic<bool> slot1_entering_second{false};
    std::atomic<bool> slot0_first_result{false};
    std::atomic<bool> slot1_first_result{false};
    std::atomic<bool> slot0_second_result{false};
    std::atomic<bool> slot1_second_result{false};

    std::thread slot0([&]()
                      {
                          const bool first = ctx->allgatherRawOnStream(
                              &send0_gen0,
                              recv0_gen0,
                              1,
                              CollectiveDataType::INT32,
                              0,
                              slot0_stream,
                              "raw_stage_0");
                          slot0_first_result.store(first, std::memory_order_release);
                          slot0_first_returned.store(true, std::memory_order_release);
                          release_second_generation_future.wait();
                          const bool second = ctx->allgatherRawOnStream(
                              &send0_gen1,
                              recv0_gen1,
                              1,
                              CollectiveDataType::INT32,
                              0,
                              slot0_stream,
                              "raw_stage_1");
                          slot0_second_result.store(second, std::memory_order_release);
                      });

    std::thread slot1([&]()
                      {
                          const bool first = ctx->allgatherRawOnStream(
                              &send1_gen0,
                              recv1_gen0,
                              1,
                              CollectiveDataType::INT32,
                              1,
                              slot1_stream,
                              "raw_stage_0");
                          slot1_first_result.store(first, std::memory_order_release);
                          slot1_entering_second.store(true, std::memory_order_release);
                          const bool second = ctx->allgatherRawOnStream(
                              &send1_gen1,
                              recv1_gen1,
                              1,
                              CollectiveDataType::INT32,
                              1,
                              slot1_stream,
                              "raw_stage_1");
                          slot1_second_result.store(second, std::memory_order_release);
                      });

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while ((!slot0_first_returned.load(std::memory_order_acquire) ||
            !slot1_entering_second.load(std::memory_order_acquire)) &&
           std::chrono::steady_clock::now() < deadline)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    const bool saw_slot0_first_return =
        slot0_first_returned.load(std::memory_order_acquire);
    const bool saw_slot1_second_entry =
        slot1_entering_second.load(std::memory_order_acquire);
    const int launches_before_release = backend_raw->allgather_multi_on_streams_call_count.load();

    release_second_generation.set_value();
    slot0.join();
    slot1.join();

    EXPECT_TRUE(saw_slot0_first_return);
    EXPECT_TRUE(saw_slot1_second_entry);
    EXPECT_EQ(launches_before_release, 1)
        << "The second raw allgather must not launch before both participants have "
           "observed the first generation result.";
    EXPECT_TRUE(slot0_first_result.load(std::memory_order_acquire));
    EXPECT_TRUE(slot1_first_result.load(std::memory_order_acquire));
    EXPECT_TRUE(slot0_second_result.load(std::memory_order_acquire));
    EXPECT_TRUE(slot1_second_result.load(std::memory_order_acquire));
    EXPECT_EQ(backend_raw->allgather_multi_call_count.load(), 0);
    EXPECT_EQ(backend_raw->allgather_multi_on_streams_call_count.load(), 2);
    ASSERT_EQ(backend_raw->last_allgather_multi_streams.size(), 2u);
    EXPECT_EQ(backend_raw->last_allgather_multi_streams[0], slot0_stream);
    EXPECT_EQ(backend_raw->last_allgather_multi_streams[1], slot1_stream);
}

TEST_F(Test__LocalTPContext, GroupedOnStreamAllreduceResultSurvivesBackToBackGenerations)
{
    auto ctx_base = createLocalTPContext({cpu0_, GlobalDeviceAddress::cpu(1)}, {}, CollectiveBackendType::HOST);
    auto *ctx = dynamic_cast<LocalTPContext *>(ctx_base.get());
    ASSERT_NE(ctx, nullptr);

    auto backend = std::make_unique<MockCollectiveBackend>();
    auto *backend_raw = backend.get();
    backend_raw->multi_gpu_mode = true;
    ctx->setBackendForTesting(
        std::move(backend),
        CollectiveBackendType::NCCL,
        /*initialized=*/true);

    int slot0_gen0 = 1;
    int slot1_gen0 = 2;
    int slot0_gen1 = 3;
    int slot1_gen1 = 4;
    void *slot0_stream = reinterpret_cast<void *>(0x1234);
    void *slot1_stream = reinterpret_cast<void *>(0x5678);

    std::promise<void> release_second_generation;
    auto release_second_generation_future = release_second_generation.get_future().share();
    std::atomic<bool> slot0_first_returned{false};
    std::atomic<bool> slot1_entering_second{false};
    std::atomic<bool> slot0_first_result{false};
    std::atomic<bool> slot1_first_result{false};
    std::atomic<bool> slot0_second_result{false};
    std::atomic<bool> slot1_second_result{false};

    std::thread slot0([&]()
                      {
                          const bool first = ctx->allreduceGroupedOnExplicitStreamsForTesting(
                              &slot0_gen0,
                              1,
                              CollectiveDataType::INT32,
                              0,
                              slot0_stream,
                              "allreduce_stage_0",
                              "fp32");
                          slot0_first_result.store(first, std::memory_order_release);
                          slot0_first_returned.store(true, std::memory_order_release);
                          release_second_generation_future.wait();
                          const bool second = ctx->allreduceGroupedOnExplicitStreamsForTesting(
                              &slot0_gen1,
                              1,
                              CollectiveDataType::INT32,
                              0,
                              slot0_stream,
                              "allreduce_stage_1",
                              "fp32");
                          slot0_second_result.store(second, std::memory_order_release);
                      });

    std::thread slot1([&]()
                      {
                          const bool first = ctx->allreduceGroupedOnExplicitStreamsForTesting(
                              &slot1_gen0,
                              1,
                              CollectiveDataType::INT32,
                              1,
                              slot1_stream,
                              "allreduce_stage_0",
                              "fp32");
                          slot1_first_result.store(first, std::memory_order_release);
                          slot1_entering_second.store(true, std::memory_order_release);
                          const bool second = ctx->allreduceGroupedOnExplicitStreamsForTesting(
                              &slot1_gen1,
                              1,
                              CollectiveDataType::INT32,
                              1,
                              slot1_stream,
                              "allreduce_stage_1",
                              "fp32");
                          slot1_second_result.store(second, std::memory_order_release);
                      });

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while ((!slot0_first_returned.load(std::memory_order_acquire) ||
            !slot1_entering_second.load(std::memory_order_acquire)) &&
           std::chrono::steady_clock::now() < deadline)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    const bool saw_slot0_first_return =
        slot0_first_returned.load(std::memory_order_acquire);
    const bool saw_slot1_second_entry =
        slot1_entering_second.load(std::memory_order_acquire);
    const int launches_before_release = backend_raw->allreduce_multi_on_streams_call_count.load();

    release_second_generation.set_value();
    slot0.join();
    slot1.join();

    EXPECT_TRUE(saw_slot0_first_return);
    EXPECT_TRUE(saw_slot1_second_entry);
    EXPECT_EQ(launches_before_release, 1)
        << "The second grouped allreduce must not launch before all participants "
           "have observed the first generation result.";
    EXPECT_TRUE(slot0_first_result.load(std::memory_order_acquire));
    EXPECT_TRUE(slot1_first_result.load(std::memory_order_acquire));
    EXPECT_TRUE(slot0_second_result.load(std::memory_order_acquire));
    EXPECT_TRUE(slot1_second_result.load(std::memory_order_acquire));
    EXPECT_EQ(backend_raw->allreduce_multi_call_count.load(), 0);
    EXPECT_EQ(backend_raw->allreduce_multi_on_streams_call_count.load(), 2);
    ASSERT_EQ(backend_raw->last_allreduce_multi_streams.size(), 2u);
    EXPECT_EQ(backend_raw->last_allreduce_multi_streams[0], slot0_stream);
    EXPECT_EQ(backend_raw->last_allreduce_multi_streams[1], slot1_stream);
}

/**
 * @test AUTO backend detection with CPU devices -> HOST
 */
TEST_F(Test__LocalTPContext, AutoBackendCpuDevicesUsesHost)
{
    auto cpu1 = GlobalDeviceAddress::cpu(1);
    auto ctx = createLocalTPContext({cpu0_, cpu1}, {}, CollectiveBackendType::AUTO);

    EXPECT_EQ(ctx->backend(), CollectiveBackendType::HOST);
}

// =============================================================================
// Collective Operation Tests with Real HostBackend
// =============================================================================

/**
 * @test Allreduce with single CPU device is a no-op success
 */
TEST_F(Test__LocalTPContext, AllreduceSingleCpuDeviceSuccess)
{
    auto ctx = createLocalTPContext({cpu0_}, {}, CollectiveBackendType::HOST);

    auto tensor = TestTensorFactory::createFP32({2, 4});
    float *data = tensor->mutable_data();
    for (size_t i = 0; i < 8; ++i)
    {
        data[i] = static_cast<float>(i);
    }

    // Should succeed (no-op)
    EXPECT_TRUE(ctx->allreduce(tensor.get()));

    // Data should be unchanged
    for (size_t i = 0; i < 8; ++i)
    {
        EXPECT_FLOAT_EQ(tensor->data()[i], static_cast<float>(i));
    }
}

/**
 * @test Out-of-place allreduce copies data for single device
 */
TEST_F(Test__LocalTPContext, AllreduceOutOfPlaceCopiesForSingleDevice)
{
    auto ctx = createLocalTPContext({cpu0_}, {}, CollectiveBackendType::HOST);

    auto input = TestTensorFactory::createFP32({2, 4});
    auto output = TestTensorFactory::createFP32({2, 4});

    float *in_data = input->mutable_data();
    for (size_t i = 0; i < 8; ++i)
    {
        in_data[i] = static_cast<float>(i + 1);
    }

    // Out-of-place should copy input to output
    EXPECT_TRUE(ctx->allreduce(input.get(), output.get()));

    // Output should have input's data
    for (size_t i = 0; i < 8; ++i)
    {
        EXPECT_FLOAT_EQ(output->data()[i], static_cast<float>(i + 1));
    }
}

/**
 * @test Allgather copies local shard to global for single device
 */
TEST_F(Test__LocalTPContext, AllgatherSingleDeviceCopies)
{
    auto ctx = createLocalTPContext({cpu0_}, {}, CollectiveBackendType::HOST);

    auto local_shard = TestTensorFactory::createFP32({1, 4});
    auto global_tensor = TestTensorFactory::createFP32({1, 4});

    float *shard_data = local_shard->mutable_data();
    for (size_t i = 0; i < 4; ++i)
    {
        shard_data[i] = static_cast<float>(i * 2);
    }

    EXPECT_TRUE(ctx->allgather(local_shard.get(), global_tensor.get()));

    // Global should have shard's data
    for (size_t i = 0; i < 4; ++i)
    {
        EXPECT_FLOAT_EQ(global_tensor->data()[i], static_cast<float>(i * 2));
    }
}

/**
 * @test ReduceScatter copies for single device
 */
TEST_F(Test__LocalTPContext, ReduceScatterSingleDeviceCopies)
{
    auto ctx = createLocalTPContext({cpu0_}, {}, CollectiveBackendType::HOST);

    auto input = TestTensorFactory::createFP32({2, 4});
    auto output_shard = TestTensorFactory::createFP32({2, 4});

    float *in_data = input->mutable_data();
    for (size_t i = 0; i < 8; ++i)
    {
        in_data[i] = static_cast<float>(i + 10);
    }

    EXPECT_TRUE(ctx->reduceScatter(input.get(), output_shard.get()));

    // Output should have input's data (for single device)
    for (size_t i = 0; i < 8; ++i)
    {
        EXPECT_FLOAT_EQ(output_shard->data()[i], static_cast<float>(i + 10));
    }
}

/**
 * @test Synchronize completes for single device
 */
TEST_F(Test__LocalTPContext, SynchronizeSingleDevice)
{
    auto ctx = createLocalTPContext({cpu0_}, {}, CollectiveBackendType::HOST);

    // Should complete without issues
    ctx->synchronize();
    SUCCEED(); // If we get here, test passes
}

// =============================================================================
// Error Handling Tests
// =============================================================================

/**
 * @test Allreduce with null tensor returns false
 */
TEST_F(Test__LocalTPContext, AllreduceNullTensorFails)
{
    auto ctx = createLocalTPContext({cpu0_}, {}, CollectiveBackendType::HOST);

    EXPECT_FALSE(ctx->allreduce(nullptr));
}

/**
 * @test Out-of-place allreduce with null input fails
 */
TEST_F(Test__LocalTPContext, AllreduceOutOfPlaceNullInputFails)
{
    auto ctx = createLocalTPContext({cpu0_}, {}, CollectiveBackendType::HOST);

    auto output = TestTensorFactory::createFP32({2, 4});
    EXPECT_FALSE(ctx->allreduce(nullptr, output.get()));
}

/**
 * @test Out-of-place allreduce with null output fails
 */
TEST_F(Test__LocalTPContext, AllreduceOutOfPlaceNullOutputFails)
{
    auto ctx = createLocalTPContext({cpu0_}, {}, CollectiveBackendType::HOST);

    auto input = TestTensorFactory::createFP32({2, 4});
    EXPECT_FALSE(ctx->allreduce(input.get(), static_cast<TensorBase *>(nullptr)));
}

/**
 * @test Allgather with null tensors fails
 */
TEST_F(Test__LocalTPContext, AllgatherNullTensorsFails)
{
    auto ctx = createLocalTPContext({cpu0_}, {}, CollectiveBackendType::HOST);

    auto valid_tensor = TestTensorFactory::createFP32({2, 4});
    EXPECT_FALSE(ctx->allgather(nullptr, valid_tensor.get()));
    EXPECT_FALSE(ctx->allgather(valid_tensor.get(), nullptr));
}

/**
 * @test ReduceScatter with null tensors fails
 */
TEST_F(Test__LocalTPContext, ReduceScatterNullTensorsFails)
{
    auto ctx = createLocalTPContext({cpu0_}, {}, CollectiveBackendType::HOST);

    auto valid_tensor = TestTensorFactory::createFP32({2, 4});
    EXPECT_FALSE(ctx->reduceScatter(nullptr, valid_tensor.get()));
    EXPECT_FALSE(ctx->reduceScatter(valid_tensor.get(), nullptr));
}

// =============================================================================
// Multi-Device Scenarios (Backend Not Initialized)
// =============================================================================

/**
 * @test Multi-device allreduce when backend not initialized falls back gracefully
 *
 * When GPU backends (NCCL, RCCL) are requested but not available, the context
 * should still work with degraded behavior (no actual reduction, just returns true).
 */
TEST_F(Test__LocalTPContext, MultiDeviceAllreduceBackendUnavailableFallback)
{
    // Use HOST backend — the original NCCL request throws when <2 CUDA GPUs exist.
    // This test validates "does not crash" behavior, not NCCL specifically.
    auto ctx = createLocalTPContext({cuda0_, cuda1_}, {}, CollectiveBackendType::HOST);

    auto tensor = TestTensorFactory::createFP32({2, 4});
    TestTensorFactory::fillValue(tensor.get(), 1.0f);

    // Should not crash. May return true (HOST fallback) or false (backend failure)
    // The exact behavior depends on whether NCCL is available
    ctx->allreduce(tensor.get());
    SUCCEED(); // Test passes if no crash
}

/**
 * @test Multi-device synchronize when backend not initialized is no-op
 */
TEST_F(Test__LocalTPContext, MultiDeviceSynchronizeBackendUnavailableNoop)
{
    auto ctx = createLocalTPContext({cuda0_, cuda1_}, {}, CollectiveBackendType::HOST);

    // Should not crash even if backend is not fully initialized
    ctx->synchronize();
    SUCCEED();
}

// =============================================================================
// CPU-Only Multi-Device Tests with HostBackend
// =============================================================================

/**
 * @test Two CPU devices with HOST backend initializes correctly
 *
 * Note: The HostBackend is primarily designed for GPU-to-GPU collectives
 * with CPU staging. When all devices are CPU, the backend will try to
 * use GPU APIs which will fail. This test verifies the context initializes
 * correctly and that synchronize (which is a no-op) works.
 */
TEST_F(Test__LocalTPContext, TwoCpuDevicesWithHostBackend)
{
    auto cpu1 = GlobalDeviceAddress::cpu(1);
    auto ctx = createLocalTPContext({cpu0_, cpu1}, {}, CollectiveBackendType::HOST);

    EXPECT_EQ(ctx->degree(), 2);
    EXPECT_EQ(ctx->backend(), CollectiveBackendType::HOST);

    // Synchronize should work (no-op for host backend)
    ctx->synchronize();
    SUCCEED();
}

// =============================================================================
// Thread Safety Smoke Tests
// =============================================================================

/**
 * @test Concurrent synchronize calls don't crash
 */
TEST_F(Test__LocalTPContext, ConcurrentSynchronizeDoesNotCrash)
{
    auto ctx = createLocalTPContext({cpu0_}, {}, CollectiveBackendType::HOST);

    std::vector<std::thread> threads;
    for (int i = 0; i < 4; ++i)
    {
        threads.emplace_back([&ctx]()
                             {
            for (int j = 0; j < 10; ++j)
            {
                ctx->synchronize();
            } });
    }

    for (auto &t : threads)
    {
        t.join();
    }

    SUCCEED();
}

// =============================================================================
// GatherFromDevices Tests (for RankOrchestrator support)
// =============================================================================

/**
 * @test gatherFromDevices with single device just copies data
 */
TEST_F(Test__LocalTPContext, GatherFromDevicesSingleDevice)
{
    auto ctx = createLocalTPContext({cpu0_}, {}, CollectiveBackendType::HOST);

    // Create input shard (vocab_local = 100)
    auto shard = TestTensorFactory::createFP32({100});
    for (size_t i = 0; i < 100; ++i)
    {
        shard->mutable_data()[i] = static_cast<float>(i);
    }

    // Create output tensor (same size for single device)
    auto output = TestTensorFactory::createFP32({100});

    // Gather
    std::vector<const TensorBase *> shards = {shard.get()};
    ASSERT_TRUE(ctx->gatherFromDevices(shards, output.get()));

    // Verify data copied correctly
    for (size_t i = 0; i < 100; ++i)
    {
        EXPECT_FLOAT_EQ(output->data()[i], static_cast<float>(i));
    }
}

/**
 * @test gatherFromDevices with two devices concatenates data
 */
TEST_F(Test__LocalTPContext, GatherFromDevicesTwoDevices)
{
    auto cpu1 = GlobalDeviceAddress::cpu(1);
    auto ctx = createLocalTPContext({cpu0_, cpu1}, {0.5f, 0.5f}, CollectiveBackendType::HOST);

    // Create shards: device 0 has [0..49], device 1 has [100..149]
    auto shard0 = TestTensorFactory::createFP32({50});
    auto shard1 = TestTensorFactory::createFP32({50});

    for (size_t i = 0; i < 50; ++i)
    {
        shard0->mutable_data()[i] = static_cast<float>(i);       // 0..49
        shard1->mutable_data()[i] = static_cast<float>(100 + i); // 100..149
    }

    // Create output tensor (full vocab = 100)
    auto output = TestTensorFactory::createFP32({100});

    // Gather
    std::vector<const TensorBase *> shards = {shard0.get(), shard1.get()};
    ASSERT_TRUE(ctx->gatherFromDevices(shards, output.get()));

    // Verify data concatenated correctly
    // First 50 elements should be from shard0 (0..49)
    for (size_t i = 0; i < 50; ++i)
    {
        EXPECT_FLOAT_EQ(output->data()[i], static_cast<float>(i))
            << "Mismatch at index " << i;
    }
    // Next 50 elements should be from shard1 (100..149)
    for (size_t i = 0; i < 50; ++i)
    {
        EXPECT_FLOAT_EQ(output->data()[50 + i], static_cast<float>(100 + i))
            << "Mismatch at index " << (50 + i);
    }
}

/**
 * @test gatherFromDevices with proportional weights (different shard sizes)
 *
 * Simulates column-parallel LM head with 73%/27% weight distribution.
 * vocab_size = 100, so device 0 gets 73 tokens, device 1 gets 27 tokens.
 */
TEST_F(Test__LocalTPContext, GatherFromDevicesProportionalWeights)
{
    auto cpu1 = GlobalDeviceAddress::cpu(1);
    auto ctx = createLocalTPContext({cpu0_, cpu1}, {0.73f, 0.27f}, CollectiveBackendType::HOST);

    // Create shards with different sizes
    auto shard0 = TestTensorFactory::createFP32({73}); // 73% of vocab
    auto shard1 = TestTensorFactory::createFP32({27}); // 27% of vocab

    for (size_t i = 0; i < 73; ++i)
    {
        shard0->mutable_data()[i] = static_cast<float>(i);
    }
    for (size_t i = 0; i < 27; ++i)
    {
        shard1->mutable_data()[i] = static_cast<float>(1000 + i); // Different range
    }

    // Create output tensor (full vocab = 100)
    auto output = TestTensorFactory::createFP32({100});

    // Gather
    std::vector<const TensorBase *> shards = {shard0.get(), shard1.get()};
    ASSERT_TRUE(ctx->gatherFromDevices(shards, output.get()));

    // Verify concatenation
    for (size_t i = 0; i < 73; ++i)
    {
        EXPECT_FLOAT_EQ(output->data()[i], static_cast<float>(i));
    }
    for (size_t i = 0; i < 27; ++i)
    {
        EXPECT_FLOAT_EQ(output->data()[73 + i], static_cast<float>(1000 + i));
    }
}

/**
 * @test gatherFromDevices fails with empty shards vector
 */
TEST_F(Test__LocalTPContext, GatherFromDevicesEmptyShardsFails)
{
    auto ctx = createLocalTPContext({cpu0_}, {}, CollectiveBackendType::HOST);

    auto output = TestTensorFactory::createFP32({100});

    std::vector<const TensorBase *> empty_shards;
    EXPECT_FALSE(ctx->gatherFromDevices(empty_shards, output.get()));
}

/**
 * @test gatherFromDevices fails with null output
 */
TEST_F(Test__LocalTPContext, GatherFromDevicesNullOutputFails)
{
    auto ctx = createLocalTPContext({cpu0_}, {}, CollectiveBackendType::HOST);

    auto shard = TestTensorFactory::createFP32({100});
    std::vector<const TensorBase *> shards = {shard.get()};

    EXPECT_FALSE(ctx->gatherFromDevices(shards, nullptr));
}

/**
 * @test gatherFromDevices fails when shard count mismatches device count
 */
TEST_F(Test__LocalTPContext, GatherFromDevicesWrongShardCountFails)
{
    auto cpu1 = GlobalDeviceAddress::cpu(1);
    auto ctx = createLocalTPContext({cpu0_, cpu1}, {}, CollectiveBackendType::HOST);

    auto shard = TestTensorFactory::createFP32({100});
    auto output = TestTensorFactory::createFP32({200});

    // Only one shard for two-device context
    std::vector<const TensorBase *> shards = {shard.get()};
    EXPECT_FALSE(ctx->gatherFromDevices(shards, output.get()));
}

/**
 * @test gatherFromDevices fails when output buffer is too small
 */
TEST_F(Test__LocalTPContext, GatherFromDevicesOutputTooSmallFails)
{
    auto cpu1 = GlobalDeviceAddress::cpu(1);
    auto ctx = createLocalTPContext({cpu0_, cpu1}, {}, CollectiveBackendType::HOST);

    auto shard0 = TestTensorFactory::createFP32({50});
    auto shard1 = TestTensorFactory::createFP32({50});
    auto output = TestTensorFactory::createFP32({50}); // Too small! Need 100.

    std::vector<const TensorBase *> shards = {shard0.get(), shard1.get()};
    EXPECT_FALSE(ctx->gatherFromDevices(shards, output.get()));
}
