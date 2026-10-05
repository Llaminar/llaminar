/**
 * @file Test__TPLocalReduceOverlap.cpp
 * @brief CUDA/NCCL and ROCm/RCCL proof of paired reduction capture and replay.
 *
 * The same production builder records a native reduction alongside a real
 * RMSNorm kernel on each participant. Twenty request resets replace the input
 * partials without recapture; exact sums and unchanged independent outputs
 * prove the fork, join, arena lifetime and replay semantics together. Host
 * copies and waits occur only at fixture setup and result observation.
 * The same fixture also exercises lossless native allgather through this
 * single paired lifecycle, requiring every participant's exact output bytes.
 * Native FP32/FP16 allreduce forks also match the ordinary canonical sum byte
 * for byte, including input values that expose FP16 cast rounding. Scratch is
 * admitted once through PhysicalMemoryAuthority, never allocated by replay.
 * Column-owned sums reuse the same lifecycle and compare each returned column
 * byte against that full allreduce, across singleton, odd and padded row sizes.
 * In-place allreduce preserves every inactive FP32 byte through both precision
 * conversions. Focused sweeps cover every live count and large-to-small replay.
 * Device-prefix cases replay full/partial/empty/growing inputs on those same
 * fork/join graphs. The fixed control is an arithmetic oracle, not a second
 * production dispatch path; native receipt tests separately prove wire bytes.
 * Canonical rank-order forks additionally preserve the ordinary stage's exact
 * arithmetic and admitted scratch with two/four participants and cancellation
 * inputs. The degree-four case guards production MoE shared-projection lowering.
 */
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <barrier>
#include <cmath>
#include <cstring>
#include <exception>
#include <thread>
#include <vector>

#include "backends/BackendManager.h"
#include "backends/GPUDeviceContextPool.h"
#include "collective/LocalTPContext.h"
#include "execution/compute_stages/stages/RMSNormStage.h"
#include "execution/compute_stages/stages/TPLocalReduceOverlap.h"
#include "execution/compute_stages/stages/NativeAllGatherStage.h"
#include "execution/compute_stages/stages/TPColumnReduceScatterStage.h"
#include "execution/local_execution/device/DeviceContext.h"
#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "execution/local_execution/graph/ComputeGraph.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "tensors/Tensors.h"
#include "transfer/TransferEngine.h"
#include "planning/CollectiveMemoryEstimator.h"
#include "../../mocks/MockComputeStage.h"

using namespace llaminar2;

namespace
{
    /** @brief Raise fixture failures without abandoning a participant barrier. */
    void require(bool result, const char *message)
    {
        if (!result) throw std::runtime_error(message);
    }

    /** @brief Exercise one native backend with no model, transport override or timing gate. */
    enum class Operation { RootedReduce, AllGather, AllreduceFP32, AllreduceFP16,
        ColumnFP32, ColumnFP16, AllreduceSidebandFP32, AllreduceSidebandFP16, CanonicalRankOrder };
    /** @brief Immutable collective geometry or a replay-varying device-owned prefix. */
    enum class RowExtent { Fixed, DevicePrefix, EveryDevicePrefix };

    /** @brief Exercise the selected native operation through its unchanged public builder. */
    void verifyCapturedOverlap(DeviceId first, Operation operation = Operation::RootedReduce,
                               int rows = 1, int columns = 16384, RowExtent extent = RowExtent::Fixed,
                               int degree = 2)
    {
        auto *backend = getBackendFor(first);
        ASSERT_NE(backend, nullptr);
        if (backend->deviceCount() < degree) GTEST_SKIP() << "Requires " << degree << " native GPUs";
        std::vector<DeviceId> devices;
        std::vector<GlobalDeviceAddress> endpoints;
        for (int i = 0; i < degree; ++i)
        {
            devices.push_back(first.is_cuda() ? DeviceId::cuda(i) : DeviceId::rocm(i));
            endpoints.push_back(first.is_cuda() ? GlobalDeviceAddress::cuda(i) : GlobalDeviceAddress::rocm(i));
        }
        auto tp = createLocalTPContext(
            std::move(endpoints),
            {}, first.is_cuda() ? CollectiveBackendType::NCCL : CollectiveBackendType::RCCL);
        ASSERT_NE(tp, nullptr);
        const size_t count = std::size_t(rows) * columns;
        const bool column_scatter = operation == Operation::ColumnFP32 || operation == Operation::ColumnFP16;
        const bool grouped_sideband = operation == Operation::AllreduceSidebandFP32 || operation == Operation::AllreduceSidebandFP16;
        const bool canonical = operation == Operation::CanonicalRankOrder;
        const bool allreduce = canonical || column_scatter || grouped_sideband || operation == Operation::AllreduceFP32 || operation == Operation::AllreduceFP16;
        const std::string precision = operation == Operation::AllreduceFP16 || operation == Operation::ColumnFP16 ||
            operation == Operation::AllreduceSidebandFP16 ? "fp16" : "fp32";
        WorkspaceRequirements scratch;
        std::shared_ptr<PhysicalMemoryAuthority> authority;
        if (allreduce)
        {
            // Both the ordinary control and the fork borrow this one exact
            // setup reservation. The native sum must not allocate on either stream.
            const auto bom = CollectiveMemoryEstimator::localTP(1, count, tp->backend());
            if (canonical)
            {
                // Derive the scratch BOM from the production operation, rather
                // than inventing a fixture-only parallel allocation formula.
                FP32Tensor shape({std::size_t(rows), std::size_t(columns)}, DeviceId::cpu());
                TPAllreduceStage::Params declaration;
                declaration.tensor = &shape;
                declaration.tp_ctx = tp.get();
                declaration.count = count;
                declaration.arithmetic_policy = TPAllreduceArithmeticPolicy::CanonicalRankOrder;
                scratch = TPAllreduceStage(declaration).getWorkspaceRequirements(rows);
            }
            const auto scratch_bytes = scratch.total_bytes_with_alignment();
            PhysicalMemoryPlanBuilder plan;
            for (auto device : devices)
            {
                const PhysicalMemoryResource resource{.world_rank = 0, .device = device,
                    .total_bytes = bom.perDeviceBytes() + scratch_bytes,
                    .admission_available_bytes = bom.perDeviceBytes() + scratch_bytes};
                plan.add(resource, PhysicalMemoryOwner::LocalCollective, bom.perDeviceBytes());
                if (scratch_bytes)
                    plan.add(resource, PhysicalMemoryOwner::ExecutionWorkspace, scratch_bytes);
            }
            authority = std::make_shared<PhysicalMemoryAuthority>(
                std::make_shared<const PhysicalMemoryPlanAdmissionCertificate>(plan.build()), 0);
            ASSERT_TRUE(tp->reserveCollectiveResources(
                bom.backend_payload_capacity_bytes, bom.fp16_scratch_elements, authority));
        }
        std::barrier rendezvous(degree);
        std::vector<std::exception_ptr> errors(degree);
        std::vector<std::thread> threads(degree);
        for (int participant = 0; participant < degree; ++participant)
        {
            threads[participant] = std::thread([&, participant]
            {
                try
                {
                    const auto device = devices[participant];
                    auto &gpu = GPUDeviceContextPool::instance().getContext(device);
                    gpu.submitAndWait([&]
                    {
                        void *const stream = gpu.getOrCreateAuxiliaryStream("reduce_overlap_regression_main");
                        auto context = IDeviceContext::create(device, 1);
                        FP32Tensor partial({std::size_t(rows), std::size_t(columns)}, device);
                        FP32Tensor serial({std::size_t(rows), std::size_t(columns)}, device);
                        FP32Tensor gathered({std::size_t(degree) * count}, device);
                        FP32Tensor input({64, 256}, device);
                        FP32Tensor output({64, 256}, device);
                        FP32Tensor gamma({256}, device);
                        FP32Tensor live_count({2}, device);
                        std::fill_n(live_count.mutable_data(), 2, 0.0f);
                        std::fill_n(partial.mutable_data(), count, 1.0f);
                        std::fill_n(serial.mutable_data(), count, 1.0f);
                        std::fill_n(gathered.mutable_data(), std::size_t(degree) * count, 0.0f);
                        std::fill_n(input.mutable_data(), 64 * 256, 1.0f);
                        std::fill_n(output.mutable_data(), 64 * 256, 0.0f);
                        std::fill_n(gamma.mutable_data(), 256, 1.0f);
                        for (auto *tensor : {&partial, &serial, &gathered, &input, &output, &gamma, &live_count})
                            require(tensor->ensureOnDevice(device, stream), "fixture upload");
                        require(gpu.synchronizeStreamChecked(stream), "fixture upload completion");
                        const auto device_rows = DeviceRowRange::deviceCounted(rows,
                            static_cast<const std::int32_t *>(live_count.gpu_data_ptr()));
                        std::unique_ptr<DeviceWorkspaceManager> workspace;
                        std::unique_ptr<TPAllreduceStage> canonical_control;
                        if (canonical)
                        {
                            workspace = std::make_unique<DeviceWorkspaceManager>(
                                device, scratch.total_bytes_with_alignment(), authority);
                            require(workspace->allocate(scratch), "canonical scratch admission");
                            TPAllreduceStage::Params control;
                            control.device_id = device;
                            control.tp_ctx = tp.get();
                            control.tensor = &serial;
                            control.count = count;
                            control.precision = "fp32";
                            control.arithmetic_policy = TPAllreduceArithmeticPolicy::CanonicalRankOrder;
                            control.sideband_device_index = participant;
                            control.stage_name = "serial_control";
                            canonical_control = std::make_unique<TPAllreduceStage>(control);
                            canonical_control->bindWorkspace(workspace.get());
                            require(canonical_control->prepareGraphLaunch(context.get(), stream), "canonical control preparation");
                        }

                        ComputeGraph graph;
                        graph.addNode("producer", std::make_unique<llaminar2::testing::MockComputeStage>(
                            ComputeStageType::GEMM, "producer", device), device);
                        RMSNormStage::Params norm;
                        norm.device_id = device;
                        norm.input = &input;
                        norm.output = &output;
                        norm.gamma = &gamma;
                        norm.input_buffer_id = BufferId::HIDDEN_STATE;
                        norm.output_buffer_id = BufferId::NORMALIZED;
                        graph.addNode("compute", std::make_unique<RMSNormStage>(norm), device);
                        TPLocalRootedCollectiveStage::Params reduce;
                        reduce.device_id = device;
                        reduce.tp_ctx = tp.get();
                        reduce.tensor = &partial;
                        reduce.count = count;
                        reduce.root_device_index = 0;
                        reduce.participant_device_index = participant;
                        reduce.stage_name = "reduce";
                        reduce.tensor_buffer_id = BufferId::MOE_SHARED_EXPERT_OUTPUT;
                        if (operation == Operation::RootedReduce)
                            addTPLocalReduceOverlap(graph, reduce, {"producer", "compute"});
                        else if (column_scatter)
                        {
                            TPColumnReduceScatterStage::Params sum;
                            sum.device_id = device;
                            sum.tp_ctx = tp.get();
                            sum.tensor = &partial;
                            sum.packing = &gathered;
                            sum.rows = rows;
                            sum.model_columns = columns;
                            sum.participant = participant;
                            sum.precision = precision;
                            sum.stage_name = "reduce";
                            sum.tensor_buffer_id = BufferId::MOE_SHARED_EXPERT_OUTPUT;
                            sum.packing_buffer_id = BufferId::MOE_PROJECTION_GATHERED_COLUMNS;
                            if (extent != RowExtent::Fixed) sum.live_rows = device_rows;
                            graph.addNode("reduce", std::make_unique<TPColumnReduceScatterStage>(sum), device);
                            graph.addDependency("reduce", "producer");
                            overlapTPLocalColumnReduceScatter(graph, "reduce", "compute");
                        }
                        else if (allreduce)
                        {
                            TPAllreduceStage::Params sum;
                            sum.device_id = device;
                            sum.tp_ctx = tp.get();
                            sum.tensor = &partial;
                            sum.count = count;
                            sum.precision = precision;
                            sum.stage_name = "reduce";
                            sum.tensor_buffer_id = BufferId::MOE_SHARED_EXPERT_OUTPUT;
                            sum.sideband_device_index = participant;
                            if (canonical) sum.arithmetic_policy = TPAllreduceArithmeticPolicy::CanonicalRankOrder;
                            if (extent != RowExtent::Fixed) sum.live_rows.emplace(device_rows, columns);
                            if (grouped_sideband)
                            {
                                LocalTPCollectiveSidebandBuffer sideband;
                                sideband.kind = LocalTPCollectiveSidebandKind::AllreduceSum;
                                sideband.recv_buffer = static_cast<std::int32_t *>(live_count.gpu_data_ptr()) + 1;
                                sideband.element_count = 1;
                                sideband.dtype = CollectiveDataType::INT32;
                                sideband.name = "live_prefix_control";
                                sum.sidebands.push_back(sideband);
                            }
                            graph.addNode("reduce", std::make_unique<TPAllreduceStage>(sum), device);
                            graph.addDependency("reduce", "producer");
                            // Production control sidebands deliberately stay in
                            // their grouped anchor, not an unsupported fork.
                            if (!grouped_sideband) overlapTPLocalAllreduce(graph, "reduce", "compute");
                        }
                        else
                        {
                            NativeAllGatherStage::Params gather;
                            gather.device_id = device;
                            gather.tp_ctx = tp.get();
                            gather.local_input = &partial;
                            gather.rank_major_output = &gathered;
                            gather.bytes_per_participant = count * sizeof(float);
                            gather.participant = participant;
                            gather.stage_name = "reduce"; // One unchanged completion identity for both operations.
                            gather.input_buffer_id = BufferId::MOE_SHARED_EXPERT_OUTPUT;
                            gather.output_buffer_id = BufferId::MOE_COMBINED_OUTPUT;
                            if (extent != RowExtent::Fixed)
                                gather.live_rows.emplace(device_rows, columns * sizeof(float));
                            graph.addNode("reduce", std::make_unique<NativeAllGatherStage>(gather), device);
                            graph.addDependency("reduce", "producer");
                            overlapTPLocalAllGather(graph, "reduce", "compute");
                        }

                        std::vector<GraphCaptureDependencyLedger::StagePlan> plans;
                        for (const auto &name : graph.getExecutionOrder())
                        {
                            auto *stage = graph.getNode(name)->stage.get();
                            if (canonical)
                                if (auto *consumer = dynamic_cast<IWorkspaceConsumer *>(stage))
                                    consumer->bindWorkspace(workspace.get());
                            require(stage->prepareGraphLaunch(context.get(), stream), "stage preparation");
                            GraphCaptureDependencyLedger::StagePlan plan;
                            plan.stage_identity = stage;
                            plan.stage_name = name;
                            if (name == "compute")
                            {
                                plan.external_inputs = {&input, &gamma};
                                plan.outputs = {&output};
                                // Resolve the existing norm facade before recording.
                                require(stage->execute(context.get()), "norm preparation");
                            }
                            else if (name == "reduce_submit" || name == "reduce")
                            {
                                plan.external_inputs = {&partial};
                                if (name == "reduce")
                                {
                                    if (column_scatter) plan.outputs = {&partial, &gathered};
                                    else if (operation == Operation::AllGather) plan.outputs = {&gathered};
                                    else if (allreduce || participant == 0) plan.outputs = {&partial};
                                }
                            }
                            plans.push_back(std::move(plan));
                        }
                        require(gpu.synchronizeStreamChecked(stream), "preparation completion");
                        // Mirror the production executor's frontier admission:
                        // an upload completion must be joined before capture,
                        // even when a fixture has already observed it on host.
                        for (auto *tensor : {&partial, &input, &gamma})
                            TransferEngine::requireDeviceInput(tensor, device, stream);
                        GraphCaptureDependencyLedger ledger(device, stream, std::move(plans), "reduce_overlap");
                        auto capture = gpu.createGraphCapture(stream);
                        rendezvous.arrive_and_wait();
                        ScopedBackendGraphCapture recording(gpu, *capture, "reduce_overlap", &ledger);
                        require(recording.begin(), "begin capture");
                        for (const auto &name : graph.getExecutionOrder())
                        {
                            auto *stage = graph.getNode(name)->stage.get();
                            ScopedGraphCaptureStage scope(stage);
                            require(stage->execute(context.get()), "captured stage");
                            scope.complete();
                        }
                        recording.finish();
                        require(capture->instantiate(), "instantiate");

                        const auto result_count = column_scatter ? count / degree : operation == Operation::AllGather ? degree * count : count;
                        auto *result = operation == Operation::AllGather ? &gathered : &partial;
                        std::vector<float> host(count), actual(result_count), expected(count), norm_actual(64 * 256);
                        const int replay_count = extent == RowExtent::EveryDevicePrefix ? rows + 8 : 20;
                        for (int request = 1; request <= replay_count; ++request)
                        {
                            const std::array<std::int32_t, 5> prefixes{rows, rows - 1, 0, 1, rows};
                            const std::array<std::int32_t, 7> replay_tail{rows, 0, rows / 2, 0, rows, 1, rows - 1};
                            const std::int32_t live = extent == RowExtent::EveryDevicePrefix
                                ? (request <= rows + 1 ? request - 1 : replay_tail[request - rows - 2])
                                : (extent == RowExtent::DevicePrefix
                                    ? prefixes[(request - 1) % prefixes.size()] : rows);
                            const auto active_count = std::size_t(live) * columns;
                            for (size_t i = 0; i < count; ++i)
                            {
                                host[i] = static_cast<float>((participant + 1) * request * (1 + i % 17)) +
                                    (allreduce ? static_cast<float>(1 + i % 7) / 37.0f : 0.0f);
                                if (canonical && i % 2 == 0)
                                {
                                    // A reassociated sum can erase these small
                                    // rank-two/rank-three addends. Ordinary and
                                    // forked stages must use one arithmetic order.
                                    const std::array<float, 4> cancellation{1.0e20f, -1.0e20f, 3.125f, 0.25f};
                                    host[i] = cancellation[participant] * request;
                                }
                            }
                            require(backend->hostToDeviceOnStream(partial.gpu_data_ptr(), host.data(),
                                count * sizeof(float), participant, stream), "request input reset");
                            const std::array<std::int32_t, 2> metadata{live, participant + 1};
                            std::array<std::int32_t, 2> observed_metadata{};
                            require(backend->hostToDeviceOnStream(live_count.gpu_data_ptr(), metadata.data(),
                                sizeof(metadata), participant, stream), "publish replay row authority");
                            if (operation == Operation::AllGather)
                            {
                                std::fill(actual.begin(), actual.end(), -12345.0f);
                                require(backend->hostToDeviceOnStream(gathered.gpu_data_ptr(), actual.data(),
                                    actual.size() * sizeof(float), participant, stream), "poison gather tails");
                            }
                            rendezvous.arrive_and_wait();
                            require(capture->launch(), "replay");
                            require(backend->deviceToHostOnStream(actual.data(), result->gpu_data_ptr(),
                                result_count * sizeof(float), participant, stream), "collective observation");
                            require(backend->deviceToHostOnStream(norm_actual.data(), output.gpu_data_ptr(),
                                norm_actual.size() * sizeof(float), participant, stream), "compute observation");
                            if (grouped_sideband)
                                require(backend->deviceToHostOnStream(observed_metadata.data(), live_count.gpu_data_ptr(),
                                    sizeof(observed_metadata), participant, stream), "sideband observation");
                            if (allreduce)
                            {
                                require(backend->hostToDeviceOnStream(serial.gpu_data_ptr(), host.data(),
                                    count * sizeof(float), participant, stream), "serial control reset");
                                require(canonical ? canonical_control->execute(context.get())
                                    : tp->allreduceOnStream(&serial, "serial_control", count, stream, precision),
                                    "canonical serial control");
                                require(backend->deviceToHostOnStream(expected.data(), serial.gpu_data_ptr(),
                                    count * sizeof(float), participant, stream), "serial control observation");
                            }
                            require(gpu.synchronizeStreamChecked(stream), "terminal completion");
                            if (grouped_sideband)
                                require(observed_metadata[0] == live && observed_metadata[1] == degree * (degree + 1) / 2,
                                    "anchor clipped sideband or changed count authority");
                            if (column_scatter)
                            {
                                const auto local_columns = std::size_t(columns / degree);
                                for (int row = 0; row < live; ++row)
                                    require(std::memcmp(actual.data() + row * local_columns,
                                        expected.data() + std::size_t(row) * columns + participant * local_columns,
                                        local_columns * sizeof(float)) == 0,
                                        "column scatter differs bytewise from canonical allreduce slice");
                            }
                            else if (allreduce)
                                require(std::memcmp(actual.data(), expected.data(), active_count * sizeof(float)) == 0,
                                    "forked allreduce differs bytewise from canonical precision path");
                            for (size_t i = 0; !column_scatter && i < result_count; ++i)
                            {
                                if (i % count >= active_count)
                                {
                                    if (operation == Operation::AllGather)
                                        require(actual[i] == -12345.0f, "native stage wrote an inactive row");
                                    // Allreduce is in-place: inactive rows are not operands of this replay.
                                    if (allreduce && !column_scatter)
                                        require(std::memcmp(&actual[i], &host[i], sizeof(float)) == 0,
                                            "in-place allreduce wrote an inactive FP32 row");
                                    continue;
                                }
                                require(actual[i] == (allreduce ? expected[i] : static_cast<float>(
                                    (operation == Operation::AllGather ? 1 + i / count :
                                        (participant == 0 ? degree * (degree + 1) / 2 : participant + 1)) *
                                    request * (1 + (i % count) % 17))),
                                    "captured reduction changed exact sum or contributor input");
                            }
                            for (float value : norm_actual)
                                require(std::isfinite(value) && std::abs(value - 1.0f) < 1e-5f,
                                    "independent captured compute was corrupted");
                        }
                        rendezvous.arrive_and_wait();
                        capture->reset();
                    });
                }
                catch (...)
                {
                    errors[participant] = std::current_exception();
                    tp->requestAbort();
                    rendezvous.arrive_and_drop();
                }
            });
        }
        for (auto &thread : threads) thread.join();
        for (auto &error : errors)
        {
            if (!error) continue;
            try { std::rethrow_exception(error); }
            catch (const std::exception &failure) { ADD_FAILURE() << failure.what(); }
        }
    }

    /** @brief Exercise row packing and unchanged FP32/FP16 precision across capture geometries. */
    void verifyColumns(DeviceId device)
    {
        for (int rows : {1, 7, 64, 512})
            for (auto operation : {Operation::ColumnFP32, Operation::ColumnFP16})
            {
                SCOPED_TRACE(::testing::Message() << "rows=" << rows << " operation=" << int(operation));
                // Both requested policies retain FP32 below the original row
                // threshold. The wide case additionally proves actual FP16 casts.
                verifyCapturedOverlap(device, operation, rows, 2048);
                if (rows <= 64) verifyCapturedOverlap(device, operation, rows, 16384);
            }
    }

    /** @brief Prove all captured native fork families preserve precision with replay-varying prefixes. */
    void verifyLivePrefixes(DeviceId device)
    {
        for (auto operation : {Operation::AllGather, Operation::AllreduceFP32,
                Operation::AllreduceFP16, Operation::ColumnFP32, Operation::ColumnFP16,
                Operation::AllreduceSidebandFP32, Operation::AllreduceSidebandFP16})
            verifyCapturedOverlap(device, operation, 17, 16384, RowExtent::DevicePrefix);
    }

    /**
     * @brief Prove live-only precision conversion through ordinary and sideband graph owners.
     * @param device First native GPU in the admitted two-participant clique.
     *
     * Every count from zero through seventeen is followed by full/empty/partial
     * transitions. FP16 runs use wide rows to cross the actual transport threshold;
     * live values retain the fixed conversion oracle and every inactive FP32
     * value retains its original bytes without graph recapture.
     */
    void verifyInactiveFP32Prefixes(DeviceId device)
    {
        for (const auto operation : {Operation::AllreduceFP32, Operation::AllreduceFP16,
                Operation::AllreduceSidebandFP32, Operation::AllreduceSidebandFP16})
            verifyCapturedOverlap(device, operation, 17, 16384, RowExtent::EveryDevicePrefix);
    }

    /**
     * @brief Prove that canonical folds preserve every inactive in-place FP32 byte.
     * @param device First native endpoint of the admitted homogeneous clique.
     * @param degree Exact physical participant count, including four-rank cancellation.
     *
     * Every prefix from zero through seventeen precedes full/empty/partial
     * replays on one retained graph. The fixed native-bank fold remains the
     * exact ascending-rank arithmetic oracle for every live output value.
     */
    void verifyInactiveCanonicalPrefixes(DeviceId device, int degree)
    {
        verifyCapturedOverlap(device, Operation::CanonicalRankOrder, 17, 256,
            RowExtent::EveryDevicePrefix, degree);
    }

    /** @brief Prove canonical fork arithmetic for M1/grouped rows and retained live prefixes.
     * @param device Native backend endpoint, with no transport substitution.
     * @param degree Actual physical participants admitted by the fixture. */
    void verifyCanonicalRanks(DeviceId device, int degree)
    {
        for (const int rows : {1, 7, 16})
            verifyCapturedOverlap(device, Operation::CanonicalRankOrder, rows, 256, RowExtent::Fixed, degree);
        verifyCapturedOverlap(device, Operation::CanonicalRankOrder, 17, 256, RowExtent::DevicePrefix, degree);
    }
}

#ifdef HAVE_CUDA
TEST(Test__TPLocalReduceOverlap, CUDA) { verifyCapturedOverlap(DeviceId::cuda(0)); }
TEST(Test__TPLocalAllGatherOverlap, CUDA) { verifyCapturedOverlap(DeviceId::cuda(0), Operation::AllGather); }
TEST(Test__TPLocalAllreduceOverlap, CUDA_FP32) { verifyCapturedOverlap(DeviceId::cuda(0), Operation::AllreduceFP32); }
TEST(Test__TPLocalAllreduceOverlap, CUDA_FP16) { verifyCapturedOverlap(DeviceId::cuda(0), Operation::AllreduceFP16); }
TEST(Test__TPLocalRankOrderOverlap, CUDA2) { verifyCanonicalRanks(DeviceId::cuda(0), 2); }
/** @test Live canonical arithmetic retains inactive bytes on 2 physical participants. */
TEST(Test__TPCanonicalInactiveRows, CUDA2) { verifyInactiveCanonicalPrefixes(DeviceId::cuda(0), 2); }
TEST(Test__TPColumnReduceScatter, CUDA) { verifyColumns(DeviceId::cuda(0)); }
TEST(Test__TPLiveRowsOverlap, CUDA) { verifyLivePrefixes(DeviceId::cuda(0)); }
/** @test Both precision modes preserve all inactive FP32 bytes across every live prefix. */
TEST(Test__TPLocalAllreduceInactiveRows, CUDA) { verifyInactiveFP32Prefixes(DeviceId::cuda(0)); }
#endif
#ifdef HAVE_ROCM
TEST(Test__TPLocalReduceOverlap, ROCm) { verifyCapturedOverlap(DeviceId::rocm(0)); }
TEST(Test__TPLocalAllGatherOverlap, ROCm) { verifyCapturedOverlap(DeviceId::rocm(0), Operation::AllGather); }
TEST(Test__TPLocalAllreduceOverlap, ROCm_FP32) { verifyCapturedOverlap(DeviceId::rocm(0), Operation::AllreduceFP32); }
TEST(Test__TPLocalAllreduceOverlap, ROCm_FP16) { verifyCapturedOverlap(DeviceId::rocm(0), Operation::AllreduceFP16); }
TEST(Test__TPLocalRankOrderOverlap, ROCm2) { verifyCanonicalRanks(DeviceId::rocm(0), 2); }
TEST(Test__TPLocalRankOrderOverlap, ROCm4) { verifyCanonicalRanks(DeviceId::rocm(0), 4); }
/** @test Live canonical arithmetic retains inactive bytes on 2 physical participants. */
TEST(Test__TPCanonicalInactiveRows, ROCm2) { verifyInactiveCanonicalPrefixes(DeviceId::rocm(0), 2); }
/** @test Live canonical arithmetic retains inactive bytes on 4 physical participants. */
TEST(Test__TPCanonicalInactiveRows, ROCm4) { verifyInactiveCanonicalPrefixes(DeviceId::rocm(0), 4); }
TEST(Test__TPColumnReduceScatter, ROCm) { verifyColumns(DeviceId::rocm(0)); }
TEST(Test__TPLiveRowsOverlap, ROCm) { verifyLivePrefixes(DeviceId::rocm(0)); }
/** @test Both precision modes preserve all inactive FP32 bytes across every live prefix. */
TEST(Test__TPLocalAllreduceInactiveRows, ROCm) { verifyInactiveFP32Prefixes(DeviceId::rocm(0)); }
#endif
