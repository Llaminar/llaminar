/**
 * @file Test__W4A4PlanningSamplesMPI.cpp
 * @brief Native online-A4 publication, exact wire extent and final-reader lifetime.
 *
 * Only root opens the tiny source. It removes that source before broadcasting
 * immutable native bytes; followers must seal their own final admitted roots.
 * A scoped PMPI observer records the actual payload call without changing it.
 * Failure consensus must precede traffic, and no source allocation may escape.
 * A receiver-only fault after real MPI completion proves malformed native bytes
 * fail sealing collectively; the observer never mutates a published root tensor.
 */
#include "planning/PlanningMatrixSample.h"
#include "planning/PhysicalMemoryAuthority.h"
#include "kernels/KernelFactory.h"
#include "tensors/W4A4Tensor.h"
#include "utils/MPIContext.h"
#include "../../utils/PlanningGGUFFixture.h"
#include "../../utils/W4A4SourceFixture.h"
#include <gtest/gtest.h>
#include <cstring>
#include <filesystem>

namespace
{
    /** @brief Test-only observation of native payload extents, never a transport replacement. */
    struct BroadcastObservation
    {
        size_t expected_bytes; ///< Exact live source bytes, excluding allocator padding.
        size_t calls = 0; ///< Native payload submissions in this publication.
        bool exact = true; ///< Every call retained byte datatype, root and exact count.
        bool corrupt_receive = false; ///< Inject only at rank one after its real receive completes.
        bool corruption_applied = false; ///< Certifies the negative control actually changed bytes.
        void *receive_buffer = nullptr; ///< Unpublished follower storage; never the root source.
        MPI_Request *receive_request = nullptr; ///< Exact in-flight payload batch, not consensus work.
    };
    thread_local BroadcastObservation *observed_broadcast = nullptr;

    /** @brief Limit interception to the current synchronous publication's native calls. */
    struct ObserveBroadcast
    {
        BroadcastObservation *previous = observed_broadcast;
        explicit ObserveBroadcast(BroadcastObservation &observation) { observed_broadcast = &observation; }
        ~ObserveBroadcast() { observed_broadcast = previous; }
    };
}

/**
 * @brief Observe the actual MPI payload call and always delegate to native PMPI.
 * @param buffer Unchanged native source/destination.
 * @param count Unchanged live extent.
 * @param datatype Unchanged MPI datatype.
 * @param root Unchanged source rank.
 * @param comm Unchanged communicator.
 * @param request Unchanged completion authority.
 * @return The real native transport result.
 */
extern "C" int MPI_Ibcast(void *buffer, int count, MPI_Datatype datatype, int root,
    MPI_Comm comm, MPI_Request *request)
{
    if (observed_broadcast)
    {
        ++observed_broadcast->calls;
        observed_broadcast->exact = observed_broadcast->exact && count > 0 &&
            static_cast<size_t>(count) == observed_broadcast->expected_bytes && datatype == MPI_BYTE && root == 0;
        if (observed_broadcast->corrupt_receive)
        {
            int rank = -1;
            if (PMPI_Comm_rank(comm, &rank) == MPI_SUCCESS && rank == 1 &&
                datatype == MPI_BYTE && count >= int(sizeof(float)))
            {
                observed_broadcast->receive_buffer = buffer;
                observed_broadcast->receive_request = request;
            }
            else observed_broadcast->exact = false;
        }
    }
    return PMPI_Ibcast(buffer, count, datatype, root, comm, request);
}

/**
 * @brief Inject an invalid unit into only an unpublished completed follower payload.
 * @param count Original request count.
 * @param requests Original requests; pointer identity distinguishes payload from consensus.
 * @param completed Native completion flag; never synthesized by the observer.
 * @param statuses Original native completion statuses.
 * @return Unchanged PMPI completion result.
 */
extern "C" int MPI_Testall(int count, MPI_Request *requests, int *completed, MPI_Status *statuses)
{
    const int result = PMPI_Testall(count, requests, completed, statuses);
    if (result == MPI_SUCCESS && completed && *completed && count == 1 &&
        observed_broadcast && observed_broadcast->receive_request == requests)
    {
        const float invalid_unit = -1.0f;
        std::memcpy(observed_broadcast->receive_buffer, &invalid_unit, sizeof(invalid_unit));
        observed_broadcast->corruption_applied = true;
        observed_broadcast->receive_request = nullptr;
    }
    return result;
}

namespace llaminar2
{
    using llaminar::v2::kernels::KernelFactory;

    TEST(W4A4PlanningPublicationMPI, ExactNativePayloadSurvivesReaderAndSampleRetirement)
    {
        const auto mpi = MPIContextFactory::global();
        ASSERT_GE(mpi->world_size(), 2);
        constexpr const char *name = "blk.0.ffn_down.weight";
        constexpr size_t n = 256, k = 512;
        constexpr auto owner = PhysicalMemoryOwner::ModelSourcePayload;
        for (auto type : {TensorType::W4A4, TensorType::W4A4H})
            for (const auto &request : std::vector<PlanningModelSampleRequest>{
                    {name, PlanningWholeMatrix{}}, {name, PlanningMatrixRows{3, 6}}})
            {
                const auto blocks = test::w4a4SourceBlocks(type, n, k);
                std::unique_ptr<test::PlanningGGUFFixture> fixture;
                std::unique_ptr<PlanningModelSource> source;
                const auto plan = PlanningMatrixSamplePublication::describe(mpi, [&] {
                    fixture = std::make_unique<test::PlanningGGUFFixture>(false, false,
                        test::PlanningGGUFFixture::sourceType(type));
                    fixture->writePayload(name, 0,
                        {reinterpret_cast<const uint8_t *>(blocks.data()), blocks.size() * sizeof(w4a4::SourceBlock)});
                    source = std::make_unique<PlanningModelSource>(fixture->path());
                    return PlanningMatrixSamplePlan::resolve(*source, request);
                });
                EXPECT_EQ(plan.originalColumns(), k);
                const auto bytes = plan.geometry().physicalBytes();
                PhysicalMemoryPlanBuilder builder;
                builder.add({mpi->rank(), DeviceId::cpu(), bytes, bytes}, owner, bytes);
                auto memory = std::make_shared<PhysicalMemoryAuthority>(
                    std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(builder.build()), mpi->rank());
                const auto claimed = [&] {
                    return memory->claimedBytes(DeviceId::cpu(), owner, PhysicalMemoryMaterializationKind::NewAllocation);
                };
                const auto load = [&] {
                    if (!mpi->is_root() || !source) throw std::logic_error("Only root owns a native source reader");
                    auto result = plan.load(*source, memory, DeviceId::cpu());
                    source.reset();
                    std::filesystem::remove(fixture->path());
                    fixture.reset();
                    return result;
                };
                {
                    BroadcastObservation failed{plan.geometry().source_bytes};
                    ObserveBroadcast observe(failed);
                    EXPECT_THROW(PlanningMatrixSamplePublication::publish(mpi, plan,
                        mpi->rank() == 1 ? nullptr : memory, DeviceId::cpu(), load), std::runtime_error);
                    EXPECT_EQ(failed.calls, 0u);
                    EXPECT_EQ(claimed(), 0u);
                }
                BroadcastObservation completed{plan.geometry().source_bytes};
                ObserveBroadcast observe(completed);
                std::optional sample(PlanningMatrixSamplePublication::publish(mpi, plan, memory, DeviceId::cpu(), load));
                EXPECT_EQ(completed.calls, 1u);
                EXPECT_TRUE(completed.exact);
                EXPECT_FALSE(std::filesystem::exists(plan.modelPath()));
                EXPECT_EQ(sample->tensor().native_type(), type);
                const auto first = std::holds_alternative<PlanningMatrixRows>(request.selection) ? 3u : 0u;
                EXPECT_EQ(std::memcmp(sample->tensor().raw_data(), blocks.data() + first * k / w4a4::kBlockElements,
                    plan.geometry().source_bytes), 0);
                const auto *native = dynamic_cast<const W4A4Tensor *>(&sample->tensor());
                ASSERT_NE(native, nullptr);
                auto engine = KernelFactory::createGemm(native, DeviceType::CPU, -1);
                sample.reset();
                EXPECT_EQ(claimed(), bytes);
                EXPECT_EQ(engine->packedWeightBytes(), plan.geometry().source_bytes);
                EXPECT_EQ(memory->claimedBytes(DeviceId::cpu(), PhysicalMemoryOwner::WeightLoadStaging,
                    PhysicalMemoryMaterializationKind::NewAllocation), 0u);
                engine.reset();
                EXPECT_EQ(claimed(), 0u);
            }
    }

    TEST(W4A4PlanningPublicationMPI, CorruptCompletedReceiveFailsSealingAndRetiresEveryOwner)
    {
        const auto mpi = MPIContextFactory::global();
        ASSERT_GE(mpi->world_size(), 2);
        constexpr const char *name = "blk.0.ffn_down.weight";
        constexpr size_t n = 256, k = 512;
        constexpr auto owner = PhysicalMemoryOwner::ModelSourcePayload;
        for (auto type : {TensorType::W4A4, TensorType::W4A4H})
            for (bool corrupt : {true, false, true, false})
            {
                const auto blocks = test::w4a4SourceBlocks(type, n, k);
                std::unique_ptr<test::PlanningGGUFFixture> fixture;
                std::unique_ptr<PlanningModelSource> source;
                const auto plan = PlanningMatrixSamplePublication::describe(mpi, [&] {
                    fixture = std::make_unique<test::PlanningGGUFFixture>(false, false,
                        test::PlanningGGUFFixture::sourceType(type));
                    fixture->writePayload(name, 0,
                        {reinterpret_cast<const uint8_t *>(blocks.data()), blocks.size() * sizeof(w4a4::SourceBlock)});
                    source = std::make_unique<PlanningModelSource>(fixture->path());
                    return PlanningMatrixSamplePlan::resolve(*source, {name, PlanningMatrixRows{3, 6}});
                });
                const auto bytes = plan.geometry().physicalBytes();
                PhysicalMemoryPlanBuilder builder;
                builder.add({mpi->rank(), DeviceId::cpu(), bytes, bytes}, owner, bytes);
                auto memory = std::make_shared<PhysicalMemoryAuthority>(
                    std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(builder.build()), mpi->rank());
                const auto load = [&] {
                    if (!mpi->is_root() || !source) throw std::logic_error("Only root owns a native source reader");
                    auto result = plan.load(*source, memory, DeviceId::cpu());
                    source.reset();
                    std::filesystem::remove(fixture->path());
                    fixture.reset();
                    return result;
                };
                BroadcastObservation observation{.expected_bytes = plan.geometry().source_bytes,
                    .corrupt_receive = corrupt && mpi->rank() == 1};
                ObserveBroadcast observe(observation);
                if (corrupt)
                {
                    EXPECT_THROW(PlanningMatrixSamplePublication::publish(mpi, plan, memory,
                        DeviceId::cpu(), load), std::runtime_error);
                }
                else
                {
                    const auto sample = PlanningMatrixSamplePublication::publish(mpi, plan, memory,
                        DeviceId::cpu(), load);
                    EXPECT_EQ(std::memcmp(sample.tensor().raw_data(), blocks.data() + 3 * k / w4a4::kBlockElements,
                        plan.geometry().source_bytes), 0);
                }
                EXPECT_EQ(observation.calls, 1u);
                EXPECT_TRUE(observation.exact);
                EXPECT_EQ(observation.corruption_applied, corrupt && mpi->rank() == 1);
                EXPECT_FALSE(std::filesystem::exists(plan.modelPath()));
                EXPECT_EQ(memory->claimedBytes(DeviceId::cpu(), owner,
                    PhysicalMemoryMaterializationKind::NewAllocation), 0u);
                EXPECT_EQ(memory->claimedBytes(DeviceId::cpu(), PhysicalMemoryOwner::WeightLoadStaging,
                    PhysicalMemoryMaterializationKind::NewAllocation), 0u);
            }
    }
}
