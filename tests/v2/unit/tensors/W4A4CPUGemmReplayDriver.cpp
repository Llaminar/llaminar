/**
 * @file W4A4CPUGemmReplayDriver.cpp
 * @brief Replay immutable native observations through the production tensor GEMM engine.
 *
 * Real retained weight leases, FP32 tensor boundaries and PMA-owned workspaces
 * participate. One admitted instance executes full, partial, empty and reset
 * work. Fused execution also checks a complete-output-row weight slice. Every
 * phase additionally records singleton verifier execution, including with only
 * ordinary projection workspace. The driver returns observations only; Python compares them with pinned upstream
 * bytes and the explicitly initialized inactive-storage sentinels.
 */
#include "W4A4ReplayTransport.h"
#include "kernels/cpu/gemm/CPUW4A4GemmKernel.h"
#include "tensors/W4A4Tensor.h"
#include "kernels/cpu/gemm/CPUPreparedWeightWriter.h"
#include "backends/BackendManager.h"
#include "../../utils/TestTensorFactory.h"
#include <bit>
#include <cstring>
#include <iostream>
#include <omp.h>

/**
 * @brief Execute one recorded matrix through exact single/fused tensor entrypoints.
 * @param argc Must name input and output transports only.
 * @param argv Owned temporary transport paths.
 * @return Zero only after all retained executions and observed-byte publications succeed.
 */
int main(int argc, char **argv)
{
    try
    {
        using namespace llaminar2;
        using namespace llaminar2::w4a4;
        using namespace test_transport;
        if (argc != 3) throw std::runtime_error("Expected replay input and output paths");
        const auto record = read(argv[1]);
        if (record.entry() != Entry::Tensor && record.entry() != Entry::FusedTensor)
            throw std::runtime_error("Tensor replay requires an explicit single/fused engine entry");
        if (static_cast<std::size_t>(record.m()) > kMaximumBytes / sizeof(float) / record.n())
            throw std::runtime_error("Replay output geometry exceeds the fixture bound");
        const auto source = SourceView::admit(record.transform(), record.n(), record.k(), record.weights);
        if (!source) throw std::runtime_error(source.error().message());
        if (!hasCPUBackend()) initCPUBackend(-1);
        omp_set_dynamic(0);

        // These fixed fixture limits derive from the bounded transport, not
        // hardware capacity or a second production admission calculation.
        constexpr std::size_t weight_limit = 2 * kMaximumBytes;
        constexpr std::size_t workspace_limit = 6 * kMaximumBytes;
        const PhysicalMemoryResource resource{.world_rank = 0, .device = DeviceId::cpu(),
            .total_bytes = weight_limit + workspace_limit,
            .admission_available_bytes = weight_limit + workspace_limit};
        PhysicalMemoryPlanBuilder builder;
        builder.add(resource, PhysicalMemoryOwner::PrimaryModelWeights, weight_limit);
        builder.add(resource, PhysicalMemoryOwner::ExecutionWorkspace, workspace_limit);
        const auto memory = std::make_shared<PhysicalMemoryAuthority>(
            std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(builder.build()), 0);
        const int first = record.n() > 1 ? 1 : 0, sliced_n = record.n() - first;
        const auto make_engines = [&] {
            constexpr auto owner = PhysicalMemoryOwner::PrimaryModelWeights;
            const auto type = record.transform() == Transform::Identity ? TensorType::W4A4 : TensorType::W4A4H;
            const CPUPreparedWeightSource spec{
                .original = {{static_cast<std::size_t>(record.k()), static_cast<std::size_t>(record.n())},
                    type, kBlockElements, kSourceBlockBytes},
                .range = {.rows = static_cast<std::size_t>(record.n()),
                    .columns = static_cast<std::size_t>(record.k()),
                    .native_row_bytes = source->blocks().size_bytes() / record.n()}};
            auto writer = CPUPreparedWeightWriter::begin(spec, CPUWeightStoragePlacement::accounted(
                memory, DeviceId::cpu(), owner, CPUWeightStoragePlacement::local()));
            if (writer->stagingRequest()) throw std::runtime_error("Native W4A4 writer requested a second source representation");
            // Bounded producer windows may split K; only the completed final
            // matrix is published as a whole-token arithmetic operand.
            for (std::size_t block = 0; block < source->blocks().size(); ++block)
            {
                const auto per_row = static_cast<std::size_t>(record.k() / kBlockElements);
                const NativeWeightSourceWindow window{
                    .first_row = block / per_row, .rows = 1,
                    .first_column = (block % per_row) * kBlockElements,
                    .columns = kBlockElements, .native_row_bytes = kSourceBlockBytes};
                writer->readDirect(window, [&](std::span<std::byte> destination) {
                    std::memcpy(destination.data(), &source->blocks()[block], destination.size());
                });
            }
            auto prepared = writer->finish();
            auto tensor = std::move(prepared.final_tensor);
            auto subset = tensor->create_view(
                {static_cast<std::size_t>(sliced_n), static_cast<std::size_t>(record.k())},
                static_cast<std::size_t>(first) * record.k());
            auto engines = std::pair{std::move(prepared.kernel), subset->createGemm()};
            tensor->release_raw_data();
            subset->release_raw_data();
            return engines;
        };
        // The constructor's source owner is now retired. Engines alone retain
        // the exact allocation, including the second engine's interior rows.
        auto [engine, sliced] = make_engines();
        auto input = test::TestTensorFactory::createFP32(
            {static_cast<std::size_t>(record.m()), static_cast<std::size_t>(record.k())});
        auto output = test::TestTensorFactory::createFP32(
            {static_cast<std::size_t>(record.m()), static_cast<std::size_t>(record.n())});
        auto sliced_output = test::TestTensorFactory::createFP32(
            {static_cast<std::size_t>(record.m()), static_cast<std::size_t>(sliced_n)});
        auto singleton_output = test::TestTensorFactory::createFP32(
            {static_cast<std::size_t>(record.m()), static_cast<std::size_t>(record.n())});
        std::memcpy(input->mutable_typed_data(), record.input.data(), record.input.size() * sizeof(float));
        const auto *consumer = dynamic_cast<const IWorkspaceConsumer *>(engine.get());
        if (!consumer) throw std::runtime_error("Native GEMM factory omitted its invocation workspace contract");
        auto requirements = consumer->getWorkspaceRequirements(record.m());
        if (record.entry() == Entry::FusedTensor)
        {
            const std::array<int, 2> columns{record.n(), sliced_n};
            consumer->appendFusedProjectionWorkspaceRequirements(requirements, record.m(), columns, record.k());
        }
        DeviceWorkspaceManager workspace(DeviceId::cpu(), workspace_limit, memory);
        if (!workspace.allocate(requirements)) throw std::runtime_error("Replay workspace admission failed");
        const auto banks = borrowCPUWorkspace(source->contract(), record.m(), &workspace);
        const auto allocated = memory->claimedBytes(DeviceId::cpu(), PhysicalMemoryOwner::ExecutionWorkspace,
                                                     PhysicalMemoryMaterializationKind::NewAllocation);
        const std::vector<ITensorGemm::TensorProjectionDesc> projections{
            {engine.get(), output.get(), record.n()}, {sliced.get(), sliced_output.get(), sliced_n}};
        const std::vector<ITensorGemm::TensorProjectionDesc> singleton{
            {engine.get(), singleton_output.get(), record.n()}};
        std::ofstream stream(argv[2], std::ios::binary | std::ios::trunc);
        writeExact(stream, record.header.data(), sizeof(record.header));
        for (std::int32_t live : {record.m(), record.m() - 1, 0, record.m()})
        {
            for (const auto bank : {std::as_writable_bytes(banks.blocks), std::as_writable_bytes(banks.ranges),
                 std::as_writable_bytes(banks.row_ranges), std::as_writable_bytes(banks.butterfly),
                 std::as_writable_bytes(banks.transformed), std::as_writable_bytes(banks.active_rows)})
                std::fill(bank.begin(), bank.end(), std::byte{0xa5});
            const float guard = std::bit_cast<float>(std::uint32_t{0x7fc12345});
            std::fill_n(output->mutable_typed_data(), output->numel(), guard);
            std::fill_n(sliced_output->mutable_typed_data(), sliced_output->numel(), guard);
            std::fill_n(singleton_output->mutable_typed_data(), singleton_output->numel(), guard);
            engine->resetDynamicState();
            sliced->resetDynamicState();
            if (engine->hasDynamicStateActive() || sliced->hasDynamicStateActive())
                throw std::runtime_error("Native engine retained input-dependent state across reset");
            auto scope = engine->beginVerifierDecodeEquivalentScope(DeviceRowRange::deviceCounted(record.m(), &live));
            const bool success = record.entry() == Entry::Tensor
                ? engine->multiply_tensor(input.get(), output.get(), record.m(), record.n(), record.k(),
                    false, 1, 0, nullptr, nullptr, -1, &workspace)
                : engine->multiply_fused_verifier_rows_decode_equivalent(
                    input.get(), projections, record.m(), record.k(), nullptr, &workspace);
            if (!success) throw std::runtime_error("Native tensor engine rejected the recorded projection");
            if (!engine->multiply_fused_verifier_rows_decode_equivalent(
                    input.get(), singleton, record.m(), record.k(), nullptr, &workspace))
                throw std::runtime_error("Native singleton verifier rejected the recorded projection");
            if (allocated != memory->claimedBytes(DeviceId::cpu(), PhysicalMemoryOwner::ExecutionWorkspace,
                                                   PhysicalMemoryMaterializationKind::NewAllocation))
                throw std::runtime_error("Native replay changed its retained workspace allocation");
            writeExact(stream, &live, sizeof(live));
            writeExact(stream, record.transform() == Transform::PaleySylvester ? banks.transformed.data() : input->typed_data(),
                       record.input.size() * sizeof(float));
            writeExact(stream, banks.blocks.data(), banks.blocks.size_bytes());
            writeExact(stream, output->typed_data(), output->numel() * sizeof(float));
            if (record.entry() == Entry::FusedTensor)
                writeExact(stream, sliced_output->typed_data(), sliced_output->numel() * sizeof(float));
            writeExact(stream, singleton_output->typed_data(), singleton_output->numel() * sizeof(float));
            writeExact(stream, banks.active_rows.data(), banks.active_rows.size_bytes());
        }
        stream.flush();
        if (!stream) throw std::runtime_error("Replay output flush failed");
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "W4A4 CPU tensor replay rejected: " << error.what() << '\n';
        return 1;
    }
}
