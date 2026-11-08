/**
 * @file Test__W4A4ModelSource.cpp
 * @brief Native GGUF online-A4 identity, source lifetime and planning admission contracts.
 *
 * Tiny owned GGUF fixtures exercise the real directory parser, retained mapping,
 * bounded original-coordinate reader, final-weight Store and metadata broadcast.
 * Source transport may split K; published tensors may not. Complete predictor
 * names do not certify rotated embedding/hidden/head basis compatibility.
 * These are device-free source/lifecycle contracts, not real-model parity.
 */
#include "config/OrchestrationConfig.h"
#include "config/OrchestrationStartupPolicy.h"
#include "execution/mtp/MTPWeightManifest.h"
#include "kernels/cpu/gemm/CPUWeightPreparationMemory.h"
#include "loaders/PreparedWeightRepresentationContract.h"
#include "loaders/PreparedWeightStore.h"
#include "loaders/WeightManager.h"
#include "backends/BackendManager.h"
#include "planning/ModelMemoryProfile.h"
#include "planning/WeightMemoryEstimator.h"
#include "tensors/W4A4Tensor.h"
#include "../../utils/PlanningGGUFFixture.h"
#include "../../utils/ServingGGUFFixturePayload.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cstring>
#include <filesystem>
#include <fstream>

namespace llaminar2::w4a4
{
    namespace
    {
        constexpr std::size_t kRows = 256, kColumns = 512;
        constexpr const char *kWeight = "blk.0.ffn_down.weight";
        constexpr auto kOwner = PhysicalMemoryOwner::PrimaryModelWeights;

        /** @return Valid native rows whose original positions remain visible after slicing. */
        std::vector<SourceBlock> source(TensorType type)
        {
            std::vector<SourceBlock> blocks(kRows * kColumns / kBlockElements);
            for (std::size_t i = 0; i < blocks.size(); ++i)
            {
                blocks[i].unit = float(1 + (i / 2) % 8) * 0.125f;
                blocks[i].descriptor = 0x60004321;
                blocks[i].rotation_length = type == TensorType::W4A4 ? kColumns : 128;
                std::fill_n(blocks[i].codes, 128, std::uint8_t(i));
            }
            return blocks;
        }

        /** @brief Fill exactly one fixture operand without changing its directory or allocating model tensors. */
        void populate(test::PlanningGGUFFixture &file, const std::vector<SourceBlock> &blocks,
                      const std::string &name = kWeight)
        {
            file.writePayload(name, 0, {reinterpret_cast<const uint8_t *>(blocks.data()),
                                        blocks.size() * sizeof(SourceBlock)});
        }

        /** @brief Original-source defects, applied before the loader seals its file. */
        enum class StreamedSourceChange { None, LastClip, LastUnit, Truncated };

        /** @brief Own one sealed source and one reusable, exactly admitted final destination. */
        struct StreamedSourceFixture
        {
            test::PlanningGGUFFixture file;
            std::vector<SourceBlock> blocks;
            ModelLoader loader;
            std::shared_ptr<PhysicalMemoryAuthority> memory;
            PreparedWeightStore store{ModelContextId{183}};
            static constexpr const char *name = "blk.1.ffn_down.weight";

            /**
             * @brief Populate the final file tensor and admit only the largest sequential destination.
             * @param type Original native W4A4/H identity.
             * @param change Optional corruption outside an ordinary selected-row request.
             */
            explicit StreamedSourceFixture(TensorType type, StreamedSourceChange change = StreamedSourceChange::None)
                : file(false, false, test::PlanningGGUFFixture::sourceType(type)), blocks(source(type))
            {
                if (change == StreamedSourceChange::LastClip)
                    blocks.back().clip_bits = std::bit_cast<uint32_t>(0.5f);
                if (change == StreamedSourceChange::LastUnit) blocks.back().unit = -1.0f;
                populate(file, blocks, name);
                if (change == StreamedSourceChange::Truncated)
                    std::filesystem::resize_file(file.path(), std::filesystem::file_size(file.path()) - 1);
                loader.setUseMmap(false);
                if (!loader.loadModel(file.path())) throw std::runtime_error("Cannot load native source-read fixture");
                const auto quote = CPUWeightPreparationMemory::sourceNative(tensorTypeName(type), kRows, kColumns);
                PhysicalMemoryPlanBuilder budget;
                budget.add({0, DeviceId::cpu(), quote.persistent_bytes, quote.persistent_bytes}, kOwner, quote.persistent_bytes);
                memory = std::make_shared<PhysicalMemoryAuthority>(
                    std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(budget.build()), 0);
                store.installPhysicalMemoryAuthority(memory);
            }

            /**
             * @brief Begin an unpublished final writer in the existing Store/PMA authority.
             * @param window Complete-K original row interval; each caller retires it before the next.
             * @return Consuming production writer, without staging or a source mirror.
             */
            auto begin(const NativeWeightSourceWindow &window)
            {
                const CPUPreparedWeightSource spec{.original = loader.nativeWeightSourceGeometry(name),
                    .range = window, .consumer = CPUWeightConsumer::ResidentTensor};
                WeightBinding binding;
                binding.binding_id = 1;
                binding.identity = makeSourceWeightIdentity(name, ModelContextId{183}, 1);
                binding.residency.home_device = DeviceId::cpu();
                binding.residency.resident_device = DeviceId::cpu();
                binding.residency.raw_host_data_available = false;
                binding.immutable = true;
                return store.beginCPUPreparation(binding, spec, kOwner, CPUStoragePlacement::local());
            }
        };

        /**
         * @return Actual kernel-accounted bytes returned by read-family syscalls, including cached file reads.
         * @throws std::runtime_error If native task-I/O accounting is absent or incomplete; never skip or synthesize it.
         */
        uint64_t sourceReadBytes()
        {
            std::ifstream stream("/proc/self/io");
            std::string key;
            uint64_t value = 0;
            while (stream >> key >> value) if (key == "rchar:") return value;
            throw std::runtime_error("W4A4 source-read economy requires native /proc/self/io rchar accounting");
        }
    }

    /** @test Whole and selected requests publish only fully admitted bytes, including repeated failures. */
    TEST(W4A4ModelSourceContract, CompleteRequestsRetainFinalBytesAcrossAdmissionOrder)
    {
        const NativeWeightSourceWindow whole{.rows = kRows, .columns = kColumns,
            .native_row_bytes = 2 * sizeof(SourceBlock)};
        const NativeWeightSourceWindow selected{.first_row = 3, .rows = 3, .columns = kColumns,
            .native_row_bytes = 2 * sizeof(SourceBlock)};
        for (auto type : {TensorType::W4A4, TensorType::W4A4H})
        for (const auto &first : {whole, selected})
        for (auto change : {StreamedSourceChange::None, StreamedSourceChange::LastClip,
                            StreamedSourceChange::LastUnit, StreamedSourceChange::Truncated})
        {
            SCOPED_TRACE(::testing::Message() << tensorTypeName(type) << " first_rows=" << first.rows
                << " change=" << int(change));
            StreamedSourceFixture fixture(type, change);
            for (const auto &window : {first, whole, selected, whole})
            {
                const auto publish = [&] {
                    auto writer = fixture.begin(window);
                    EXPECT_FALSE(writer->stagingRequest());
                    writer->readDirect(window, [&](std::span<std::byte> destination) {
                        fixture.loader.readNativeWeightSourceWindow(fixture.name, window, destination);
                    });
                    return fixture.store.finishCPUPreparation(std::move(writer));
                };
                if (change == StreamedSourceChange::None)
                {
                    const auto result = publish();
                    ASSERT_NE(result.tensor, nullptr);
                    EXPECT_EQ(result.tensor->shape(), (std::vector<size_t>{window.rows, kColumns}));
                    EXPECT_EQ(std::memcmp(result.tensor->raw_data(), fixture.blocks.data() + window.first_row * 2,
                        window.nativeBytes()), 0);
                    fixture.store.clear();
                }
                else
                {
                    EXPECT_THROW(static_cast<void>(publish()), std::exception);
                    EXPECT_FALSE(fixture.store.preparedRefForBinding(1, DeviceId::cpu()));
                }
                EXPECT_EQ(fixture.memory->claimedBytes(DeviceId::cpu(), kOwner,
                    PhysicalMemoryMaterializationKind::NewAllocation), 0u);
                EXPECT_EQ(fixture.memory->claimedBytes(DeviceId::cpu(), PhysicalMemoryOwner::WeightLoadStaging,
                    PhysicalMemoryMaterializationKind::NewAllocation), 0u);
            }
        }
    }

    /** @test A first complete matrix is read once; later complete/selected destinations still receive their own bytes. */
    TEST(W4A4SourceReadEconomy, CompleteAdmissionDoesNotRereadTheFinalDestination)
    {
        const NativeWeightSourceWindow whole{.rows = kRows, .columns = kColumns,
            .native_row_bytes = 2 * sizeof(SourceBlock)};
        const NativeWeightSourceWindow selected{.first_row = 3, .rows = 3, .columns = kColumns,
            .native_row_bytes = 2 * sizeof(SourceBlock)};
        // The observer itself and a filebuf read-ahead may add small reads. This
        // bound is much smaller than the 72-KiB matrix, so a second payload read
        // cannot pass. Page-cache warmth cannot hide it from the kernel's rchar.
        constexpr uint64_t observation_and_buffer_bytes = 16 * 1024;
        for (auto type : {TensorType::W4A4, TensorType::W4A4H})
        {
            StreamedSourceFixture fixture(type);
            size_t invocation = 0;
            for (const auto &window : {whole, selected, whole})
            {
                SCOPED_TRACE(::testing::Message() << tensorTypeName(type) << " rows=" << window.rows);
                {
                    auto writer = fixture.begin(window);
                    writer->readDirect(window, [&](std::span<std::byte> destination) {
                        const auto before = sourceReadBytes();
                        fixture.loader.readNativeWeightSourceWindow(fixture.name, window, destination);
                        const auto after = sourceReadBytes();
                        ASSERT_GE(after, before);
                        const auto bytes = after - before;
                        EXPECT_GE(bytes, window.nativeBytes());
                        EXPECT_LE(bytes, window.nativeBytes() + observation_and_buffer_bytes);
                        RecordProperty(std::string(tensorTypeName(type)) + "_kernel_read_bytes_" +
                            std::to_string(invocation), std::to_string(bytes));
                    });
                    const auto result = fixture.store.finishCPUPreparation(std::move(writer));
                    ASSERT_NE(result.tensor, nullptr);
                    EXPECT_EQ(std::memcmp(result.tensor->raw_data(), fixture.blocks.data() + window.first_row * 2,
                        window.nativeBytes()), 0);
                    fixture.store.clear();
                }
                ++invocation;
                EXPECT_EQ(fixture.memory->claimedBytes(DeviceId::cpu(), kOwner,
                    PhysicalMemoryMaterializationKind::NewAllocation), 0u);
            }
        }
    }

    TEST(W4A4ModelSourceFixtureContract, NormalizedDirectoryPreservesOriginalNativeRows)
    {
        const std::array<std::array<GGUFTensorType, 2>, 3> formats{{
            {GGUFTensorType::W4A4, GGUFTensorType::W4A4},
            {GGUFTensorType::W4A4H, GGUFTensorType::W4A4H},
            {GGUFTensorType::W4A4, GGUFTensorType::W4A4H}}};
        int publications = 0;
        for (const auto &format : formats)
        {
            test::PlanningGGUFFixture file(false, false, format[0], 512, std::nullopt,
                8, test::PlanningGGUFFixture::Geometry::FullAttentionGraph, 2, 8, format[1]);
            test::populateServingGGUFFixture(file);
            ModelLoader loader;
            loader.setUseMmap(true);
            ASSERT_TRUE(loader.loadModel(file.path()));
            int native_sources = 0;
            for (const auto &info : loader.getModel().tensors)
            {
                if (info.type != GGUFTensorType::W4A4 && info.type != GGUFTensorType::W4A4H) continue;
                SCOPED_TRACE(info.name);
                const bool down = info.name.ends_with("ffn_down.weight");
                const std::size_t rows = down ? 256 : 512, columns = down ? 512 : 256;
                ASSERT_EQ(info.dimensions, (std::vector<std::uint64_t>{rows, columns}));
                const auto type = info.type == GGUFTensorType::W4A4 ? TensorType::W4A4 : TensorType::W4A4H;
                auto tensor = loader.loadTensor(info.name);
                ASSERT_NE(tensor, nullptr);
                ASSERT_EQ(tensor->native_type(), type);
                EXPECT_EQ(tensor->shape(), (std::vector<std::size_t>{rows, columns}));
                const auto *native = dynamic_cast<const W4A4Tensor *>(tensor.get());
                ASSERT_NE(native, nullptr);
                const auto *blocks = native->typed_data();
                const auto blocks_per_row = columns / kBlockElements;
                ASSERT_EQ(tensor->size_bytes(), rows * blocks_per_row * sizeof(SourceBlock));
                // Check original rows independently of the payload generator.
                // A transposition leaves this byte extent unchanged, but moves
                // the unit boundary and, for identity, changes the source K.
                for (std::size_t row = 0; row < rows; ++row)
                    for (std::size_t block = 0; block < blocks_per_row; ++block)
                    {
                        const auto index = row * blocks_per_row + block;
                        EXPECT_EQ(std::bit_cast<std::uint32_t>(blocks[index].unit),
                            std::bit_cast<std::uint32_t>(float(1 + row % 8) / 4096.0f));
                        EXPECT_EQ(blocks[index].rotation_length, type == TensorType::W4A4 ? int(columns) : 128);
                        EXPECT_EQ(blocks[index].descriptor, 0x60004321);
                        EXPECT_TRUE(std::all_of(std::begin(blocks[index].codes), std::end(blocks[index].codes),
                            [index](std::uint8_t code) { return code == std::uint8_t(index); }));
                    }
                ++native_sources;
                ++publications;
            }
            EXPECT_EQ(native_sources, 6);
        }
        EXPECT_EQ(publications, 18);
        RecordProperty("complete_original_k_source_publications", publications);
    }

    TEST(W4A4ModelSourceLifecycle, CompleteRowsAndEnginesRetainOriginalFileAfterLoaderRetirement)
    {
        for (auto type : {TensorType::W4A4, TensorType::W4A4H})
        {
            test::PlanningGGUFFixture file(false, false, test::PlanningGGUFFixture::sourceType(type));
            const auto blocks = source(type);
            populate(file, blocks);
            auto loader = std::make_unique<ModelLoader>();
            ASSERT_TRUE(loader->loadModel(file.path()));
            const auto *info = loader->getModel().findTensor(kWeight);
            ASSERT_NE(info, nullptr);
            EXPECT_EQ(info->getTypeSize(), sizeof(SourceBlock));
            EXPECT_EQ(info->getBlockSize(), kBlockElements);
            EXPECT_TRUE(info->isQuantized());
            EXPECT_EQ(info->size_bytes, blocks.size() * sizeof(SourceBlock));
            const auto original_offset = loader->getModel().data_offset + info->offset;
            std::weak_ptr<MmapRegion> mapping = loader->getMmapRegion(info);
            auto complete = loader->loadTensor(kWeight);
            auto selected = loader->loadTensorRowSlice(kWeight, 3, 6);
            auto unsliced_columns = loader->loadTensorColumnSlice(kWeight, 0, kColumns);
            ASSERT_NE(complete, nullptr);
            ASSERT_NE(selected, nullptr);
            ASSERT_NE(unsliced_columns, nullptr);
            ASSERT_EQ(selected->native_type(), type);
            EXPECT_EQ(selected->shape(), (std::vector<std::size_t>{3, kColumns}));
            EXPECT_TRUE(selected->is_mmap_data());
            EXPECT_EQ(std::memcmp(selected->raw_data(), blocks.data() + 6, 6 * sizeof(SourceBlock)), 0);
            const auto provenance = MmapRegion::resolveFileSource(selected->raw_data(), 6 * sizeof(SourceBlock));
            ASSERT_TRUE(provenance);
            EXPECT_EQ(provenance->offset, original_offset + 6 * sizeof(SourceBlock));
            auto engine = selected->createGemm();
            ASSERT_NE(engine, nullptr);
            loader.reset();
            complete.reset();
            unsliced_columns.reset();
            selected->release_raw_data();
            selected.reset();
            EXPECT_FALSE(mapping.expired());
            EXPECT_EQ(engine->packedWeightBytes(), 6 * sizeof(SourceBlock));
            engine.reset();
            EXPECT_TRUE(mapping.expired());
        }
    }

    TEST(W4A4ModelSourceContract, SourceTransportFragmentsPublishOnlyCompleteOriginalK)
    {
        for (auto type : {TensorType::W4A4, TensorType::W4A4H})
        {
            test::PlanningGGUFFixture file(false, false, test::PlanningGGUFFixture::sourceType(type));
            const auto blocks = source(type);
            populate(file, blocks);
            ModelLoader loader;
            loader.setUseMmap(false);
            ASSERT_TRUE(loader.loadModel(file.path()));
            const auto original = loader.nativeWeightSourceGeometry(kWeight);
            EXPECT_EQ(original.original_shape, (std::vector<std::size_t>{kColumns, kRows}));
            EXPECT_EQ(original.original_type, type);
            EXPECT_EQ(original.native_group_elements, kBlockElements);
            EXPECT_EQ(original.native_group_bytes, kSourceBlockBytes);
            EXPECT_THROW(static_cast<void>(loader.loadTensor(kWeight)), std::logic_error);
            const CPUPreparedWeightSource spec{.original = original,
                .range = {.first_row = 3, .rows = 3, .columns = kColumns,
                    .native_row_bytes = 2 * sizeof(SourceBlock)}, .consumer = CPUWeightConsumer::PreparedGemm};
            const auto quoted = CPUWeightPreparationMemory::sourceNative(tensorTypeName(type), 3, kColumns);
            EXPECT_EQ(quoted.persistent_payload_bytes, 6 * sizeof(SourceBlock));
            EXPECT_EQ(quoted.temporary_bytes, 0u);
            const auto allocator = DeviceId::cpu();
            PhysicalMemoryPlanBuilder plan;
            plan.add({0, allocator, quoted.persistent_bytes, quoted.persistent_bytes}, kOwner, quoted.persistent_bytes);
            auto memory = std::make_shared<PhysicalMemoryAuthority>(
                std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(plan.build()), 0);
            constexpr ModelContextId model{178};
            PreparedWeightStore store(model);
            store.installPhysicalMemoryAuthority(memory);
            WeightBinding binding;
            binding.binding_id = 1;
            binding.identity = makeSourceWeightIdentity(kWeight, model, 1);
            binding.residency.home_device = allocator;
            binding.residency.resident_device = allocator;
            binding.residency.raw_host_data_available = false;
            binding.immutable = true;
            auto writer = store.beginCPUPreparation(binding, spec, kOwner, CPUStoragePlacement::local());
            EXPECT_FALSE(writer->stagingRequest());
            for (std::size_t i = 0; i < 6; ++i)
            {
                const NativeWeightSourceWindow window{.first_row = 3 + i / 2, .rows = 1,
                    .first_column = (i % 2) * kBlockElements, .columns = kBlockElements,
                    .native_row_bytes = sizeof(SourceBlock)};
                writer->readDirect(window, [&](std::span<std::byte> destination) {
                    loader.readNativeWeightSourceWindow(kWeight, window, destination);
                });
                EXPECT_FALSE(store.preparedRefForBinding(1, allocator));
            }
            auto committed = store.finishCPUPreparation(std::move(writer));
            ASSERT_NE(committed.tensor, nullptr);
            ASSERT_TRUE(committed.prepared);
            EXPECT_EQ(committed.tensor->shape(), (std::vector<std::size_t>{3, kColumns}));
            EXPECT_EQ(std::memcmp(committed.tensor->raw_data(), blocks.data() + 6, 6 * sizeof(SourceBlock)), 0);
            EXPECT_EQ(memory->claimedBytes(allocator, kOwner, PhysicalMemoryMaterializationKind::NewAllocation),
                      quoted.persistent_bytes);
            store.clear();
            committed = {};
            EXPECT_EQ(memory->claimedBytes(allocator, kOwner, PhysicalMemoryMaterializationKind::NewAllocation), 0u);
        }
    }

    TEST(W4A4ModelSourceContract, StreamedRowsCannotHideInvalidUnselectedSourcePolicy)
    {
        enum class SourceChange { None, Clip, Rotation, Unit, Truncated };
        for (auto type : {TensorType::W4A4, TensorType::W4A4H})
        for (auto change : {SourceChange::None, SourceChange::Clip, SourceChange::Rotation,
                            SourceChange::Unit, SourceChange::Truncated})
        {
            SCOPED_TRACE(::testing::Message() << tensorTypeName(type) << " change=" << int(change));
            test::PlanningGGUFFixture file(false, false, test::PlanningGGUFFixture::sourceType(type));
            auto blocks = source(type);
            const std::string name = "blk.1.ffn_down.weight"; // Final payload permits an exact short-file control.
            switch (change)
            {
            case SourceChange::None: case SourceChange::Truncated: break;
            case SourceChange::Clip: blocks.back().clip_bits = std::bit_cast<uint32_t>(0.5f); break;
            case SourceChange::Rotation: blocks.back().rotation_length /= 2; break;
            case SourceChange::Unit: blocks.back().unit = -1.0f; break;
            }
            populate(file, blocks, name);
            if (change == SourceChange::Truncated)
                std::filesystem::resize_file(file.path(), std::filesystem::file_size(file.path()) - 1);
            ModelLoader loader;
            loader.setUseMmap(false);
            ASSERT_TRUE(loader.loadModel(file.path()));
            const CPUPreparedWeightSource spec{.original = loader.nativeWeightSourceGeometry(name),
                .range = {.first_row = 3, .rows = 3, .columns = kColumns,
                    .native_row_bytes = 2 * sizeof(SourceBlock)}, .consumer = CPUWeightConsumer::PreparedGemm};
            const auto quoted = CPUWeightPreparationMemory::sourceNative(tensorTypeName(type), 3, kColumns);
            const auto allocator = DeviceId::cpu();
            PhysicalMemoryPlanBuilder plan;
            plan.add({0, allocator, quoted.persistent_bytes, quoted.persistent_bytes}, kOwner, quoted.persistent_bytes);
            auto memory = std::make_shared<PhysicalMemoryAuthority>(
                std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(plan.build()), 0);
            constexpr ModelContextId model{181};
            PreparedWeightStore store(model);
            store.installPhysicalMemoryAuthority(memory);
            WeightBinding binding;
            binding.binding_id = 1;
            binding.identity = makeSourceWeightIdentity(name, model, 1);
            binding.residency.home_device = allocator;
            binding.residency.resident_device = allocator;
            binding.residency.raw_host_data_available = false;
            binding.immutable = true;
            const auto publish = [&] {
                auto writer = store.beginCPUPreparation(binding, spec, kOwner, CPUStoragePlacement::local());
                EXPECT_FALSE(writer->stagingRequest());
                // Transport may split K and the final allocation contains only
                // selected rows; neither can narrow the source policy authority.
                for (size_t i = 0; i < 6; ++i)
                {
                    const NativeWeightSourceWindow window{.first_row = 3 + i / 2, .rows = 1,
                        .first_column = (i % 2) * kBlockElements, .columns = kBlockElements,
                        .native_row_bytes = sizeof(SourceBlock)};
                    writer->readDirect(window, [&](std::span<std::byte> destination) {
                        loader.readNativeWeightSourceWindow(name, window, destination);
                    });
                }
                return store.finishCPUPreparation(std::move(writer));
            };
            if (change == SourceChange::None)
            {
                auto committed = publish();
                ASSERT_NE(committed.tensor, nullptr);
                EXPECT_EQ(std::memcmp(committed.tensor->raw_data(), blocks.data() + 6, 6 * sizeof(SourceBlock)), 0);
                store.clear();
            }
            else
            {
                EXPECT_THROW(static_cast<void>(publish()), std::exception);
                EXPECT_FALSE(store.preparedRefForBinding(1, allocator));
            }
            EXPECT_EQ(memory->claimedBytes(allocator, kOwner, PhysicalMemoryMaterializationKind::NewAllocation), 0u);
            EXPECT_EQ(memory->claimedBytes(allocator, PhysicalMemoryOwner::WeightLoadStaging,
                PhysicalMemoryMaterializationKind::NewAllocation), 0u);
        }
    }

    TEST(W4A4ModelSourceLifecycle, WeightManagerRetainsOnlyAdmittedCompleteKRows)
    {
        if (!hasCPUBackend()) initCPUBackend(-1);
        for (auto type : {TensorType::W4A4, TensorType::W4A4H})
        for (bool mapped : {false, true})
        for (bool corrupt_unselected : {false, true})
        {
            SCOPED_TRACE(::testing::Message() << tensorTypeName(type) << " mapped=" << mapped
                << " corrupt_unselected=" << corrupt_unselected);
            test::PlanningGGUFFixture file(false, false, test::PlanningGGUFFixture::sourceType(type));
            auto blocks = source(type);
            if (corrupt_unselected) blocks.back().clip_bits = std::bit_cast<uint32_t>(0.5f);
            populate(file, blocks);
            const auto quote = CPUWeightPreparationMemory::sourceNative(tensorTypeName(type), 3, kColumns);
            PhysicalMemoryPlanBuilder admission;
            const auto device = DeviceId::cpu();
            admission.add({0, device, quote.persistent_bytes, quote.persistent_bytes}, kOwner, quote.persistent_bytes);
            auto memory = std::make_shared<PhysicalMemoryAuthority>(
                std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(admission.build()), 0);
            std::shared_ptr<ITensorGemm> retained;
            {
                ModelLoader loader;
                loader.setUseMmap(mapped);
                ASSERT_TRUE(loader.loadModel(file.path()));
                WeightManager manager(loader);
                manager.installPhysicalMemoryAuthority(memory);
                InferenceStrategy strategy;
                strategy.mode = WeightInferenceMode::SingleDevice;
                strategy.model_id = ModelContextId{182};
                strategy.devices = {device};
                WeightPlan plan(strategy);
                WeightRequirement requirement;
                requirement.canonical_name = kWeight;
                requirement.target_device = device;
                requirement.lookup_device = device;
                requirement.slice = {.source_rows = kRows, .source_cols = kColumns,
                    .row_start = 3, .row_count = 3, .col_start = 0, .col_count = kColumns};
                plan.add(requirement);
                if (corrupt_unselected)
                {
                    EXPECT_THROW(static_cast<void>(manager.materialize(plan)), std::exception);
                }
                else
                {
                    const auto frozen = manager.materialize(plan);
                    const auto &binding = frozen.layer(0, "ffn_down.weight");
                    ASSERT_TRUE(binding.prepared);
                    ASSERT_NE(binding.tensor, nullptr);
                    EXPECT_EQ(binding.tensor->native_type(), type);
                    EXPECT_EQ(binding.tensor->shape(), (std::vector<size_t>{3, kColumns}));
                    EXPECT_FALSE(binding.tensor->is_mmap_data());
                    EXPECT_EQ(std::memcmp(binding.tensor->raw_data(), blocks.data() + 6, 6 * sizeof(SourceBlock)), 0);
                    retained = manager.preparedWeightStore()->gemmKernelLifetime(*binding.prepared);
                    ASSERT_NE(retained, nullptr);
                    const auto repeated = manager.materialize(plan);
                    EXPECT_EQ(repeated.layer(0, "ffn_down.weight").tensor, binding.tensor);
                    EXPECT_EQ(memory->claimedBytes(device, kOwner, PhysicalMemoryMaterializationKind::NewAllocation),
                        quote.persistent_bytes);
                }
                EXPECT_EQ(memory->claimedBytes(device, PhysicalMemoryOwner::WeightLoadStaging,
                    PhysicalMemoryMaterializationKind::NewAllocation), 0u);
            }
            std::filesystem::remove(file.path());
            EXPECT_EQ(memory->claimedBytes(device, kOwner, PhysicalMemoryMaterializationKind::NewAllocation),
                corrupt_unselected ? 0u : quote.persistent_bytes);
            if (retained)
            {
                EXPECT_EQ(retained->get_n(), 3);
                EXPECT_EQ(retained->get_k(), kColumns);
                EXPECT_EQ(retained->packedWeightBytes(), 6 * sizeof(SourceBlock));
            }
            retained.reset();
            EXPECT_EQ(memory->claimedBytes(device, kOwner, PhysicalMemoryMaterializationKind::NewAllocation), 0u);
        }
    }

    TEST(W4A4ModelSourceContract, ConversionsKShardsAndMalformedNativeSourcesCannotPublish)
    {
        for (auto type : {TensorType::W4A4, TensorType::W4A4H})
        {
            test::PlanningGGUFFixture file(false, false, test::PlanningGGUFFixture::sourceType(type));
            auto blocks = source(type);
            populate(file, blocks);
            {
                ModelLoader loader;
                ASSERT_TRUE(loader.loadModel(file.path()));
                for (auto precision : {WeightPrecision::CONVERT_TO_FP32, WeightPrecision::CONVERT_TO_FP16,
                                       WeightPrecision::CONVERT_TO_BF16, WeightPrecision::CONVERT_TO_INT8})
                {
                    EXPECT_THROW(static_cast<void>(loader.loadTensor(kWeight, DeviceId::cpu(), precision)), std::invalid_argument);
                    EXPECT_THROW(static_cast<void>(loader.loadTensorRowSlice(kWeight, 3, 6, DeviceId::cpu(), precision)), std::invalid_argument);
                    EXPECT_THROW(static_cast<void>(PreparedWeightRepresentationContract::resolveCPUWeightPrecision(
                        inferWeightRole(kWeight), type, precision, ModelPreparedWeightRepresentation::SourceNative)), std::invalid_argument);
                }
                EXPECT_THROW(static_cast<void>(loader.loadTensorColumnSlice(kWeight, 0, 256)), std::invalid_argument);
                EXPECT_THROW(static_cast<void>(loader.loadTensorColumnSlice(kWeight, 256, 512)), std::invalid_argument);
            }
            // The original matrix's policy cannot be narrowed to evade a bad
            // unselected row. No mapping is live when the fixture is changed.
            blocks.back().clip_bits = std::bit_cast<std::uint32_t>(0.5f);
            populate(file, blocks);
            {
                ModelLoader loader;
                ASSERT_TRUE(loader.loadModel(file.path()));
                EXPECT_THROW(static_cast<void>(loader.loadTensorRowSlice(kWeight, 3, 6)), std::invalid_argument);
            }
            const std::string last = "blk.1.ffn_down.weight";
            populate(file, source(type), last);
            std::filesystem::resize_file(file.path(), std::filesystem::file_size(file.path()) - 1);
            ModelLoader truncated;
            ASSERT_TRUE(truncated.loadModel(file.path()));
            EXPECT_THROW(static_cast<void>(truncated.loadTensor(last)), std::invalid_argument);
            test::PlanningGGUFFixture rank_three(true, false, test::PlanningGGUFFixture::sourceType(type));
            ModelLoader unsupported;
            EXPECT_THROW(static_cast<void>(unsupported.loadModel(rank_three.path())), std::invalid_argument);
        }
    }

    TEST(W4A4ModelPlanningContract, BroadcastPreservesNativeBOMAndRejectsUnqualifiedMTPBasis)
    {
        for (auto type : {TensorType::W4A4, TensorType::W4A4H})
        {
            test::PlanningGGUFFixture file(false, true, test::PlanningGGUFFixture::sourceType(type));
            ModelLoader loader;
            loader.setUseMmap(false);
            ASSERT_TRUE(loader.loadModel(file.path()));
            auto profile = ModelMemoryProfile::fromLoader(loader);
            const auto wire = profile.serialize();
            const auto published = ModelMemoryProfile::deserialize(wire.data(), wire.size());
            ASSERT_EQ(published.tensors.size(), profile.tensors.size());
            EXPECT_EQ(published.mainLayerCount(), 2);
            EXPECT_EQ(mainLayerCountExcludingMTP(loader, loader.architecture(), 3), 2);
            for (std::size_t i = 0; i < profile.tensors.size(); ++i)
            {
                EXPECT_EQ(published.tensors[i].quant_type, profile.tensors[i].quant_type);
                EXPECT_EQ(published.tensors[i].native_bytes, profile.tensors[i].native_bytes);
                EXPECT_EQ(published.tensors[i].K, profile.tensors[i].K);
            }
            const auto *original = loader.getModel().findTensor(kWeight);
            ASSERT_NE(original, nullptr);
            const auto found = std::find_if(published.tensors.begin(), published.tensors.end(),
                [](const auto &tensor) { return tensor.name == kWeight; });
            ASSERT_NE(found, published.tensors.end());
            EXPECT_EQ(found->quant_type, tensorTypeName(type));
            EXPECT_EQ(found->native_bytes, original->size_bytes);
            EXPECT_EQ(WeightMemoryEstimator::getNativeBytesPerWeight(found->quant_type), 0.5625f);
            EXPECT_EQ(WeightMemoryEstimator::getCPUPackedBytesPerWeight(found->quant_type), 0.5625f);
            EXPECT_THROW(static_cast<void>(WeightMemoryEstimator::getGPUPackedBytesPerWeight(found->quant_type, kColumns)),
                         std::invalid_argument);
            // This is a source-only profile of one real directory operand, not
            // an alternative BOM formula or claim of graph workspace admission.
            auto single = published;
            single.tensors = {*found};
            single.total_native_bytes = found->native_bytes;
            const auto estimate = WeightMemoryEstimator::estimate(single, DeviceId::cpu());
            const auto quote = CPUWeightPreparationMemory::sourceNative(found->quant_type, kRows, kColumns);
            EXPECT_EQ(estimate.native_bytes, found->native_bytes);
            EXPECT_EQ(estimate.device_bytes, quote.persistent_bytes);
            EXPECT_THROW(static_cast<void>(WeightMemoryEstimator::estimate(single, DeviceId::cuda(0))), std::invalid_argument);
            for (bool explicit_mtp : {false, true})
            {
                const auto source_manifest = discoverMTPWeightManifest(loader, loader.architecture(), 3, explicit_mtp);
                const auto remote_manifest = discoverMTPWeightManifest(published, explicit_mtp);
                EXPECT_FALSE(source_manifest.available);
                EXPECT_FALSE(remote_manifest.available);
                EXPECT_EQ(source_manifest.diagnostic, remote_manifest.diagnostic);
                EXPECT_NE(source_manifest.diagnostic.find("basis compatibility has not been established"), std::string::npos);
            }
            EXPECT_THROW(static_cast<void>(requireMTPWeightManifest(loader, loader.architecture(), 3)), std::invalid_argument);
            for (auto activation : {MTPActivationPolicy::Automatic, MTPActivationPolicy::Enabled})
            {
                OrchestrationConfig config;
                config.mtp_activation_policy = activation;
                EXPECT_THROW(resolveMTPStartupPolicy(config, published), std::invalid_argument);
            }
            OrchestrationConfig disabled;
            disabled.mtp_activation_policy = MTPActivationPolicy::Disabled;
            EXPECT_NO_THROW(resolveMTPStartupPolicy(disabled, published));
            EXPECT_FALSE(disabled.mtp.enabled);
        }
        // A normal complete source still retains the existing learned-block
        // admission; the negative decision is scoped to the online-A4 basis.
        test::PlanningGGUFFixture ordinary(false, true);
        ModelLoader loader;
        loader.setUseMmap(false);
        ASSERT_TRUE(loader.loadModel(ordinary.path()));
        EXPECT_TRUE(discoverMTPWeightManifest(loader, loader.architecture(), 3, true).available);
        EXPECT_TRUE(discoverMTPWeightManifest(ModelMemoryProfile::fromLoader(loader), true).available);
    }
} // namespace llaminar2::w4a4
