/**
 * @file Test__W4A4SplitModelSource.cpp
 * @brief Native split-GGUF source coordinates, admission isolation and final-reader lifetimes.
 *
 * Each private two-part fixture contains distinct W4A4/H bytes and policies,
 * different directory lengths and nonzero tensor offsets. Actual loader and
 * Store/PMA interfaces must retain the correct file and admit every original
 * row before publishing a selection. Synthetic payloads prove storage contracts,
 * not numerical replay or model parity; the pinned upstream recordings own math.
 */
#include "kernels/cpu/gemm/CPUWeightPreparationMemory.h"
#include "loaders/ModelLoader.h"
#include "loaders/PreparedWeightStore.h"
#include "planning/PhysicalMemoryAuthority.h"
#include "tensors/W4A4Tensor.h"
#include "../../utils/W4A4SourceFixture.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace llaminar2::w4a4
{
    namespace
    {
        constexpr size_t kRows = 256, kColumns = 512;
        constexpr auto kOwner = PhysicalMemoryOwner::PrimaryModelWeights;
        constexpr ModelContextId kModel{184};

        /** @brief Mutations affect only the explicitly selected unpublished fixture file. */
        enum class SplitFault
        {
            None, LastClip, LastUnit, Truncated, PartialK, RankThree,
            AlignmentZero, AlignmentNonPowerOfTwo, AlignmentWrongType, AlignmentDuplicate
        };

        /** @brief Own two uniquely named GGUF parts, never a shared model or cache path. */
        struct SplitSourceFiles final
        {
            std::filesystem::path directory;
            std::array<std::filesystem::path, 2> files;
            std::array<TensorType, 2> types;
            std::array<std::vector<SourceBlock>, 2> blocks;
            std::array<uint64_t, 2> data_offsets{};
            std::array<uint64_t, 2> tensor_offsets{64, 128};

            /**
             * @brief Serialize two independently calibrated native matrices before source admission.
             * @param first_type W4A4/H identity in part zero; part one uses the other identity.
             * @param alignment Common GGUF alignment, exercised at 32 and 64 bytes.
             * @param fault Optional bad policy, extent or directory geometry in part one.
             * @throws std::runtime_error If the uniquely owned fixture directory cannot be created.
             */
            SplitSourceFiles(TensorType first_type, uint32_t alignment, SplitFault fault = SplitFault::None)
                : SplitSourceFiles(first_type, std::array<std::optional<uint32_t>, 2>{alignment, alignment}, fault) {}

            /**
             * @brief Give each file its own declared alignment or the GGUF default.
             * @param first_type Native identity in part zero; part one uses the other format.
             * @param alignments Per-file metadata; absence means the standard 32-byte layout.
             * @param fault Optional mutation in exactly one unpublished file.
             * @param fault_part Original file ordinal receiving that mutation.
             * @throws std::invalid_argument For an invalid test-owned file ordinal.
             * @throws std::runtime_error If the uniquely owned directory cannot be created.
             */
            SplitSourceFiles(TensorType first_type, std::array<std::optional<uint32_t>, 2> alignments,
                SplitFault fault = SplitFault::None, size_t fault_part = 1)
                : types{first_type, first_type == TensorType::W4A4 ? TensorType::W4A4H : TensorType::W4A4}
            {
                if (fault_part >= files.size()) throw std::invalid_argument("Invalid split fixture mutation ordinal");
                for (const auto alignment : alignments)
                    if (alignment && (*alignment == 0 || (*alignment & (*alignment - 1)) != 0))
                        throw std::invalid_argument("Fixture physical alignment must remain valid");
                if (fault >= SplitFault::AlignmentZero && !alignments[fault_part])
                    throw std::invalid_argument("Alignment mutation requires a declared fixture field");
                std::string pattern = (std::filesystem::temp_directory_path() / "llaminar_w4a4_split_XXXXXX").string();
                const auto *created = ::mkdtemp(pattern.data());
                if (!created) throw std::runtime_error("Cannot create private W4A4 split fixture");
                directory = created;
                files = {directory / "weights-00001-of-00002.gguf", directory / "weights-00002-of-00002.gguf"};
                try
                {
                    for (size_t part = 0; part < 2; ++part)
                    {
                        blocks[part] = test::w4a4SourceBlocks(types[part], kRows, kColumns);
                        for (auto &block : blocks[part])
                        {
                            block.unit *= float(part + 1);
                            block.clip_bits = std::bit_cast<uint32_t>(part == 0 ? 0.75f : 0.875f);
                            for (auto &code : block.codes) code = uint8_t(code ^ (part == 0 ? 0u : 0x5au));
                        }
                    }
                    if (fault == SplitFault::LastClip)
                        blocks[fault_part].back().clip_bits = std::bit_cast<uint32_t>(0.5f);
                    if (fault == SplitFault::LastUnit) blocks[fault_part].back().unit = -1.0f;
                    for (size_t part = 0; part < files.size(); ++part)
                        write(part, alignments[part], part == fault_part ? fault : SplitFault::None);
                    if (fault == SplitFault::Truncated)
                        std::filesystem::resize_file(files[fault_part], std::filesystem::file_size(files[fault_part]) - 1);
                }
                catch (...)
                {
                    remove();
                    throw;
                }
            }

            /** @brief Retire only this fixture's private directory, including construction failures. */
            ~SplitSourceFiles() { remove(); }
            SplitSourceFiles(const SplitSourceFiles &) = delete;
            SplitSourceFiles &operator=(const SplitSourceFiles &) = delete;

            /** @param part Original file ordinal. @return Its distinct original tensor identity. */
            static std::string name(size_t part) { return "blk." + std::to_string(part) + ".ffn_down.weight"; }

        private:
            /**
             * @brief Encode one fixed-width unsigned GGUF field in little-endian order.
             * @tparam T Exact unsigned on-disk integer type.
             * @param stream Owned fixture output. @param value Field value to serialize.
             */
            template <class T>
            static void scalar(std::ostream &stream, T value)
            {
                static_assert(std::is_unsigned_v<T>);
                for (size_t byte = 0; byte < sizeof(T); ++byte)
                    stream.put(static_cast<char>((value >> (byte * 8)) & T{0xff}));
            }

            /**
             * @brief Encode one length-prefixed GGUF string, without an implicit terminator.
             * @param stream Owned fixture output. @param value Exact string bytes.
             */
            static void string(std::ostream &stream, const std::string &value)
            {
                scalar(stream, uint64_t(value.size()));
                stream.write(value.data(), std::streamsize(value.size()));
            }

            /**
             * @brief Serialize one exact part; padding and directory lengths intentionally differ.
             * @param part Original file ordinal. @param declared_alignment Optional file-local alignment.
             * @param fault Optional malformed geometry or alignment metadata in this directory.
             * @throws std::ios_base::failure If the owned file cannot be completely written.
             */
            void write(size_t part, std::optional<uint32_t> declared_alignment, SplitFault fault)
            {
                static_assert(std::endian::native == std::endian::little,
                    "Native source-block fixture requires the supported little-endian host ABI");
                std::ofstream stream(files[part], std::ios::binary | std::ios::trunc);
                stream.exceptions(std::ios::badbit | std::ios::failbit);
                stream.write("GGUF", 4);
                scalar(stream, uint32_t{3});
                scalar(stream, uint64_t{1});
                const auto alignment = declared_alignment.value_or(32);
                scalar(stream, uint64_t{4} + uint64_t(declared_alignment.has_value()) +
                    uint64_t(fault == SplitFault::AlignmentDuplicate));
                string(stream, "general.architecture"); scalar(stream, uint32_t{8}); string(stream, "qwen35");
                for (const auto &[key, value] : std::array<std::pair<std::string, uint32_t>, 2>{
                    {{"split.count", 2}, {"split.no", uint32_t(part)}}})
                {
                    string(stream, key); scalar(stream, uint32_t{4}); scalar(stream, value);
                }
                if (declared_alignment)
                {
                    string(stream, "general.alignment");
                    scalar(stream, uint32_t(fault == SplitFault::AlignmentWrongType ? 6 : 4));
                    scalar(stream, fault == SplitFault::AlignmentZero ? uint32_t{0} :
                        fault == SplitFault::AlignmentNonPowerOfTwo ? uint32_t{48} : alignment);
                    if (fault == SplitFault::AlignmentDuplicate)
                    {
                        string(stream, "general.alignment"); scalar(stream, uint32_t{4}); scalar(stream, alignment);
                    }
                }
                string(stream, "general.name"); scalar(stream, uint32_t{8});
                string(stream, part == 0 ? "part-zero" : std::string(97, 'x'));
                string(stream, name(part));
                scalar(stream, uint32_t(fault == SplitFault::RankThree ? 3 : 2));
                scalar(stream, uint64_t(fault == SplitFault::PartialK ? kColumns - 1 : kColumns));
                scalar(stream, uint64_t{kRows});
                if (fault == SplitFault::RankThree) scalar(stream, uint64_t{1});
                scalar(stream, uint32_t(types[part] == TensorType::W4A4
                    ? GGUFTensorType::W4A4 : GGUFTensorType::W4A4H));
                scalar(stream, tensor_offsets[part]);
                const auto end = uint64_t(stream.tellp());
                data_offsets[part] = (end + alignment - 1) / alignment * alignment;
                stream.seekp(std::streamoff(data_offsets[part] + tensor_offsets[part]));
                stream.write(reinterpret_cast<const char *>(blocks[part].data()),
                    std::streamsize(blocks[part].size() * sizeof(SourceBlock)));
            }

            /** @brief Best-effort cleanup of the exact mkdtemp-owned root, never a parent directory. */
            void remove() noexcept
            {
                std::error_code ignored;
                std::filesystem::remove_all(directory, ignored);
            }
        };

        /** @return The sole final-buffer budget, sized by production preparation geometry. */
        std::shared_ptr<PhysicalMemoryAuthority> finalMemory()
        {
            const auto quote = CPUWeightPreparationMemory::sourceNative("W4A4", kRows, kColumns);
            PhysicalMemoryPlanBuilder budget;
            budget.add({0, DeviceId::cpu(), quote.persistent_bytes, quote.persistent_bytes}, kOwner, quote.persistent_bytes);
            return std::make_shared<PhysicalMemoryAuthority>(
                std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(budget.build()), 0);
        }

        /** @param part Original file ordinal. @return Its immutable source binding, not a second prepared identity. */
        WeightBinding binding(size_t part)
        {
            WeightBinding result;
            result.binding_id = part + 1;
            result.identity = makeSourceWeightIdentity(SplitSourceFiles::name(part), kModel, result.binding_id);
            result.residency.home_device = DeviceId::cpu();
            result.residency.resident_device = DeviceId::cpu();
            result.residency.raw_host_data_available = false;
            result.immutable = true;
            return result;
        }
    }

    /** @test Declared and absent alignments resolve independently before mapped or streamed publication. */
    TEST(GGUFAlignmentContract, EachFileOwnsItsDeclaredOrDefaultAlignment)
    {
        using Alignments = std::array<std::optional<uint32_t>, 2>;
        size_t publications = 0;
        for (auto first_type : {TensorType::W4A4, TensorType::W4A4H})
        for (const auto alignments : {Alignments{32u, 64u}, Alignments{64u, 32u},
                 Alignments{std::nullopt, 64u}, Alignments{64u, std::nullopt},
                 Alignments{std::nullopt, std::nullopt}})
        for (bool mapped : {false, true})
        {
            SCOPED_TRACE(::testing::Message() << tensorTypeName(first_type) << " mapped=" << mapped
                << " alignment=" << alignments[0].value_or(32) << "," << alignments[1].value_or(32));
            SplitSourceFiles files(first_type, alignments);
            ModelLoader loader;
            loader.setUseMmap(mapped);
            ASSERT_TRUE(loader.loadModel(files.files[0].string()));
            EXPECT_EQ(loader.getModel().alignment, alignments[0].value_or(32));
            ASSERT_EQ(loader.getModel().split_data_offsets.size(), 2u);
            auto memory = finalMemory();
            PreparedWeightStore store(kModel);
            store.installPhysicalMemoryAuthority(memory);
            for (size_t part = 0; part < files.files.size(); ++part)
            {
                EXPECT_EQ(loader.getModel().split_data_offsets[part], files.data_offsets[part]);
                const auto name = SplitSourceFiles::name(part);
                {
                    // Mapped loading returns a lease; streamed publication
                    // borrows the Store's sole owner until the following clear.
                    std::shared_ptr<TensorBase> mapped_tensor;
                    const TensorBase *tensor = nullptr;
                    if (mapped)
                    {
                        mapped_tensor = loader.loadTensorRowSlice(name, 3, 6);
                        tensor = mapped_tensor.get();
                    }
                    else
                    {
                        const NativeWeightSourceWindow window{.first_row = 3, .rows = 3, .columns = kColumns,
                            .native_row_bytes = 2 * sizeof(SourceBlock)};
                        const CPUPreparedWeightSource source{.original = loader.nativeWeightSourceGeometry(name),
                            .range = window, .consumer = CPUWeightConsumer::PreparedGemm};
                        auto writer = store.beginCPUPreparation(binding(part), source, kOwner, CPUStoragePlacement::local());
                        ASSERT_FALSE(writer->stagingRequest());
                        writer->readDirect(window, [&](std::span<std::byte> destination) {
                            loader.readNativeWeightSourceWindow(name, window, destination);
                        });
                        tensor = store.finishCPUPreparation(std::move(writer)).tensor;
                    }
                    ASSERT_NE(tensor, nullptr);
                    EXPECT_EQ(tensor->native_type(), files.types[part]);
                    EXPECT_EQ(tensor->size_bytes(), 6 * sizeof(SourceBlock));
                    EXPECT_EQ(std::memcmp(tensor->raw_data(), files.blocks[part].data() + 6,
                        6 * sizeof(SourceBlock)), 0);
                    ++publications;
                }
                store.clear();
                EXPECT_EQ(memory->claimedBytes(DeviceId::cpu(), kOwner,
                    PhysicalMemoryMaterializationKind::NewAllocation), 0u);
            }
        }
        EXPECT_EQ(publications, 40u);
        RecordProperty("file_local_alignment_publications", int(publications));
    }

    /** @test A bad or repeated alignment is rejected in either file, never replaced by 32. */
    TEST(GGUFAlignmentContract, InvalidDeclarationsFailBeforeSourcePublication)
    {
        size_t rejected = 0;
        for (auto first_type : {TensorType::W4A4, TensorType::W4A4H})
        for (size_t part : {size_t{0}, size_t{1}})
        for (auto fault : {SplitFault::AlignmentZero, SplitFault::AlignmentNonPowerOfTwo,
                 SplitFault::AlignmentWrongType, SplitFault::AlignmentDuplicate})
        for (bool mapped : {false, true})
        {
            SCOPED_TRACE(::testing::Message() << tensorTypeName(first_type) << " part=" << part
                << " fault=" << int(fault) << " mapped=" << mapped);
            SplitSourceFiles files(first_type, std::array<std::optional<uint32_t>, 2>{64u, 64u}, fault, part);
            ModelLoader loader;
            loader.setUseMmap(mapped);
            try
            {
                static_cast<void>(loader.loadModel(files.files[0].string()));
                FAIL() << "Invalid file-local alignment was admitted";
            }
            catch (const std::invalid_argument &error)
            {
                EXPECT_NE(std::string(error.what()).find("general.alignment"), std::string::npos);
                ++rejected;
            }
        }
        EXPECT_EQ(rejected, 32u);
        RecordProperty("rejected_alignment_declarations", int(rejected));
    }

    /** @test Both file mappings retain exact source coordinates until their own final engine retires. */
    TEST(W4A4ModelSourceSplitLifecycle, IndependentPartsRetainExactMappedBytesAndFinalReaders)
    {
        size_t checked_parts = 0;
        for (auto first_type : {TensorType::W4A4, TensorType::W4A4H})
        for (uint32_t alignment : {32u, 64u})
        for (size_t first_retired : {size_t{0}, size_t{1}})
        {
            SCOPED_TRACE(::testing::Message() << tensorTypeName(first_type) << " alignment=" << alignment
                << " first_retired=" << first_retired);
            SplitSourceFiles files(first_type, alignment);
            auto loader = std::make_unique<ModelLoader>();
            loader->setUseMmap(true);
            ASSERT_TRUE(loader->loadModel(files.files[0].string()));
            ASSERT_EQ(loader->getModel().tensors.size(), 2u);
            ASSERT_EQ(loader->getModel().split_data_offsets.size(), 2u);
            EXPECT_NE(files.data_offsets[0], files.data_offsets[1]);
            std::array<std::weak_ptr<MmapRegion>, 2> mappings;
            std::array<std::unique_ptr<ITensorGemm>, 2> engines;
            std::array<uint64_t, 2> selected_offsets{};
            for (size_t part = 0; part < 2; ++part)
            {
                const auto name = SplitSourceFiles::name(part);
                const auto *info = loader->getModel().findTensor(name);
                ASSERT_NE(info, nullptr);
                EXPECT_EQ(info->split_idx, part);
                EXPECT_EQ(info->size_bytes, files.blocks[part].size() * sizeof(SourceBlock));
                EXPECT_EQ(loader->getModel().split_data_offsets[part], files.data_offsets[part]);
                const auto geometry = loader->nativeWeightSourceGeometry(name);
                EXPECT_EQ(geometry.original_type, files.types[part]);
                EXPECT_EQ(geometry.original_shape, (std::vector<size_t>{kColumns, kRows}));
                mappings[part] = loader->getMmapRegion(info);
                auto selected = loader->loadTensorRowSlice(name, 3, 6);
                ASSERT_NE(selected, nullptr);
                EXPECT_EQ(selected->native_type(), files.types[part]);
                EXPECT_EQ(selected->shape(), (std::vector<size_t>{3, kColumns}));
                EXPECT_TRUE(selected->is_mmap_data());
                EXPECT_EQ(std::memcmp(selected->raw_data(), files.blocks[part].data() + 6, 6 * sizeof(SourceBlock)), 0);
                const auto provenance = MmapRegion::resolveFileSource(selected->raw_data(), selected->size_bytes());
                ASSERT_TRUE(provenance);
                selected_offsets[part] = files.data_offsets[part] + files.tensor_offsets[part] + 6 * sizeof(SourceBlock);
                EXPECT_EQ(provenance->path, files.files[part].string());
                EXPECT_EQ(provenance->offset, selected_offsets[part]);
                EXPECT_THROW(static_cast<void>(loader->loadTensorColumnSlice(name, 0, kBlockElements)), std::invalid_argument);
                engines[part] = selected->createGemm();
                ASSERT_NE(engines[part], nullptr);
                selected->release_raw_data();
                ++checked_parts;
            }
            loader.reset();
            for (size_t part = 0; part < 2; ++part)
            {
                ASSERT_TRUE(std::filesystem::remove(files.files[part]));
                EXPECT_FALSE(mappings[part].expired());
                const auto retained = mappings[part].lock();
                ASSERT_NE(retained, nullptr);
                EXPECT_EQ(std::memcmp(retained->data() + selected_offsets[part],
                    files.blocks[part].data() + 6, 6 * sizeof(SourceBlock)), 0);
                EXPECT_EQ(engines[part]->packedWeightBytes(), 6 * sizeof(SourceBlock));
            }
            engines[first_retired].reset();
            EXPECT_TRUE(mappings[first_retired].expired());
            EXPECT_FALSE(mappings[1 - first_retired].expired());
            engines[1 - first_retired].reset();
            EXPECT_TRUE(mappings[1 - first_retired].expired());
        }
        EXPECT_EQ(checked_parts, 16u);
        RecordProperty("checked_native_parts", int(checked_parts));
    }

    /** @test A bad unselected row or short tail in part one cannot publish, even after part zero is admitted. */
    TEST(W4A4ModelSourceSplitContract, MappedSelectionAdmitsTheEntireCorrectPart)
    {
        for (auto first_type : {TensorType::W4A4, TensorType::W4A4H})
        for (auto fault : {SplitFault::LastClip, SplitFault::LastUnit, SplitFault::Truncated})
        {
            SCOPED_TRACE(::testing::Message() << tensorTypeName(first_type) << " fault=" << int(fault));
            SplitSourceFiles files(first_type, 64, fault);
            ModelLoader loader;
            loader.setUseMmap(true);
            ASSERT_TRUE(loader.loadModel(files.files[0].string()));
            for (size_t part : {size_t{0}, size_t{1}, size_t{0}, size_t{1}})
            {
                const auto name = SplitSourceFiles::name(part);
                if (part == 1)
                    EXPECT_THROW(static_cast<void>(loader.loadTensorRowSlice(name, 3, 6)), std::invalid_argument);
                else
                {
                    const auto selected = loader.loadTensorRowSlice(name, 3, 6);
                    ASSERT_NE(selected, nullptr);
                    EXPECT_EQ(std::memcmp(selected->raw_data(), files.blocks[part].data() + 6, 6 * sizeof(SourceBlock)), 0);
                }
            }
        }
    }

    /** @test Complete/partial final writers keep admission, file selection and failure retirement independent per source. */
    TEST(W4A4ModelSourceSplitContract, StreamedPartsShareOnlySequentialFinalCapacity)
    {
        const NativeWeightSourceWindow whole{.rows = kRows, .columns = kColumns,
            .native_row_bytes = 2 * sizeof(SourceBlock)};
        const NativeWeightSourceWindow selected{.first_row = 3, .rows = 3, .columns = kColumns,
            .native_row_bytes = 2 * sizeof(SourceBlock)};
        size_t valid_publications = 0, rejected_publications = 0;
        for (auto first_type : {TensorType::W4A4, TensorType::W4A4H})
        for (uint32_t alignment : {32u, 64u})
        for (size_t first_part : {size_t{0}, size_t{1}})
        for (auto fault : {SplitFault::None, SplitFault::LastClip, SplitFault::LastUnit, SplitFault::Truncated})
        {
            SCOPED_TRACE(::testing::Message() << tensorTypeName(first_type) << " alignment=" << alignment
                << " first_part=" << first_part << " fault=" << int(fault));
            SplitSourceFiles files(first_type, alignment, fault);
            ModelLoader loader;
            loader.setUseMmap(false);
            ASSERT_TRUE(loader.loadModel(files.files[0].string()));
            auto memory = finalMemory();
            PreparedWeightStore store(kModel);
            store.installPhysicalMemoryAuthority(memory);
            for (size_t phase = 0; phase < 4; ++phase)
            {
                const size_t part = (first_part + phase) % 2;
                const auto name = SplitSourceFiles::name(part);
                const auto &window = phase == 0 || phase == 3 ? whole : selected;
                const auto publish = [&] {
                    const CPUPreparedWeightSource spec{.original = loader.nativeWeightSourceGeometry(name),
                        .range = window, .consumer = CPUWeightConsumer::PreparedGemm};
                    auto writer = store.beginCPUPreparation(binding(part), spec, kOwner, CPUStoragePlacement::local());
                    EXPECT_FALSE(writer->stagingRequest());
                    writer->readDirect(window, [&](std::span<std::byte> destination) {
                        loader.readNativeWeightSourceWindow(name, window, destination);
                    });
                    EXPECT_FALSE(store.preparedRefForBinding(part + 1, DeviceId::cpu()));
                    return store.finishCPUPreparation(std::move(writer));
                };
                if (part == 1 && fault != SplitFault::None)
                {
                    if (fault == SplitFault::Truncated)
                        EXPECT_THROW(static_cast<void>(publish()), std::runtime_error);
                    else
                        EXPECT_THROW(static_cast<void>(publish()), std::invalid_argument);
                    EXPECT_FALSE(store.preparedRefForBinding(part + 1, DeviceId::cpu()));
                    ++rejected_publications;
                }
                else
                {
                    const auto result = publish();
                    ASSERT_NE(result.tensor, nullptr);
                    ASSERT_TRUE(result.prepared);
                    EXPECT_FALSE(result.tensor->is_mmap_data());
                    EXPECT_EQ(result.tensor->native_type(), files.types[part]);
                    EXPECT_EQ(result.tensor->shape(), (std::vector<size_t>{window.rows, kColumns}));
                    EXPECT_EQ(std::memcmp(result.tensor->raw_data(), files.blocks[part].data() + 2 * window.first_row,
                        window.nativeBytes()), 0);
                    const auto quote = CPUWeightPreparationMemory::sourceNative(tensorTypeName(files.types[part]), window.rows, kColumns);
                    EXPECT_EQ(memory->claimedBytes(DeviceId::cpu(), kOwner, PhysicalMemoryMaterializationKind::NewAllocation),
                        quote.persistent_bytes);
                    store.clear();
                    ++valid_publications;
                }
                EXPECT_EQ(memory->claimedBytes(DeviceId::cpu(), kOwner, PhysicalMemoryMaterializationKind::NewAllocation), 0u);
                EXPECT_EQ(memory->claimedBytes(DeviceId::cpu(), PhysicalMemoryOwner::WeightLoadStaging,
                    PhysicalMemoryMaterializationKind::NewAllocation), 0u);
            }
        }
        EXPECT_EQ(valid_publications, 80u);
        EXPECT_EQ(rejected_publications, 48u);
        RecordProperty("valid_native_publications", int(valid_publications));
        RecordProperty("rejected_native_publications", int(rejected_publications));
    }

    /** @test Additional-part parsing enforces the same complete rank-two native geometry as part zero. */
    TEST(W4A4ModelSourceSplitContract, AdditionalPartCannotAdvertiseFragmentsOrExpertMatrices)
    {
        for (auto first_type : {TensorType::W4A4, TensorType::W4A4H})
        for (auto fault : {SplitFault::PartialK, SplitFault::RankThree})
        for (bool mapped : {false, true})
        {
            SCOPED_TRACE(::testing::Message() << tensorTypeName(first_type) << " fault=" << int(fault)
                << " mapped=" << mapped);
            SplitSourceFiles files(first_type, 32, fault);
            ModelLoader loader;
            loader.setUseMmap(mapped);
            EXPECT_THROW(static_cast<void>(loader.loadModel(files.files[0].string())), std::invalid_argument);
        }
    }
} // namespace llaminar2::w4a4
