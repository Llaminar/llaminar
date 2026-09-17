/**
 * @file Test__PlanningModelSample.cpp
 * @brief Native GGUF slice identity and admitted payload lifetime regressions.
 *
 * Tiny temporary fixtures cover every GGUF weight format without preparing a
 * kernel or touching a GPU. Nonzero byte patterns expose wrong offsets, column
 * strides and expert selection. PMA assertions prove source aliases retain
 * their claim after the metadata owner retires and failed reads release both
 * source and temporary-reader claims. These are not numerical model proofs.
 */
#include "planning/PlanningModelMetadata.h"
#include "planning/PlanningExpertSample.h"
#include "planning/PlanningMatrixSample.h"
#include "planning/PhysicalMemoryAuthority.h"
#include "../../utils/PlanningGGUFFixture.h"
#include <gtest/gtest.h>
#include <cstring>
#include <limits>
#include <nlohmann/json.hpp>

using namespace llaminar2;
namespace { std::vector<GGUFTensorType> sourceFormats(); }

TEST(PlanningModelSample, ExpertSampleWirePreservesEveryNativeFormatWithoutFiles)
{
    // Publication uses the same codec on root and follower; it must not need
    // the source file merely to price a prepared sample's physical BOM.
    for (auto format : sourceFormats())
    {
        test::PlanningGGUFFixture fixture(true, false, format, 512);
        PlanningModelSource source(fixture.path());
        const PlanningExpertSampleRequest request{{"blk.0.ffn_gate_exps.weight", PlanningExpertMatrix{7}},
            {"blk.0.ffn_up_exps.weight", PlanningExpertMatrix{7}},
            {"blk.0.ffn_down_exps.weight", PlanningExpertMatrix{7}}};
        const auto plan = PlanningExpertSamplePlan::resolve(source, request);
        const auto bytes = plan.serialize();
        std::filesystem::remove(fixture.path());
        const auto received = PlanningExpertSamplePlan::deserialize(bytes);
        EXPECT_EQ(received.serialize(), bytes);
        EXPECT_EQ(received.description().matrices[0].n, 512u);
        EXPECT_EQ(received.description().matrices[2].n, 256u);
        EXPECT_EQ(received.tensorType(0), ggufToTensorType(format));
        EXPECT_THROW(received.tensorType(3), std::out_of_range);
        for (int defect = 0; defect < 10; ++defect)
        {
            auto malformed = nlohmann::json::parse(bytes);
            if (defect == 0) malformed["schema"] = "stale";
            if (defect == 1) malformed["model"] = "";
            if (defect == 2) malformed["layer"] = -1;
            if (defect == 3) malformed["matrices"][0]["type"] = 99999;
            if (defect == 4) malformed["matrices"][0]["bytes"] = 1;
            if (defect == 5) malformed["matrices"][0]["format"] = "not-the-native-type";
            if (defect == 6) malformed["matrices"][0]["n"] = 0;
            if (defect == 7) malformed["matrices"][0]["name"] = malformed["matrices"][1]["name"];
            if (defect == 8) malformed["expert"] = -1;
            if (defect == 9) malformed["matrices"][0]["k"] = std::numeric_limits<uint64_t>::max();
            const auto text = malformed.dump();
            const std::vector<uint8_t> invalid(text.begin(), text.end());
            EXPECT_THROW(PlanningExpertSamplePlan::deserialize(invalid), std::exception) << defect;
        }
    }
}

using namespace llaminar2;

namespace
{
    /** @return All native GGUF formats; Q8_1 is runtime-only in the source loader. */
    std::vector<GGUFTensorType> sourceFormats()
    {
        using T = GGUFTensorType;
        return {T::F32, T::F16, T::BF16, T::Q4_0, T::Q4_1, T::Q5_0, T::Q5_1, T::Q8_0,
            T::Q2_K, T::Q3_K, T::Q4_K, T::Q5_K, T::Q6_K, T::Q8_K, T::IQ1_S, T::IQ1_M,
            T::IQ2_XXS, T::IQ2_XS, T::IQ2_S, T::IQ3_XXS, T::IQ3_S, T::IQ4_NL, T::IQ4_XS};
    }

    /** @brief Fill only one fixture-owned extent before opening the sample's source. */
    std::vector<uint8_t> populate(const std::string &path, const std::string &name)
    {
        PlanningModelSource directory(path);
        const auto &model = directory.loader().getModel();
        const auto *info = model.findTensor(name);
        if (!info) throw std::runtime_error("Missing fixture matrix");
        std::vector<uint8_t> bytes(info->size_bytes);
        for (size_t index = 0; index != bytes.size(); ++index)
            bytes[index] = static_cast<uint8_t>((index * 37 + index / 17 + index / 65536) % 251);
        std::fstream stream(path, std::ios::binary | std::ios::in | std::ios::out);
        stream.exceptions(std::ios::badbit | std::ios::failbit);
        stream.seekp(model.data_offset + info->offset);
        stream.write(reinterpret_cast<const char *>(bytes.data()), bytes.size());
        return bytes;
    }

    /** @return A device-free PMA envelope admitting exactly the two read allocations. */
    std::shared_ptr<PhysicalMemoryAuthority> admit(size_t source_bytes, size_t staging_bytes)
    {
        const PhysicalMemoryResource resource{.world_rank = 0, .device = DeviceId::cpu(),
            .total_bytes = source_bytes + staging_bytes + 1, .admission_available_bytes = source_bytes + staging_bytes + 1};
        PhysicalMemoryPlanBuilder plan;
        plan.add(resource, PhysicalMemoryOwner::ModelSourcePayload, source_bytes);
        plan.add(resource, PhysicalMemoryOwner::WeightLoadStaging, staging_bytes);
        return std::make_shared<PhysicalMemoryAuthority>(
            std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(plan.build()), 0);
    }

    /** @return Current new allocation claims, never a second ledger. */
    size_t claimed(const std::shared_ptr<PhysicalMemoryAuthority> &memory, PhysicalMemoryOwner owner)
    {
        return memory->claimedBytes(DeviceId::cpu(), owner, PhysicalMemoryMaterializationKind::NewAllocation);
    }
}

TEST(PlanningModelSample, EveryNativeFormatReadsExactDenseRowsAndColumns)
{
    for (auto format : sourceFormats())
    {
        SCOPED_TRACE(static_cast<int>(format));
        test::PlanningGGUFFixture fixture(false, false, format);
        const std::string name = "blk.0.ffn_down.weight"; // [N=256,K=512].
        const auto expected = populate(fixture.path(), name);
        PlanningModelSource source(fixture.path());
        const size_t row_bytes = expected.size() / 256;
        const std::vector<PlanningModelSampleRequest> requests{
            {name, PlanningWholeMatrix{}}, {name, PlanningMatrixRows{3, 8}},
            {name, PlanningMatrixColumns{256, 512}}};
        for (size_t selection = 0; selection != requests.size(); ++selection)
        {
            const auto plan = PlanningMatrixSamplePublication::describe(nullptr, [&] {
                return PlanningMatrixSamplePlan::resolve(source, requests[selection]);
            });
            const auto received = PlanningMatrixSamplePlan::deserialize(plan.serialize());
            EXPECT_EQ(received.serialize(), plan.serialize());
            EXPECT_EQ(received.executionType(), received.tensorType());
            EXPECT_EQ(received.executionFormat(), received.format());
            EXPECT_EQ(received.executionPayloadBytes(), received.geometry().source_bytes);
            const auto geometry = received.geometry();
            auto memory = admit(geometry.source_bytes, geometry.source_bytes);
            std::optional sample(PlanningMatrixSamplePublication::publish(nullptr, received, memory,
                DeviceId::cpu(), [&] { return received.load(source, memory, DeviceId::cpu()); }));
            const auto &tensor = sample->tensor();
            EXPECT_EQ(tensor.shape(), (std::vector<size_t>{geometry.n, geometry.k}));
            EXPECT_EQ(claimed(memory, PhysicalMemoryOwner::ModelSourcePayload), geometry.source_bytes);
            EXPECT_EQ(claimed(memory, PhysicalMemoryOwner::WeightLoadStaging), 0u);
            const auto *actual = static_cast<const uint8_t *>(tensor.raw_data());
            if (selection < 2)
                EXPECT_EQ(std::memcmp(actual, expected.data() + (selection ? 3 * row_bytes : 0), geometry.source_bytes), 0);
            else
                for (size_t row = 0; row != 256; ++row)
                    EXPECT_EQ(std::memcmp(actual + row * row_bytes / 2,
                        expected.data() + row * row_bytes + row_bytes / 2, row_bytes / 2), 0);
            sample.reset();
            EXPECT_EQ(claimed(memory, PhysicalMemoryOwner::ModelSourcePayload), 0u);
        }
    }
}

TEST(PlanningModelSample, RuntimePromotionPreservesNativeWireIdentityForEverySourceFormat)
{
    for (auto format : sourceFormats())
    {
        SCOPED_TRACE(static_cast<int>(format));
        test::PlanningGGUFFixture fixture(false, false, format, 256, format);
        PlanningModelSource source(fixture.path());
        for (const auto name : {"blk.0.ssm_alpha.weight", "blk.0.ssm_beta.weight"})
        {
            const auto source_plan = PlanningMatrixSamplePlan::resolve(source, {name, PlanningMatrixRows{3, 8}});
            const auto received = PlanningMatrixSamplePlan::deserialize(source_plan.serialize());
            EXPECT_EQ(received.tensorType(), ggufToTensorType(format));
            EXPECT_EQ(received.serialize(), source_plan.serialize());
            EXPECT_EQ(received.executionType(), format == GGUFTensorType::Q8_0 ? TensorType::FP32 : received.tensorType());
            EXPECT_EQ(received.executionFormat(), format == GGUFTensorType::Q8_0 ? "F32" : received.format());
            EXPECT_EQ(received.executionPayloadBytes(), format == GGUFTensorType::Q8_0 ?
                5 * 256 * sizeof(float) : received.geometry().source_bytes);
            if (format == GGUFTensorType::Q8_0)
                EXPECT_GT(received.executionPayloadBytes(), received.geometry().source_bytes);
        }
    }
}

TEST(PlanningModelSample, EveryNativeExpertViewRetainsItsParentAndLeaseAfterSourceDestruction)
{
    for (auto format : sourceFormats())
    {
        SCOPED_TRACE(static_cast<int>(format));
        test::PlanningGGUFFixture fixture(true, false, format, 512);
        const std::string name = "blk.0.ffn_gate_exps.weight";
        const auto expected = populate(fixture.path(), name);
        const auto bytes_per_expert = expected.size() / 8;
        auto memory = admit(bytes_per_expert, bytes_per_expert);
        std::optional<PlanningLoadedMatrixSample> sample;
        {
            PlanningModelSource source(fixture.path());
            const auto plan = PlanningMatrixSamplePlan::resolve(source, {name, PlanningExpertMatrix{7}});
            sample = PlanningMatrixSamplePlan::deserialize(plan.serialize()).load(source, memory, DeviceId::cpu());
        }
        ASSERT_TRUE(sample.has_value());
        EXPECT_EQ(sample->tensor().shape(), (std::vector<size_t>{512, 256}));
        EXPECT_EQ(sample->tensor().size_bytes(), bytes_per_expert);
        EXPECT_EQ(claimed(memory, PhysicalMemoryOwner::ModelSourcePayload), bytes_per_expert);
        EXPECT_EQ(std::memcmp(sample->tensor().raw_data(), expected.data() + 7 * bytes_per_expert, bytes_per_expert), 0);
        auto borrower = sample;
        sample.reset();
        EXPECT_EQ(claimed(memory, PhysicalMemoryOwner::ModelSourcePayload), bytes_per_expert);
        borrower.reset();
        EXPECT_EQ(claimed(memory, PhysicalMemoryOwner::ModelSourcePayload), 0u);
        EXPECT_EQ(claimed(memory, PhysicalMemoryOwner::WeightLoadStaging), 0u);
    }
}

/** @test A shard's source coordinates, format and native extent are an indivisible identity. */
TEST(PlanningModelSample, MatrixWireRejectsMalformedOrChangedSourceBeforeAllocation)
{
    for (auto format : sourceFormats())
    {
        SCOPED_TRACE(static_cast<int>(format));
        test::PlanningGGUFFixture fixture(false, false, format);
        PlanningModelSource source(fixture.path());
        const auto plan = PlanningMatrixSamplePlan::resolve(source,
            {"blk.0.ffn_down.weight", PlanningMatrixColumns{256, 512}});
        const auto bytes = plan.serialize();
        auto memory = admit(plan.geometry().source_bytes, plan.geometry().source_bytes);
        for (int defect = 0; defect < 18; ++defect)
        {
            auto malformed = nlohmann::json::parse(bytes);
            if (defect == 0) malformed["schema"] = "stale";
            if (defect == 1) malformed["model"] = "";
            if (defect == 2) malformed["architecture"] = "";
            if (defect == 3) malformed["matrix"]["type"] = 99999;
            if (defect == 4) malformed["matrix"]["format"] = "wrong";
            if (defect == 5) malformed["matrix"]["bytes"] = 1;
            if (defect == 6) malformed["matrix"]["n"] = 0;
            if (defect == 7) malformed["matrix"]["k"] = std::numeric_limits<uint64_t>::max();
            if (defect == 8) malformed["selection"]["first"] = -1;
            if (defect == 9) malformed["selection"]["last"] = 256;
            if (defect == 10) malformed["selection"]["last"] = 511;
            if (defect == 11) malformed["selection"]["kind"] = "unknown";
            if (defect == 12) malformed["selection"]["extra"] = true;
            if (defect == 13) malformed["matrix"]["n"] = 256.5;
            if (defect == 14) malformed["matrix"]["bytes"] = -1;
            if (defect == 15) malformed["extra"] = true;
            if (defect == 16) malformed["selection"] = {{"kind", "expert"}, {"index", -1}};
            if (defect == 17) malformed["selection"] = {{"kind", "whole"}, {"first", 0}};
            const auto text = malformed.dump();
            EXPECT_THROW(PlanningMatrixSamplePlan::deserialize(std::vector<uint8_t>(text.begin(), text.end())),
                std::exception) << defect;
        }
        // A structurally valid receipt still cannot authorize a different file,
        // architecture or matrix than the one whose directory root validated.
        for (const std::string field : {"model", "architecture"})
        {
            auto changed = nlohmann::json::parse(bytes);
            changed[field] = "another-source";
            const auto text = changed.dump();
            const auto other = PlanningMatrixSamplePlan::deserialize(std::vector<uint8_t>(text.begin(), text.end()));
            EXPECT_THROW(other.load(source, memory, DeviceId::cpu()), std::invalid_argument);
            EXPECT_EQ(claimed(memory, PhysicalMemoryOwner::ModelSourcePayload), 0u);
        }
        std::filesystem::remove(fixture.path());
        EXPECT_EQ(PlanningMatrixSamplePlan::deserialize(bytes).serialize(), bytes);
    }
}

TEST(PlanningModelSample, InvalidSelectionAndAdmissionFailBeforeSourceReads)
{
    test::PlanningGGUFFixture fixture(true, false, GGUFTensorType::IQ4_XS);
    PlanningModelSource source(fixture.path());
    const std::string dense = "blk.0.attn_q.weight", expert = "blk.0.ffn_gate_exps.weight";
    for (const PlanningModelSampleRequest request : std::vector<PlanningModelSampleRequest>{
            {"absent", PlanningWholeMatrix{}}, {"output_norm.weight", PlanningWholeMatrix{}},
            {expert, PlanningWholeMatrix{}}, {dense, PlanningExpertMatrix{0}},
            {dense, PlanningMatrixRows{0, 0}}, {dense, PlanningMatrixRows{0, 257}},
            {dense, PlanningMatrixColumns{256, 257}}, {dense, PlanningMatrixColumns{9, 8}},
            {expert, PlanningExpertMatrix{8}}, {expert, PlanningExpertMatrix{SIZE_MAX}}})
        EXPECT_THROW((void)source.sampleGeometry(request), std::invalid_argument);
    const PlanningModelSampleRequest request{expert, PlanningExpertMatrix{3}};
    const auto geometry = source.sampleGeometry(request);
    auto memory = admit(geometry.source_bytes, geometry.source_bytes - 1);
    EXPECT_THROW((void)source.loadSample(request, memory, DeviceId::cpu()), std::exception);
    EXPECT_EQ(claimed(memory, PhysicalMemoryOwner::ModelSourcePayload), 0u);
    EXPECT_EQ(claimed(memory, PhysicalMemoryOwner::WeightLoadStaging), 0u);
    EXPECT_THROW((void)source.loadSample(request, nullptr, DeviceId::cpu()), std::invalid_argument);
    EXPECT_THROW((void)source.loadSample(request, memory, DeviceId::cuda(0)), std::invalid_argument);
}

TEST(PlanningModelSample, TruncatedPayloadReadReleasesBothClaims)
{
    test::PlanningGGUFFixture fixture;
    PlanningModelSource source(fixture.path());
    const PlanningModelSampleRequest request{"blk.1.ffn_down.weight", PlanningWholeMatrix{}};
    const auto geometry = source.sampleGeometry(request);
    auto memory = admit(geometry.source_bytes, geometry.source_bytes);
    std::filesystem::resize_file(fixture.path(), source.loader().getModel().data_offset);
    EXPECT_THROW((void)source.loadSample(request, memory, DeviceId::cpu()), std::runtime_error);
    EXPECT_EQ(claimed(memory, PhysicalMemoryOwner::ModelSourcePayload), 0u);
    EXPECT_EQ(claimed(memory, PhysicalMemoryOwner::WeightLoadStaging), 0u);
}

/** @test CPU/GPU samples share source identity and reject partial or mixed experts before I/O. */
TEST(PlanningModelSample, ExpertTripletDescriptionIsBackendIndependentAndMetadataOnly)
{
    for (auto format : sourceFormats())
    {
        SCOPED_TRACE(static_cast<int>(format));
        test::PlanningGGUFFixture fixture(true, false, format, 512);
        PlanningModelSource source(fixture.path());
        const PlanningExpertSampleRequest request{
            {"blk.0.ffn_gate_exps.weight", PlanningExpertMatrix{7}},
            {"blk.0.ffn_up_exps.weight", PlanningExpertMatrix{7}},
            {"blk.0.ffn_down_exps.weight", PlanningExpertMatrix{7}}};
        // After the directory has been retained, no source payload is needed
        // to declare either backend's exact full-expert preparation geometry.
        std::filesystem::resize_file(fixture.path(), source.loader().getModel().data_offset);
        const auto description = PlanningExpertSampleDescription::resolve(source, request);
        EXPECT_EQ(description.layer, 0);
        EXPECT_EQ(description.matrices[0].n, 512u);
        EXPECT_EQ(description.matrices[0].k, 256u);
        EXPECT_EQ(description.matrices[2].n, 256u);
        EXPECT_EQ(description.matrices[2].k, 512u);
        EXPECT_EQ(description.source_bytes, 3 * description.largest_source_bytes);
        for (const auto &matrix : description.matrices)
            EXPECT_EQ(matrix.source_bytes, description.largest_source_bytes);
        for (int invalid_case = 0; invalid_case < 5; ++invalid_case)
        {
            auto invalid = request;
            if (invalid_case == 0) invalid.up = invalid.gate;
            if (invalid_case == 1) invalid.up.selection = PlanningExpertMatrix{6};
            if (invalid_case == 2) invalid.up.selection = PlanningMatrixRows{0, 1};
            if (invalid_case == 3) invalid.up.tensor_name = "blk.1.ffn_up_exps.weight";
            if (invalid_case == 4) std::swap(invalid.gate, invalid.down);
            EXPECT_THROW(PlanningExpertSampleDescription::resolve(source, invalid), std::invalid_argument);
        }
    }
}
