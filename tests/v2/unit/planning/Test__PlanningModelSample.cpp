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
#include "../../utils/ScopedGDNProjectionArithmetic.h"
#include "planning/PlanningModelMetadata.h"
#include "planning/PlanningExpertSample.h"
#include "planning/PlanningMatrixSample.h"
#include "utils/DebugEnv.h"
#include "planning/PhysicalMemoryAuthority.h"
#include "../../utils/PlanningGGUFFixture.h"
#include "../../utils/Qwen4ExpPlanningGGUFFixture.h"
#include "../../utils/ServingGGUFFixturePayload.h"
#include <gtest/gtest.h>
#include <cstring>
#include <cmath>
#include <limits>
#include <nlohmann/json.hpp>

using namespace llaminar2;
namespace { std::vector<GGUFTensorType> sourceFormats(); }

/** @test Serving fixture source data is nondegenerate without relaxing verification. */
TEST(PlanningModelSample, FullAttentionServingPayloadIsFiniteNativeAndPreservesDirectory)
{
    for (const auto format : {GGUFTensorType::F32, GGUFTensorType::Q4_0})
    {
        test::PlanningGGUFFixture fixture(true, false, format, 256, std::nullopt, 8,
            test::PlanningGGUFFixture::Geometry::FullAttentionGraph, 3);
        ModelLoader before;
        before.setUseMmap(false);
        ASSERT_TRUE(before.loadModel(fixture.path()));
        const auto original = before.getModel();
        std::ifstream sparse(fixture.path(), std::ios::binary);
        const auto *embedding = original.findTensor("token_embd.weight");
        ASSERT_NE(embedding, nullptr);
        float zero = 1;
        sparse.seekg(original.data_offset + embedding->offset);
        sparse.read(reinterpret_cast<char *>(&zero), sizeof(zero));
        ASSERT_EQ(zero, 0.0F); // Old setup-only source is the negative control.
        test::populateServingGGUFFixture(fixture);
        ModelLoader after;
        after.setUseMmap(false);
        ASSERT_TRUE(after.loadModel(fixture.path()));
        const auto &model = after.getModel();
        ASSERT_EQ(model.tensors.size(), original.tensors.size());
        ASSERT_EQ(model.data_offset, original.data_offset);
        std::ifstream stream(fixture.path(), std::ios::binary);
        stream.exceptions(std::ios::badbit | std::ios::failbit);
        for (std::size_t index = 0; index < model.tensors.size(); ++index)
        {
            const auto &tensor = model.tensors[index];
            const auto &old = original.tensors[index];
            EXPECT_EQ(tensor.name, old.name);
            EXPECT_EQ(tensor.dimensions, old.dimensions);
            EXPECT_EQ(tensor.type, old.type);
            EXPECT_EQ(tensor.offset, old.offset);
            EXPECT_EQ(tensor.size_bytes, old.size_bytes);
            stream.seekg(model.data_offset + tensor.offset);
            if (tensor.type == GGUFTensorType::F32)
            {
                std::vector<float> values(tensor.size_bytes / sizeof(float));
                stream.read(reinterpret_cast<char *>(values.data()), tensor.size_bytes);
                ASSERT_FALSE(values.empty());
                ASSERT_TRUE(std::all_of(values.begin(), values.end(), [](float value) {
                    return std::isfinite(value) && value > 0.0F && value <= 1.0F;
                })) << tensor.name;
                if (tensor.name.find("norm.weight") != std::string::npos)
                    EXPECT_TRUE(std::all_of(values.begin(), values.end(), [](float value) { return value == 1.0F; }));
            }
            else
            {
                std::vector<Q4_0Block> blocks(tensor.size_bytes / sizeof(Q4_0Block));
                ASSERT_FALSE(blocks.empty());
                stream.read(reinterpret_cast<char *>(blocks.data()), tensor.size_bytes);
                ASSERT_TRUE(std::all_of(blocks.begin(), blocks.end(), [](const auto &block) {
                    const float scale = fp16_to_fp32(block.d);
                    return std::isfinite(scale) && scale > 0.0F &&
                        std::all_of(std::begin(block.qs), std::end(block.qs), [scale](auto byte) {
                            return scale * (int(byte & 15) - 8) > 0.0F &&
                                scale * (int(byte >> 4) - 8) > 0.0F;
                        });
                })) << tensor.name;
            }
        }
    }
    test::PlanningGGUFFixture unsupported(true, false, GGUFTensorType::Q5_0);
    EXPECT_THROW(test::populateServingGGUFFixture(unsupported), std::invalid_argument);
}

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
    const test::ScopedGDNProjectionArithmetic arithmetic(GDNProjectionArithmetic::Native);
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

/** @brief Router arithmetic survives publication independently of process environment. */
TEST(PlanningModelSample, RouterOperandArithmeticIsSealedInSampleIdentity)
{
    const test::ScopedGDNProjectionArithmetic arithmetic(GDNProjectionArithmetic::Native);
    test::PlanningGGUFFixture fixture(false, false, GGUFTensorType::Q6_K, 256, GGUFTensorType::Q6_K);
    PlanningModelSource source(fixture.path());
    const auto original = PlanningMatrixSamplePlan::resolve(source,
        {"blk.0.ssm_alpha.weight", PlanningMatrixRows{3, 8}});
    auto wire = nlohmann::json::parse(original.serialize());
    wire["matrix"]["name"] = "blk.0.ffn_gate_inp.weight";
    wire["gdn_arithmetic"] = "native";
    wire["router_arithmetic"] = "fp16";
    const auto decode = [](const auto &value) {
        const auto text = value.dump();
        return PlanningMatrixSamplePlan::deserialize(std::vector<uint8_t>(text.begin(), text.end()));
    };
    const auto half = decode(wire);
    EXPECT_EQ(half.executionType(), TensorType::FP16);
    EXPECT_EQ(half.operandPrecision(), FloatingActivationOperandPrecision::FP16RNE);
    EXPECT_EQ(half.executionPayloadBytes(), 5 * 256 * sizeof(uint16_t));
    EXPECT_EQ(half.serialize(), decode(nlohmann::json::parse(half.serialize())).serialize());
    wire["router_arithmetic"] = "native";
    const auto native = decode(wire);
    EXPECT_EQ(native.executionType(), TensorType::Q6_K);
    EXPECT_NE(native.serialize(), half.serialize());
    wire["router_arithmetic"] = "native_hadamard32";
    EXPECT_THROW(decode(wire), std::invalid_argument) << "Q6 router source must never be rotated and requantized";
    wire["router_arithmetic"] = "fp16";
    wire["matrix"]["name"] = "blk.0.ffn_gate_exps.weight";
    EXPECT_THROW(decode(wire), std::invalid_argument);
    wire["matrix"]["name"] = "blk.0.ffn_gate_inp.weight";
    for (const auto invalid : {"", "auto", "FP16", "fp32"})
    {
        wire["router_arithmetic"] = invalid;
        EXPECT_THROW(decode(wire), std::invalid_argument);
    }
    wire.erase("router_arithmetic");
    EXPECT_THROW(decode(wire), std::invalid_argument);
}

/** @brief Source format alone cannot alias a different floating activation program. */
TEST(PlanningModelSample, GDNOperandArithmeticIsSealedInSampleIdentity)
{
    struct Restore {
        GDNProjectionArithmetic previous = debugEnv().gemm.gdn_projection_arithmetic;
        ~Restore() { mutableDebugEnv().gemm.gdn_projection_arithmetic = previous; }
    } restore;
    mutableDebugEnv().gemm.gdn_projection_arithmetic = GDNProjectionArithmetic::Native;
    test::PlanningGGUFFixture fixture(false, false, GGUFTensorType::Q6_K, 256, GGUFTensorType::Q6_K);
    PlanningModelSource source(fixture.path());
    const auto original = PlanningMatrixSamplePlan::resolve(source,
        {"blk.0.ssm_alpha.weight", PlanningMatrixRows{3, 8}});
    auto wire = nlohmann::json::parse(original.serialize());
    wire["architecture"] = "qwen35moe";
    const auto decode = [](const auto &value) {
        const auto text = value.dump();
        return PlanningMatrixSamplePlan::deserialize(std::vector<uint8_t>(text.begin(), text.end()));
    };
    wire["gdn_arithmetic"] = "fp16";
    const auto half = decode(wire);
    EXPECT_EQ(half.executionType(), TensorType::FP16);
    EXPECT_EQ(half.operandPrecision(), FloatingActivationOperandPrecision::FP16RNE);
    EXPECT_EQ(half.executionPayloadBytes(), 5 * 256 * sizeof(uint16_t));
    wire["gdn_arithmetic"] = "fp32";
    const auto fp32 = decode(wire);
    EXPECT_EQ(fp32.executionType(), TensorType::FP32);
    EXPECT_EQ(fp32.operandPrecision(), FloatingActivationOperandPrecision::FP32);
    EXPECT_NE(half.serialize(), fp32.serialize());
    EXPECT_EQ(half.executionType(), TensorType::FP16);
    EXPECT_EQ(half.operandPrecision(), FloatingActivationOperandPrecision::FP16RNE);
    for (const auto mode : {"auto", "FP16", ""})
    {
        wire["gdn_arithmetic"] = mode;
        EXPECT_THROW(decode(wire), std::invalid_argument);
    }
    wire["gdn_arithmetic"] = "fp16_alpha_beta";
    EXPECT_THROW(decode(wire), std::invalid_argument); // Original Q6 controls cannot be converted.
    test::PlanningGGUFFixture controls(false, false, GGUFTensorType::Q6_K, 256, GGUFTensorType::F32);
    PlanningModelSource controls_source(controls.path());
    auto selective_wire = nlohmann::json::parse(PlanningMatrixSamplePlan::resolve(controls_source,
        {"blk.0.ssm_alpha.weight", PlanningMatrixRows{3, 8}}).serialize());
    selective_wire["architecture"] = "qwen35moe";
    selective_wire["gdn_arithmetic"] = "fp16_alpha_beta";
    const auto selective = decode(selective_wire);
    EXPECT_EQ(selective.executionType(), TensorType::FP16);
    EXPECT_EQ(selective.operandPrecision(), FloatingActivationOperandPrecision::FP16RNE);
    EXPECT_EQ(selective.executionPayloadBytes(), 5 * 256 * sizeof(uint16_t));
    EXPECT_EQ(nlohmann::json::parse(selective.serialize())["gdn_arithmetic"], "fp16_alpha_beta");
    EXPECT_EQ(decode(nlohmann::json::parse(selective.serialize())).serialize(), selective.serialize());
    selective_wire["gdn_arithmetic"] = "fp16_moe_alpha_beta";
    const auto moe_only = decode(selective_wire);
    EXPECT_EQ(moe_only.executionType(), TensorType::FP16);
    EXPECT_EQ(moe_only.operandPrecision(), FloatingActivationOperandPrecision::FP16RNE);
    EXPECT_NE(moe_only.serialize(), selective.serialize());
    EXPECT_EQ(decode(nlohmann::json::parse(moe_only.serialize())).serialize(), moe_only.serialize());
    selective_wire["architecture"] = "qwen35";
    const auto dense_native = decode(selective_wire);
    EXPECT_EQ(dense_native.executionType(), TensorType::FP32);
    EXPECT_EQ(dense_native.operandPrecision(), FloatingActivationOperandPrecision::Native);
    selective_wire["architecture"] = "qwen35moe";
    selective_wire["matrix"]["name"] = "blk.0.attn_qkv.weight";
    const auto wide = decode(selective_wire);
    EXPECT_EQ(wide.executionType(), TensorType::FP32);
    EXPECT_EQ(wide.operandPrecision(), FloatingActivationOperandPrecision::Native);
    wire["gdn_arithmetic"] = "fp16";
    wire["matrix"]["name"] = "blk.0.ffn_gate.weight";
    EXPECT_THROW(decode(wire), std::invalid_argument);
}

/** @brief Source resolution seals the concrete router mode before wire publication or later environment changes. */
TEST(PlanningModelSample, QFNRouterDefaultIsResolvedBeforeWirePublication)
{
    struct Restore final
    {
        MoERouterArithmetic previous = debugEnv().gemm.moe_router_arithmetic;
        /** @brief Restore the process selection after every assertion/unwind path. */
        ~Restore() { mutableDebugEnv().gemm.moe_router_arithmetic = previous; }
    } restore;
    test::Qwen4ExpPlanningGGUFFixture qfn_fixture;
    test::PlanningGGUFFixture other_fixture(true);
    PlanningModelSource qfn(qfn_fixture.path()), other(other_fixture.path());
    ASSERT_EQ(qfn.metadata().memoryProfile().architecture, "qwen4exp");
    ASSERT_EQ(other.metadata().memoryProfile().architecture, "qwen35moe");
    const PlanningModelSampleRequest request{"blk.0.ffn_gate_inp.weight", PlanningWholeMatrix{}};
    mutableDebugEnv().gemm.moe_router_arithmetic = MoERouterArithmetic::ModelDefault;
    const auto default_qfn = PlanningMatrixSamplePlan::resolve(qfn, request);
    const auto default_other = PlanningMatrixSamplePlan::resolve(other, request);
    const auto qfn_bytes = default_qfn.serialize();
    EXPECT_EQ(nlohmann::json::parse(qfn_bytes)["router_arithmetic"], "fp16");
    EXPECT_EQ(nlohmann::json::parse(default_other.serialize())["router_arithmetic"], "native");
    EXPECT_EQ(default_qfn.executionType(), TensorType::FP16);
    EXPECT_EQ(default_qfn.operandPrecision(), FloatingActivationOperandPrecision::FP16RNE);
    EXPECT_EQ(default_other.executionType(), TensorType::FP32);
    EXPECT_EQ(default_other.operandPrecision(), FloatingActivationOperandPrecision::Native);
    mutableDebugEnv().gemm.moe_router_arithmetic = MoERouterArithmetic::Native;
    const auto native_qfn = PlanningMatrixSamplePlan::resolve(qfn, request);
    EXPECT_EQ(nlohmann::json::parse(native_qfn.serialize())["router_arithmetic"], "native");
    EXPECT_EQ(native_qfn.executionType(), TensorType::FP32);
    EXPECT_NE(native_qfn.serialize(), qfn_bytes);
    EXPECT_EQ(PlanningMatrixSamplePlan::resolve(other, request).serialize(), default_other.serialize());
    const auto received = PlanningMatrixSamplePlan::deserialize(qfn_bytes);
    EXPECT_EQ(received.serialize(), qfn_bytes);
    EXPECT_EQ(received.executionType(), TensorType::FP16);
    EXPECT_EQ(received.operandPrecision(), FloatingActivationOperandPrecision::FP16RNE);
    mutableDebugEnv().gemm.moe_router_arithmetic = MoERouterArithmetic::FP16;
    EXPECT_EQ(PlanningMatrixSamplePlan::resolve(qfn, request).serialize(), qfn_bytes);
    auto unresolved = nlohmann::json::parse(qfn_bytes);
    unresolved["router_arithmetic"] = "model_default";
    const auto text = unresolved.dump();
    EXPECT_THROW(PlanningMatrixSamplePlan::deserialize(std::vector<uint8_t>(text.begin(), text.end())),
                 std::invalid_argument);
}
