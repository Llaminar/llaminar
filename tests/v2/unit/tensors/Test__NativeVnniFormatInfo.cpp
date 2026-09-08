/**
 * @file Test__NativeVnniFormatInfo.cpp
 * @brief Device-free totality tests for NativeVNNI source, execution, migration,
 *        and capture-envelope metadata.
 *
 * The catalog is the accounting authority shared by model loading, expert-slot
 * allocation, cross-tier repacking, and grouped-kernel graph capture. These
 * tests deliberately iterate the entire catalog so a newly supported source
 * format cannot omit one phase of that lifecycle.
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "tensors/Tensors.h"
#include "../../utils/TestTensorFactory.h"

using namespace llaminar2;
using namespace llaminar2::test;

namespace
{
    struct FormatExpectation
    {
        std::string name;
        uint8_t codebook_id = 0;
        int payload_bytes = 0;
        bool is_asymmetric = false;
        bool is_superblock = false;
        bool has_emins = false;
        std::function<std::unique_ptr<TensorBase>()> create;
    };

    const std::vector<FormatExpectation> kExpectations = {
        {"Q4_0", 0, 16, false, false, false,
         [] { return TestTensorFactory::createQ4_0Random({2, 256}); }},
        {"IQ4_NL", 4, 16, false, false, false,
         [] { return TestTensorFactory::createIQ4_NLRandom({2, 256}); }},
        {"IQ4_XS", 4, 16, false, true, false,
         [] { return TestTensorFactory::createIQ4_XSRandom({2, 256}); }},
        {"Q4_1", 5, 16, true, false, false,
         [] { return TestTensorFactory::createQ4_1Random({2, 256}); }},
        {"Q4_K", 5, 16, true, true, false,
         [] { return TestTensorFactory::createQ4_KRandom({2, 256}); }},
        {"Q5_0", 6, 20, false, false, false,
         [] { return TestTensorFactory::createQ5_0Random({2, 256}); }},
        {"Q5_1", 7, 20, true, false, false,
         [] { return TestTensorFactory::createQ5_1Random({2, 256}); }},
        {"Q5_K", 7, 20, true, true, false,
         [] { return TestTensorFactory::createQ5_KRandom({2, 256}); }},
        {"Q6_K", 8, 24, true, true, false,
         [] { return TestTensorFactory::createQ6_KRandom({2, 256}); }},
        {"Q3_K", 9, 12, true, true, false,
         [] { return TestTensorFactory::createQ3_KRandom({2, 256}); }},
        {"Q2_K", 10, 8, true, true, true,
         [] { return TestTensorFactory::createQ2_KRandom({2, 256}); }},
        {"IQ3_S", 11, 13, false, true, false,
         [] { return TestTensorFactory::createIQ3_SRandom({2, 256}); }},
        {"IQ3_XXS", 12, 12, false, true, false,
         [] { return TestTensorFactory::createIQ3_XXSRandom({2, 256}); }},
        {"IQ2_S", 13, 9, true, true, false,
         [] { return TestTensorFactory::createIQ2_SRandom({2, 256}); }},
        {"IQ2_XS", 14, 9, true, true, false,
         [] { return TestTensorFactory::createIQ2_XSRandom({2, 256}); }},
        {"IQ2_XXS", 15, 8, false, true, false,
         [] { return TestTensorFactory::createIQ2_XXSRandom({2, 256}); }},
        {"IQ1_S", 16, 6, true, true, false,
         [] { return TestTensorFactory::createIQ1_SRandom({2, 256}); }},
        {"IQ1_M", 17, 6, true, true, false,
         [] { return TestTensorFactory::createIQ1_MRandom({2, 256}); }},
        {"Q8_0", 19, 32, false, false, false,
         [] { return TestTensorFactory::createQ8_0Random({2, 256}); }},
        {"Q8_1", 20, 32, false, false, false,
         [] { return TestTensorFactory::createQ8_1Random({2, 256}); }},
        {"Q8_K", 21, 32, false, true, false,
         [] { return TestTensorFactory::createQ8_KRandom({2, 256}); }},
    };
}

TEST(Test__NativeVnniFormatInfo, SourceIdentitiesAreExhaustiveUniqueAndCanonical)
{
    ASSERT_EQ(native_vnni_formats::kAllSourceFormats.size(), 21u);
    ASSERT_EQ(native_vnni_formats::kAllSourceFormats.size(), kExpectations.size());

    for (size_t i = 0; i < native_vnni_formats::kAllSourceFormats.size(); ++i)
    {
        const NativeVnniSourceFormat &entry =
            native_vnni_formats::kAllSourceFormats[i];
        ASSERT_NE(entry.metadata, nullptr) << entry.quant_type;
        EXPECT_EQ(
            native_vnni_formats::forQuantType(entry.quant_type),
            entry.metadata)
            << entry.quant_type;
        EXPECT_EQ(
            native_vnni_formats::forSourceIdentity(
                entry.metadata->codebook_id,
                entry.metadata->is_superblock),
            entry.metadata)
            << entry.quant_type;

        const uint8_t expected_device_codebook =
            entry.metadata->codebook_id == 20 ||
                    entry.metadata->codebook_id == 21
                ? static_cast<uint8_t>(19)
                : entry.metadata->codebook_id;
        EXPECT_EQ(
            canonicalDeviceVnniCodebookId(entry.metadata->codebook_id),
            expected_device_codebook)
            << entry.quant_type;

        for (size_t j = i + 1;
             j < native_vnni_formats::kAllSourceFormats.size();
             ++j)
        {
            const NativeVnniSourceFormat &other =
                native_vnni_formats::kAllSourceFormats[j];
            ASSERT_NE(other.metadata, nullptr) << other.quant_type;
            EXPECT_FALSE(
                entry.metadata->codebook_id == other.metadata->codebook_id &&
                entry.metadata->is_superblock == other.metadata->is_superblock)
                << entry.quant_type << " and " << other.quant_type
                << " have an ambiguous migration identity";
        }
    }
}

TEST(Test__NativeVnniFormatInfo, TensorMetadataMatchesPerfSweepCodebookIds)
{
    for (const auto &expected : kExpectations)
    {
        auto tensor = expected.create();
        ASSERT_NE(tensor, nullptr) << expected.name;

        const auto *unpackable = dynamic_cast<const IINT8Unpackable *>(tensor.get());
        ASSERT_NE(unpackable, nullptr) << expected.name << " must expose IINT8Unpackable";

        const NativeVnniFormatInfo *info = unpackable->vnniFormatInfo();
        ASSERT_NE(info, nullptr) << expected.name << " must expose NativeVnniFormatInfo";

        EXPECT_EQ(info->codebook_id, expected.codebook_id) << expected.name;
        EXPECT_EQ(info->payload_bytes, expected.payload_bytes) << expected.name;
        EXPECT_EQ(info->is_asymmetric, expected.is_asymmetric) << expected.name;
        EXPECT_EQ(info->is_superblock, expected.is_superblock) << expected.name;
        EXPECT_EQ(info->has_emins, expected.has_emins) << expected.name;
        EXPECT_EQ(
            info,
            native_vnni_formats::forQuantType(expected.name))
            << expected.name
            << " tensor metadata and planner catalog must be one object";
    }
}

TEST(Test__NativeVnniFormatInfo, PackedRegionSizingMatchesCanonicalMetadata)
{
    constexpr size_t rows = 17;
    constexpr size_t columns = 256;

    for (const auto &expected : kExpectations)
    {
        SCOPED_TRACE(expected.name);
        const NativeVnniFormatInfo *format =
            native_vnni_formats::forQuantType(expected.name);
        ASSERT_NE(format, nullptr);

        const NativeVnniPackedRegionSizes regions =
            nativeVnniPackedRegionSizes(rows, columns, *format);
        const size_t blocks = rows * columns / 32;
        EXPECT_EQ(
            regions.payload_bytes,
            blocks * static_cast<size_t>(expected.payload_bytes));
        EXPECT_EQ(regions.scales_bytes, blocks * sizeof(uint16_t));
        EXPECT_EQ(
            regions.mins_bytes,
            expected.is_asymmetric
                ? blocks * sizeof(uint16_t)
                : 0);
        EXPECT_EQ(
            regions.emins_bytes,
            expected.has_emins
                ? blocks * sizeof(uint32_t)
                : 0);
    }
}

TEST(Test__NativeVnniFormatInfo, MigrationStableAndReusableFormatsCoverEveryCodebook)
{
    for (const auto &expected : kExpectations)
    {
        SCOPED_TRACE(expected.name);
        const NativeVnniFormatInfo *source =
            native_vnni_formats::forQuantType(expected.name);
        ASSERT_NE(source, nullptr);

        const auto migrated = migrationStableDeviceVnniFormat(*source);
        const int expected_migrated_payload =
            source->codebook_id == 8
                ? 24
                : (source->codebook_id == 0 ||
                           source->codebook_id == 4 ||
                           source->codebook_id == 5
                       ? 16
                       : 32);
        const uint8_t expected_migrated_codebook =
            expected_migrated_payload < 32
                ? canonicalDeviceVnniCodebookId(source->codebook_id)
                : (source->is_asymmetric
                       ? kNativeVnniExpandedInt8MinCodebook
                       : static_cast<uint8_t>(19));
        EXPECT_EQ(
            migrated.payload_bytes_per_block,
            expected_migrated_payload);
        EXPECT_EQ(migrated.codebook_id, expected_migrated_codebook);
        EXPECT_EQ(migrated.is_asymmetric, source->is_asymmetric);
        EXPECT_FALSE(migrated.has_emins);
        EXPECT_TRUE(deviceVnniExecutionCompatibleWithSource(
            *source, migrated.codebook_id));

        const auto allocation =
            reusableDeviceVnniAllocationFormat(*source);
        EXPECT_EQ(
            allocation.payload_bytes_per_block,
            std::max(source->payload_bytes, expected_migrated_payload));
        EXPECT_EQ(allocation.has_mins, source->is_asymmetric);
        EXPECT_EQ(allocation.has_emins, source->has_emins);
        EXPECT_GE(
            allocation.payload_bytes_per_block,
            source->payload_bytes);
        EXPECT_GE(
            allocation.payload_bytes_per_block,
            migrated.payload_bytes_per_block);
    }
}

TEST(Test__NativeVnniFormatInfo, RuntimeCaptureEnvelopeCoversEveryTierRepresentation)
{
    for (const NativeVnniSourceFormat &entry :
         native_vnni_formats::kAllSourceFormats)
    {
        SCOPED_TRACE(entry.quant_type);
        ASSERT_NE(entry.metadata, nullptr);

        const NativeVnniFormatInfo &source = *entry.metadata;
        const uint8_t canonical_codebook =
            canonicalDeviceVnniCodebookId(source.codebook_id);
        const uint8_t migration_codebook =
            migrationStableDeviceVnniFormat(source).codebook_id;
        const NativeVnniSourceIdentity source_identity{
            .codebook_id = source.codebook_id,
            .is_superblock = source.is_superblock,
            .present = true,
        };
        const uint32_t expected_policy_mask =
            nativeVnniCodebookMaskBit(canonical_codebook);
        const uint32_t expected_runtime_mask =
            expected_policy_mask |
            nativeVnniCodebookMaskBit(migration_codebook);

        NativeVnniExecutionFormatEnvelope immutable_envelope;
        ASSERT_TRUE(addNativeVnniExecutionFormat(
            immutable_envelope,
            canonical_codebook,
            source_identity,
            false));
        EXPECT_EQ(
            immutable_envelope.execution_codebook_mask,
            expected_policy_mask);
        EXPECT_EQ(
            immutable_envelope.policy_codebook_mask,
            expected_policy_mask);

        NativeVnniExecutionFormatEnvelope compact_runtime_envelope;
        ASSERT_TRUE(addNativeVnniExecutionFormat(
            compact_runtime_envelope,
            canonical_codebook,
            source_identity,
            true));
        EXPECT_EQ(
            compact_runtime_envelope.execution_codebook_mask,
            expected_runtime_mask);
        EXPECT_EQ(
            compact_runtime_envelope.policy_codebook_mask,
            expected_policy_mask);

        // Publication may first observe either physical representation. Both
        // must authenticate the same capture identity so tier movement never
        // requires graph recapture or silently loses a decoder launch.
        NativeVnniExecutionFormatEnvelope migrated_runtime_envelope;
        ASSERT_TRUE(addNativeVnniExecutionFormat(
            migrated_runtime_envelope,
            migration_codebook,
            source_identity,
            true));
        EXPECT_EQ(
            migrated_runtime_envelope.execution_codebook_mask,
            expected_runtime_mask);
        EXPECT_EQ(
            migrated_runtime_envelope.policy_codebook_mask,
            expected_policy_mask);
        EXPECT_EQ(
            migrated_runtime_envelope.execution_codebook_mask,
            compact_runtime_envelope.execution_codebook_mask);
        EXPECT_EQ(
            migrated_runtime_envelope.policy_codebook_mask,
            compact_runtime_envelope.policy_codebook_mask);
    }
}

TEST(Test__NativeVnniFormatInfo, MutableEnvelopeRequiresAuthenticatedSourceIdentity)
{
    NativeVnniExecutionFormatEnvelope envelope;
    EXPECT_FALSE(addNativeVnniExecutionFormat(
        envelope,
        native_vnni_formats::Q5_K.codebook_id,
        NativeVnniSourceIdentity{},
        true));
    EXPECT_FALSE(envelope.valid());

    EXPECT_TRUE(addNativeVnniExecutionFormat(
        envelope,
        native_vnni_formats::Q5_K.codebook_id,
        NativeVnniSourceIdentity{},
        false));
    EXPECT_TRUE(envelope.valid());
}
