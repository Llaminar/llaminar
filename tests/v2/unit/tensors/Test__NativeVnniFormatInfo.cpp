#include <gtest/gtest.h>

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
    };
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
    }
}
