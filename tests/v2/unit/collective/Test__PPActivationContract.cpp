/**
 * @file Test__PPActivationContract.cpp
 * @brief Unit tests for PPActivationContract and PPStageTransferContract
 *
 * Tests activation transfer contract construction, validation, byte
 * calculation, and transfer method selection for PP pipelines.
 */

#include <gtest/gtest.h>

#include "collective/PPActivationContract.h"
#include "backends/DeviceId.h"

using namespace llaminar2;

// =============================================================================
// PPStageTransferContract Tests
// =============================================================================

TEST(Test__PPActivationContract, TransferContractActiveBytes)
{
    PPStageTransferContract xfer;
    xfer.source_stage = 0;
    xfer.target_stage = 1;
    xfer.source_device = DeviceId::cuda(0);
    xfer.target_device = DeviceId::cuda(1);
    xfer.embedding_dim = 896;
    xfer.dtype = ActivationDType::FP32;
    xfer.max_seq_len = 2048;

    // FP32: 4 bytes per element
    EXPECT_EQ(xfer.activeBytes(1), 896u * 4u);
    EXPECT_EQ(xfer.activeBytes(128), 128u * 896u * 4u);
    EXPECT_EQ(xfer.maxBytes(), 2048u * 896u * 4u);
}

TEST(Test__PPActivationContract, TransferContractBF16Bytes)
{
    PPStageTransferContract xfer;
    xfer.source_stage = 0;
    xfer.target_stage = 1;
    xfer.source_device = DeviceId::cuda(0);
    xfer.target_device = DeviceId::cuda(1);
    xfer.embedding_dim = 4096;
    xfer.dtype = ActivationDType::BF16;
    xfer.max_seq_len = 1024;

    // BF16: 2 bytes per element
    EXPECT_EQ(xfer.activeBytes(1), 4096u * 2u);
    EXPECT_EQ(xfer.activeBytes(64), 64u * 4096u * 2u);
    EXPECT_EQ(xfer.maxBytes(), 1024u * 4096u * 2u);
}

TEST(Test__PPActivationContract, TransferContractSameDevice)
{
    PPStageTransferContract xfer;
    xfer.source_device = DeviceId::cuda(0);
    xfer.target_device = DeviceId::cuda(0);

    EXPECT_TRUE(xfer.isSameDevice());
    EXPECT_FALSE(xfer.isCrossVendor());
    EXPECT_EQ(xfer.selectMethod(), TransferMethod::NOOP);
}

TEST(Test__PPActivationContract, TransferContractSameVendor)
{
    PPStageTransferContract xfer;
    xfer.source_device = DeviceId::cuda(0);
    xfer.target_device = DeviceId::cuda(1);

    EXPECT_FALSE(xfer.isSameDevice());
    EXPECT_FALSE(xfer.isCrossVendor());
    EXPECT_EQ(xfer.selectMethod(), TransferMethod::DEVICE_TO_DEVICE_SAME_BACKEND);
}

TEST(Test__PPActivationContract, TransferContractCrossVendor)
{
    PPStageTransferContract xfer;
    xfer.source_device = DeviceId::cuda(0);
    xfer.target_device = DeviceId::rocm(0);

    EXPECT_FALSE(xfer.isSameDevice());
    EXPECT_TRUE(xfer.isCrossVendor());
    EXPECT_EQ(xfer.selectMethod(), TransferMethod::HOST_STAGED);
}

TEST(Test__PPActivationContract, TransferContractCPUInvolved)
{
    PPStageTransferContract xfer;
    xfer.source_device = DeviceId::cpu();
    xfer.target_device = DeviceId::cuda(0);

    EXPECT_TRUE(xfer.involvesCPU());
    EXPECT_EQ(xfer.selectMethod(), TransferMethod::HOST_STAGED);
}

TEST(Test__PPActivationContract, TransferContractDescribe)
{
    PPStageTransferContract xfer;
    xfer.source_stage = 0;
    xfer.target_stage = 1;
    xfer.source_device = DeviceId::cuda(0);
    xfer.target_device = DeviceId::rocm(0);
    xfer.embedding_dim = 896;
    xfer.dtype = ActivationDType::FP32;
    xfer.max_seq_len = 2048;

    auto desc = xfer.describe();
    EXPECT_NE(desc.find("Stage 0"), std::string::npos);
    EXPECT_NE(desc.find("896"), std::string::npos);
    EXPECT_NE(desc.find("FP32"), std::string::npos);
}

// =============================================================================
// PPActivationContract Tests
// =============================================================================

TEST(Test__PPActivationContract, EmptyContractValidates)
{
    PPActivationContract contract;
    EXPECT_TRUE(contract.validate().empty());
    EXPECT_EQ(contract.numTransfers(), 0u);
}

TEST(Test__PPActivationContract, SingleTransfer)
{
    PPActivationContract contract;

    PPStageTransferContract xfer;
    xfer.source_stage = 0;
    xfer.target_stage = 1;
    xfer.source_device = DeviceId::cuda(0);
    xfer.target_device = DeviceId::cuda(1);
    xfer.embedding_dim = 896;
    xfer.dtype = ActivationDType::FP32;
    xfer.max_seq_len = 2048;

    contract.addTransfer(xfer);

    EXPECT_EQ(contract.numTransfers(), 1u);
    EXPECT_TRUE(contract.validate().empty());

    const auto &t = contract.transfer(0);
    EXPECT_EQ(t.source_stage, 0);
    EXPECT_EQ(t.target_stage, 1);
    EXPECT_EQ(t.activeBytes(32), 32u * 896u * 4u);
}

TEST(Test__PPActivationContract, MultiStage)
{
    PPActivationContract contract;

    // Stage 0 → 1
    contract.addTransfer({
        .source_stage = 0,
        .target_stage = 1,
        .source_device = DeviceId::cuda(0),
        .target_device = DeviceId::cuda(1),
        .embedding_dim = 4096,
        .dtype = ActivationDType::FP32,
        .max_seq_len = 1024,
    });

    // Stage 1 → 2
    contract.addTransfer({
        .source_stage = 1,
        .target_stage = 2,
        .source_device = DeviceId::cuda(1),
        .target_device = DeviceId::rocm(0),
        .embedding_dim = 4096,
        .dtype = ActivationDType::FP32,
        .max_seq_len = 1024,
    });

    EXPECT_EQ(contract.numTransfers(), 2u);
    EXPECT_TRUE(contract.validate().empty());
    EXPECT_TRUE(contract.hasCrossVendorTransfer());

    EXPECT_EQ(contract.transfer(0).activeBytes(1), 4096u * 4u);
    EXPECT_EQ(contract.transfer(1).selectMethod(), TransferMethod::HOST_STAGED);
}

TEST(Test__PPActivationContract, TransferLookupThrowsOnMissing)
{
    PPActivationContract contract;

    contract.addTransfer({
        .source_stage = 0,
        .target_stage = 1,
        .source_device = DeviceId::cuda(0),
        .target_device = DeviceId::cuda(1),
        .embedding_dim = 896,
        .dtype = ActivationDType::FP32,
        .max_seq_len = 2048,
    });

    EXPECT_THROW(contract.transfer(5), std::runtime_error);
}

TEST(Test__PPActivationContract, ValidationFailsOnZeroEmbedding)
{
    PPActivationContract contract;

    contract.addTransfer({
        .source_stage = 0,
        .target_stage = 1,
        .source_device = DeviceId::cuda(0),
        .target_device = DeviceId::cuda(1),
        .embedding_dim = 0, // invalid
        .dtype = ActivationDType::FP32,
        .max_seq_len = 2048,
    });

    auto err = contract.validate();
    EXPECT_FALSE(err.empty());
    EXPECT_NE(err.find("embedding_dim"), std::string::npos);
}

TEST(Test__PPActivationContract, ValidationFailsOnZeroMaxSeqLen)
{
    PPActivationContract contract;

    contract.addTransfer({
        .source_stage = 0,
        .target_stage = 1,
        .source_device = DeviceId::cuda(0),
        .target_device = DeviceId::cuda(1),
        .embedding_dim = 896,
        .dtype = ActivationDType::FP32,
        .max_seq_len = 0, // invalid
    });

    auto err = contract.validate();
    EXPECT_FALSE(err.empty());
    EXPECT_NE(err.find("max_seq_len"), std::string::npos);
}

TEST(Test__PPActivationContract, NoCrossVendorWhenSameBackend)
{
    PPActivationContract contract;

    contract.addTransfer({
        .source_stage = 0,
        .target_stage = 1,
        .source_device = DeviceId::cuda(0),
        .target_device = DeviceId::cuda(1),
        .embedding_dim = 896,
        .dtype = ActivationDType::FP32,
        .max_seq_len = 2048,
    });

    EXPECT_FALSE(contract.hasCrossVendorTransfer());
}

TEST(Test__PPActivationContract, DTypeHelpers)
{
    EXPECT_EQ(activationDTypeBytes(ActivationDType::FP32), 4u);
    EXPECT_EQ(activationDTypeBytes(ActivationDType::BF16), 2u);
    EXPECT_EQ(activationDTypeBytes(ActivationDType::FP16), 2u);

    EXPECT_STREQ(activationDTypeName(ActivationDType::FP32), "FP32");
    EXPECT_STREQ(activationDTypeName(ActivationDType::BF16), "BF16");
    EXPECT_STREQ(activationDTypeName(ActivationDType::FP16), "FP16");
}
