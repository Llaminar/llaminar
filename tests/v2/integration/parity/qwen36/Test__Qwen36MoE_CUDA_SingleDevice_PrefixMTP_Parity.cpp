#include "Qwen36MoEParityTestBase.h"

#include "backends/GPUDeviceContextPool.h"
#include "collective/BackendRouter.h"

#include <algorithm>
#include <unistd.h>

using namespace llaminar2;
using namespace llaminar2::test::parity::qwen36;

namespace
{
    MoEPrefixRestoreParityCase cudaSingleDeviceCase()
    {
        auto test_case = qwen36MoEPrefixParityCase(
            "Qwen3.6 MoE CUDA SingleDevice parity",
            MoEPrefixParityTopology::SingleDevice);
        test_case.devices = {GlobalDeviceAddress::cuda(0)};
        test_case.required_cuda_devices = 1;
        test_case.required_rocm_devices = 0;
        // The metadata fixture's third greedy token is a near-tie on CUDA MoE
        // (760 vs 71093) across fresh model loads. Keep exact prefix/MTP
        // restore checks on the stable first two tokens; the dedicated CUDA
        // math parity suite remains responsible for layer/logit tolerances.
        test_case.decode_steps = 2;
        return test_case;
    }

    std::string cudaMoEBenchmarkPrompt()
    {
        return "The following is a comprehensive analysis of machine learning systems "
               "and their applications in modern computing environments. "
               "We will explore the fundamental concepts, examine practical implementations, "
               "and discuss the future directions of this rapidly evolving field. "
               "Machine learning has transformed how we approach problem-solving across "
               "numerous domains, from natural language processing to computer vision, "
               "from autonomous vehicles to medical diagnosis. "
               "The key to understanding these systems lies in grasping the underlying "
               "mathematical foundations while also appreciating the engineering challenges "
               "involved in deploying them at scale. "
               "Let us begin our exploration with an overview of the main paradigms: "
               "supervised learning, unsupervised learning, and reinforcement learning. "
               "Each of these approaches has its own strengths and is suited to different "
               "types of problems. In supervised learning, we train models using labeled data, "
               "where the correct output is known for each input example. "
               "This approach is particularly effective for classification and regression tasks. "
               "Unsupervised learning, on the other hand, deals with finding patterns in data "
               "without explicit labels. Clustering, dimensionality reduction, and anomaly detection "
               "are common applications. Reinforcement learning takes a different approach, "
               "where agents learn optimal behaviors through interaction with an environment, "
               "receiving rewards or penalties based on their actions. "
               "Deep learning, a subset of machine learning, has revolutionized the field "
               "by enabling the training of neural networks with many layers. "
               "These deep neural networks can learn hierarchical representations of data, "
               "automatically extracting features at multiple levels of abstraction. "
               "Convolutional neural networks have become the standard for image processing, "
               "while recurrent neural networks and transformers excel at sequential data. "
               "The transformer architecture, introduced in 2017, has become particularly influential, "
               "forming the basis for large language models like GPT, BERT, and LLaMA. "
               "These models are trained on vast amounts of text data and can perform "
               "a wide range of natural language tasks with impressive accuracy. "
               "The training process involves optimizing millions or billions of parameters "
               "using gradient descent and backpropagation algorithms. "
               "Modern training infrastructure relies on specialized hardware like GPUs and TPUs, "
               "distributed computing frameworks, and sophisticated optimization techniques. "
               "Transfer learning has emerged as a powerful paradigm, allowing models "
               "pre-trained on large datasets to be fine-tuned for specific tasks "
               "with relatively little additional data. This approach has democratized "
               "access to state-of-the-art AI capabilities for researchers and practitioners "
               "who may not have the resources to train large models from scratch. "
               "As we look to the future, several exciting developments are on the horizon. "
               "Multimodal models that can process text, images, audio, and video together "
               "are becoming increasingly sophisticated. Federated learning enables "
               "training on distributed data while preserving privacy. "
               "Neural architecture search automates the design of optimal network structures. "
               "And new hardware accelerators promise to make AI more efficient and accessible. "
               "The ethical implications of these technologies cannot be overlooked. "
               "Issues of bias, fairness, transparency, and accountability must be addressed "
               "as AI systems become more prevalent in society. Responsible AI development "
               "requires collaboration between technologists, policymakers, and the public "
               "to ensure these powerful tools benefit humanity as a whole.";
    }

    MoEPrefixRestoreParityCase cudaSingleDeviceBenchmarkPromptCase()
    {
        auto test_case = cudaSingleDeviceCase();
        test_case.name = "Qwen3.6 MoE CUDA SingleDevice benchmark-prompt MTP diagnostic";
        test_case.prompt = cudaMoEBenchmarkPrompt();
        test_case.metadata_envs = {"LLAMINAR_QWEN36_MOE_CUDA_MTP_DIAGNOSTIC_METADATA"};
        test_case.default_metadata_path =
            "pytorch_qwen36_moe_cuda_mtp_diagnostic_snapshots/metadata.txt";
        test_case.decode_steps = 4;
        test_case.max_seq_len = 768;
        return test_case;
    }

    MoEPrefixRestoreParityCase cudaSingleDeviceDepth3Case()
    {
        auto test_case = cudaSingleDeviceBenchmarkPromptCase();
        test_case.name = "Qwen3.6 MoE CUDA SingleDevice depth-3 MTP parity";
        test_case.decode_steps = 4;
        return test_case;
    }

    void expectCudaMoESharedExpertGroupedDecodePath()
    {
        const auto records = PerfStatsCollector::snapshot(
            {"kernel.cuda_moe_grouped_decode_gateup_calls",
             "kernel.cuda_moe_grouped_decode_down_calls"});
        auto tag_equals = [](const PerfStatRecord &record,
                             const char *key,
                             const char *value) -> bool
        {
            const auto it = record.tags.find(key);
            return it != record.tags.end() && it->second == value;
        };
        auto has_shared_decode_record = [&](const char *name) -> bool
        {
            return std::find_if(
                       records.begin(),
                       records.end(),
                       [&](const PerfStatRecord &record)
                       {
                           return record.name == name &&
                                  tag_equals(record, "source", "table") &&
                                  (tag_equals(record, "route", "serial") ||
                                   tag_equals(record, "route", "kpart")) &&
                                  tag_equals(record, "active_slots", "1") &&
                                  tag_equals(record, "d_model", "2048") &&
                                  tag_equals(record, "intermediate", "512");
                       }) != records.end();
        };

        ASSERT_TRUE(has_shared_decode_record("cuda_moe_grouped_decode_gateup_calls"))
            << "CUDA shared expert decode must use the grouped table gate/up path.\n"
            << PerfStatsCollector::summaryString(
                   {"kernel.cuda_moe_grouped_decode_gateup_calls",
                    "kernel.cuda_moe_grouped_decode_down_calls"});
        ASSERT_TRUE(has_shared_decode_record("cuda_moe_grouped_decode_down_calls"))
            << "CUDA shared expert decode must use the grouped table down path.\n"
            << PerfStatsCollector::summaryString(
                   {"kernel.cuda_moe_grouped_decode_gateup_calls",
                    "kernel.cuda_moe_grouped_decode_down_calls"});
    }

} // namespace

TEST(Qwen36MoECUDASingleDevicePrefixMTPParity, PrefixRestoreFullHit)
{
    runMoEPrefixRestoreParity(cudaSingleDeviceCase(), PrefixRestoreParityMode::FullHit);
}

TEST(Qwen36MoECUDASingleDevicePrefixMTPParity, PrefixRestorePartialHit)
{
    runMoEPrefixRestoreParity(cudaSingleDeviceCase(), PrefixRestoreParityMode::PartialHit);
}

TEST(Qwen36MoECUDASingleDevicePrefixMTPParity, IncrementalDecodeMatchesFullContext)
{
    runMoEIncrementalDecodeMatchesFullContext(cudaSingleDeviceCase());
}

TEST(Qwen36MoECUDASingleDevicePrefixMTPParity, GreedyFreshRunnerDeterminism)
{
    runMoEGreedyFreshRunnerDeterminism(cudaSingleDeviceCase());
}

TEST(Qwen36MoECUDASingleDevicePrefixMTPParity, MTPGreedyMatchesPyTorchDecodeTokens)
{
    runMoEMTPParity(cudaSingleDeviceCase(), false);
}

TEST(Qwen36MoECUDASingleDevicePrefixMTPParity, PrefixCacheMTPRestore)
{
    runMoEMTPParity(cudaSingleDeviceCase(), true);
}

TEST(Qwen36MoECUDASingleDevicePrefixMTPParity, MTPBenchmarkPromptSidecarStageBreakdown)
{
    runMoEMTPSidecarStageBreakdownDiagnostic(
        cudaSingleDeviceBenchmarkPromptCase(),
        4);
}

TEST(Qwen36MoECUDASingleDevicePrefixMTPParity, MTPBenchmarkStyleSkipGatherGreedyMatchesReference)
{
    runMoEMTPBenchmarkStyleSkipGatherParity(
        cudaSingleDeviceBenchmarkPromptCase(),
        4);
}

TEST(Qwen36MoECUDASingleDevicePrefixMTPParity, MTPBenchmarkStyleDepth1EightTokensMatchesReference)
{
    ScopedEnvironmentValues replay_check({
        {"LLAMINAR_MTP_VERIFY_COMMIT_REPLAY_CHECK", "1"},
        {"LLAMINAR_MTP_VERIFY_COMMIT_REPLAY_DEPTH", "4"},
    });
    runMoEMTPBenchmarkStyleSkipGatherParity(
        cudaSingleDeviceBenchmarkPromptCase(),
        8,
        1,
        {},
        true);
}

TEST(Qwen36MoECUDASingleDevicePrefixMTPParity, MTPBudgetLimitedOneTokenStepsMatchReference)
{
    runMoEMTPBudgetOneStepMatchesReference(
        cudaSingleDeviceBenchmarkPromptCase(),
        4);
}

TEST(Qwen36MoECUDASingleDevicePrefixMTPParity, MTPBenchmarkStyleUsesFusedVerifierPrefillPath)
{
    ScopedEnvironmentValues perf_stats_enabled({
        {"LLAMINAR_PERF_STATS_SUMMARY", "1"},
    });
    PerfStatsCollector::reset();
    runMoEMTPBenchmarkStyleSkipGatherParity(
        cudaSingleDeviceBenchmarkPromptCase(),
        4);
    expectCudaMoEMTPVerifierGDNProjectionFusedPath();
    expectCudaMoEMTPVerifierFusedPrefillPath();
    expectCudaMoEMTPVerifierSharedExpertFusedPrefillPath();
    PerfStatsCollector::reset();
}

TEST(Qwen36MoECUDASingleDevicePrefixMTPParity, MTPBenchmarkStyleDynamicDepthRequestStateResets)
{
    runMoEMTPDynamicDepthRequestStateResetBenchmarkStyle(
        cudaSingleDeviceBenchmarkPromptCase(),
        16);
}

TEST(Qwen36MoECUDASingleDevicePrefixMTPParity, MTPGreedyDepth3MatchesBaselineTokens)
{
    runMoEMTPParity(cudaSingleDeviceDepth3Case(), false, 3);
}

TEST(Qwen36MoECUDASingleDevicePrefixMTPParity, MTPBenchmarkStyleDepth3LongPromptGreedyMatchesReference)
{
    auto test_case = cudaSingleDeviceDepth3Case();
    test_case.max_seq_len = 4096;
    runMoEMTPBenchmarkStyleSkipGatherParity(
        test_case,
        16,
        3,
        {},
        true);
}

TEST(Qwen36MoECUDASingleDevicePrefixMTPParity, NoMTPBenchmarkStyleSkipGatherGreedyMatchesGatheredArgmax)
{
    ScopedEnvironmentValues perf_stats_enabled({
        {"LLAMINAR_PERF_STATS_SUMMARY", "1"},
    });
    PerfStatsCollector::reset();
    runMoENoMTPBenchmarkStyleSkipGatherArgmaxParity(
        cudaSingleDeviceBenchmarkPromptCase(),
        16);
    expectCudaMoESharedExpertGroupedDecodePath();
    PerfStatsCollector::reset();
}

TEST(Qwen36MoECUDASingleDevicePrefixMTPParity, VerifierRowShortcutTwoRowStateMatchesFullReplay)
{
    ScopedEnvironmentValues perf_stats_enabled({
        {"LLAMINAR_PERF_STATS_SUMMARY", "1"},
    });
    PerfStatsCollector::reset();
    runMoEMTPVerifierRowShortcutEquivalence(cudaSingleDeviceBenchmarkPromptCase());
    PerfStatsCollector::reset();
}

int main(int argc, char **argv)
{
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();

    GlobalBackendRouter::shutdown();
    GPUDeviceContextPool::instance().shutdown();

    MPI_Finalize();
    std::cout.flush();
    std::cerr.flush();
    _exit(result);
}
