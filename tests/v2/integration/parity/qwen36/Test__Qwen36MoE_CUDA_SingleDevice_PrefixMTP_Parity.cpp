#include "Qwen36MoEParityTestBase.h"

#include "backends/GPUDeviceContextPool.h"
#include "collective/BackendRouter.h"

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
        test_case.max_seq_len = 1024;
        return test_case;
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
