#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "app/AppContext.h"
#include "app/modes/CompletionMode.h"
#include "app/modes/SingleShotChatMode.h"
#include "mocks/MockMPIContext.h"
#include "mocks/MockOrchestrationRunner.h"
#include "mocks/MockTokenizer.h"
#include "utils/Logger.h"

#include <mpi.h>
#include <memory>
#include <stdexcept>

using namespace llaminar2;
using namespace llaminar2::test;
using ::testing::_;
using ::testing::ElementsAre;
using ::testing::HasSubstr;
using ::testing::NiceMock;
using ::testing::Not;
using ::testing::Return;
using ::testing::Throw;

namespace
{
    GenerationResult tokenResult(int32_t token, bool complete = false)
    {
        GenerationResult result;
        result.tokens = {token};
        result.is_complete = complete;
        return result;
    }

    GenerationResult tokenResult(std::initializer_list<int32_t> tokens, bool complete = false)
    {
        GenerationResult result;
        result.tokens.assign(tokens.begin(), tokens.end());
        result.is_complete = complete;
        return result;
    }

    GenerationResult errorResult(const std::string &error)
    {
        GenerationResult result;
        result.error = error;
        return result;
    }

    struct ModeHarness
    {
        explicit ModeHarness(int rank, int world_size)
        {
            auto owned_runner = std::make_unique<NiceMock<MockOrchestrationRunner>>();
            runner = owned_runner.get();
            tokenizer = std::make_shared<NiceMock<MockTokenizer>>();
            mpi = std::make_shared<MockMPIContext>(rank, world_size);

            ctx.mpi_ctx = mpi;
            ctx.runner = std::move(owned_runner);
            ctx.tokenizer = tokenizer;
            ctx.config.prompt = "Hello";
            ctx.config.n_predict = 1;
        }

        AppContext ctx;
        NiceMock<MockOrchestrationRunner> *runner = nullptr;
        std::shared_ptr<MockMPIContext> mpi;
        std::shared_ptr<NiceMock<MockTokenizer>> tokenizer;
    };

    class ScopedLogLevel
    {
    public:
        explicit ScopedLogLevel(LogLevel level)
            : previous_level_(Logger::getInstance().getLogLevel())
        {
            Logger::getInstance().setLogLevel(level);
        }

        ~ScopedLogLevel()
        {
            Logger::getInstance().setLogLevel(previous_level_);
        }

        ScopedLogLevel(const ScopedLogLevel &) = delete;
        ScopedLogLevel &operator=(const ScopedLogLevel &) = delete;

    private:
        LogLevel previous_level_;
    };
}

TEST(Test__CompletionMode, NonRootRankEntersWorkerLoopWithoutTokenization)
{
    ModeHarness h(/*rank=*/1, /*world_size=*/2);

    EXPECT_CALL(*h.runner, setMPICoordinatedMode(true)).Times(1);
    EXPECT_CALL(*h.runner, runMPIWorkerLoop()).Times(1);
    EXPECT_CALL(*h.runner, shutdown()).Times(1);
    EXPECT_CALL(*h.runner, shutdownMPIWorkers()).Times(0);
    EXPECT_CALL(*h.runner, prefill(_)).Times(0);
    EXPECT_CALL(*h.runner, decodeStep()).Times(0);
    EXPECT_CALL(*h.runner, setSamplingParams(_)).Times(0);
    EXPECT_CALL(*h.tokenizer, encode(_, _, _)).Times(0);

    CompletionMode mode;
    EXPECT_EQ(mode.execute(h.ctx), 0);
}

TEST(Test__CompletionMode, RootRankShutsDownWorkersOnSuccess)
{
    ModeHarness h(/*rank=*/0, /*world_size=*/2);

    EXPECT_CALL(*h.runner, setMPICoordinatedMode(true)).Times(1);
    EXPECT_CALL(*h.tokenizer, encode("Hello", false, false)).WillOnce(Return(std::vector<int>{1, 2}));
    EXPECT_CALL(*h.runner, prefill(ElementsAre(1, 2))).WillOnce(Return(true));
    EXPECT_CALL(*h.runner, setSamplingParams(_)).Times(1);
    EXPECT_CALL(*h.runner, decodeStep()).WillOnce(Return(tokenResult(42)));
    EXPECT_CALL(*h.tokenizer, is_stop_token(42)).Times(1).WillRepeatedly(Return(false));
    EXPECT_CALL(*h.tokenizer, decode_token(42)).WillOnce(Return(" answer"));
    EXPECT_CALL(*h.runner, flushStageTimeline()).Times(1);
    EXPECT_CALL(*h.runner, shutdownMPIWorkers()).Times(1);
    EXPECT_CALL(*h.runner, shutdown()).Times(1);

    CompletionMode mode;
    EXPECT_EQ(mode.execute(h.ctx), 0);
}

TEST(Test__CompletionMode, RootRankShutsDownWorkersOnTokenizationFailure)
{
    ModeHarness h(/*rank=*/0, /*world_size=*/2);

    EXPECT_CALL(*h.runner, setMPICoordinatedMode(true)).Times(1);
    EXPECT_CALL(*h.tokenizer, encode("Hello", false, false))
        .WillOnce(Throw(std::runtime_error("tokenizer failed")));
    EXPECT_CALL(*h.runner, prefill(_)).Times(0);
    EXPECT_CALL(*h.runner, decodeStep()).Times(0);
    EXPECT_CALL(*h.runner, shutdownMPIWorkers()).Times(1);
    EXPECT_CALL(*h.runner, shutdown()).Times(1);

    CompletionMode mode;
    EXPECT_EQ(mode.execute(h.ctx), 1);
}

TEST(Test__CompletionMode, RootRankShutsDownWorkersOnPrefillFailure)
{
    ModeHarness h(/*rank=*/0, /*world_size=*/2);

    EXPECT_CALL(*h.runner, setMPICoordinatedMode(true)).Times(1);
    EXPECT_CALL(*h.tokenizer, encode("Hello", false, false)).WillOnce(Return(std::vector<int>{1}));
    EXPECT_CALL(*h.runner, prefill(ElementsAre(1))).WillOnce(Return(false));
    EXPECT_CALL(*h.runner, setSamplingParams(_)).Times(0);
    EXPECT_CALL(*h.runner, decodeStep()).Times(0);
    EXPECT_CALL(*h.runner, shutdownMPIWorkers()).Times(1);
    EXPECT_CALL(*h.runner, shutdown()).Times(1);

    CompletionMode mode;
    EXPECT_EQ(mode.execute(h.ctx), 1);
}

TEST(Test__CompletionMode, RootRankShutsDownWorkersWhenPrefillThrows)
{
    ModeHarness h(/*rank=*/0, /*world_size=*/2);

    EXPECT_CALL(*h.runner, setMPICoordinatedMode(true)).Times(1);
    EXPECT_CALL(*h.tokenizer, encode("Hello", false, false)).WillOnce(Return(std::vector<int>{1}));
    EXPECT_CALL(*h.runner, prefill(ElementsAre(1)))
        .WillOnce(Throw(std::runtime_error("prefill threw")));
    EXPECT_CALL(*h.runner, setSamplingParams(_)).Times(0);
    EXPECT_CALL(*h.runner, decodeStep()).Times(0);
    EXPECT_CALL(*h.runner, flushStageTimeline()).Times(0);
    EXPECT_CALL(*h.runner, shutdownMPIWorkers()).Times(0);
    EXPECT_CALL(*h.runner, abortMPIWorkers("prefill threw")).Times(1);
    EXPECT_CALL(*h.runner, shutdown()).Times(1);

    CompletionMode mode;
    EXPECT_EQ(mode.execute(h.ctx), 1);
}

TEST(Test__CompletionMode, RootRankShutsDownWorkersOnDecodeFailure)
{
    ModeHarness h(/*rank=*/0, /*world_size=*/2);

    EXPECT_CALL(*h.runner, setMPICoordinatedMode(true)).Times(1);
    EXPECT_CALL(*h.tokenizer, encode("Hello", false, false)).WillOnce(Return(std::vector<int>{1}));
    EXPECT_CALL(*h.runner, prefill(ElementsAre(1))).WillOnce(Return(true));
    EXPECT_CALL(*h.runner, setSamplingParams(_)).Times(1);
    EXPECT_CALL(*h.runner, decodeStep()).WillOnce(Return(errorResult("decode failed")));
    EXPECT_CALL(*h.runner, shutdownMPIWorkers()).Times(1);
    EXPECT_CALL(*h.runner, shutdown()).Times(1);

    CompletionMode mode;
    EXPECT_EQ(mode.execute(h.ctx), 1);
}

TEST(Test__CompletionMode, RootRankShutsDownWorkersWhenDecodeThrows)
{
    ModeHarness h(/*rank=*/0, /*world_size=*/2);

    EXPECT_CALL(*h.runner, setMPICoordinatedMode(true)).Times(1);
    EXPECT_CALL(*h.tokenizer, encode("Hello", false, false)).WillOnce(Return(std::vector<int>{1}));
    EXPECT_CALL(*h.runner, prefill(ElementsAre(1))).WillOnce(Return(true));
    EXPECT_CALL(*h.runner, setSamplingParams(_)).Times(1);
    EXPECT_CALL(*h.runner, decodeStep())
        .WillOnce(Throw(std::runtime_error("decode threw")));
    EXPECT_CALL(*h.runner, flushStageTimeline()).Times(0);
    EXPECT_CALL(*h.runner, shutdownMPIWorkers()).Times(0);
    EXPECT_CALL(*h.runner, abortMPIWorkers("decode threw")).Times(1);
    EXPECT_CALL(*h.runner, shutdown()).Times(1);

    CompletionMode mode;
    EXPECT_EQ(mode.execute(h.ctx), 1);
}

TEST(Test__CompletionMode, SingleRankUsesDirectExecutionWithoutWorkerCoordination)
{
    ModeHarness h(/*rank=*/0, /*world_size=*/1);

    EXPECT_CALL(*h.runner, setMPICoordinatedMode(_)).Times(0);
    EXPECT_CALL(*h.runner, runMPIWorkerLoop()).Times(0);
    EXPECT_CALL(*h.runner, shutdownMPIWorkers()).Times(0);
    EXPECT_CALL(*h.tokenizer, encode("Hello", false, false)).WillOnce(Return(std::vector<int>{1}));
    EXPECT_CALL(*h.runner, prefill(ElementsAre(1))).WillOnce(Return(true));
    EXPECT_CALL(*h.runner, setSamplingParams(_)).Times(1);
    EXPECT_CALL(*h.runner, decodeStep()).WillOnce(Return(tokenResult(42)));
    EXPECT_CALL(*h.tokenizer, is_stop_token(42)).Times(1).WillRepeatedly(Return(false));
    EXPECT_CALL(*h.tokenizer, decode_token(42)).WillOnce(Return(" answer"));
    EXPECT_CALL(*h.runner, shutdown()).Times(1);

    CompletionMode mode;
    EXPECT_EQ(mode.execute(h.ctx), 0);
}

TEST(Test__CompletionMode, EmitsAllTokensReturnedByMultiTokenDecodeStep)
{
    ModeHarness h(/*rank=*/0, /*world_size=*/1);
    h.ctx.config.n_predict = 2;

    EXPECT_CALL(*h.tokenizer, encode("Hello", false, false)).WillOnce(Return(std::vector<int>{1}));
    EXPECT_CALL(*h.runner, prefill(ElementsAre(1))).WillOnce(Return(true));
    EXPECT_CALL(*h.runner, setSamplingParams(_)).Times(1);
    EXPECT_CALL(*h.runner, setDecodeStepTokenBudget(2)).Times(1);
    EXPECT_CALL(*h.runner, setDecodeStepTokenBudget(0)).Times(1);
    EXPECT_CALL(*h.runner, decodeStep()).WillOnce(Return(tokenResult({10, 11, 12})));
    EXPECT_CALL(*h.tokenizer, is_stop_token(10)).WillOnce(Return(false));
    EXPECT_CALL(*h.tokenizer, is_stop_token(11)).WillOnce(Return(false));
    EXPECT_CALL(*h.tokenizer, is_stop_token(12)).Times(0);
    EXPECT_CALL(*h.tokenizer, decode_token(10)).WillOnce(Return("A"));
    EXPECT_CALL(*h.tokenizer, decode_token(11)).WillOnce(Return("B"));
    EXPECT_CALL(*h.tokenizer, decode_token(12)).Times(0);
    EXPECT_CALL(*h.runner, shutdown()).Times(1);

    CompletionMode mode;
    EXPECT_EQ(mode.execute(h.ctx), 0);
}

TEST(Test__CompletionMode, UserTextStaysOnStdoutAndInfoLogsUseStderr)
{
    ScopedLogLevel log_level(LogLevel::INFO);
    ModeHarness h(/*rank=*/0, /*world_size=*/1);
    h.ctx.config.prompt = "Explain laminar flow";
    h.ctx.config.n_predict = 2;

    EXPECT_CALL(*h.tokenizer, encode("Explain laminar flow", false, false))
        .WillOnce(Return(std::vector<int>{1}));
    EXPECT_CALL(*h.runner, prefill(ElementsAre(1))).WillOnce(Return(true));
    EXPECT_CALL(*h.runner, setSamplingParams(_)).Times(1);
    EXPECT_CALL(*h.runner, setDecodeStepTokenBudget(2)).Times(1);
    EXPECT_CALL(*h.runner, setDecodeStepTokenBudget(0)).Times(1);
    EXPECT_CALL(*h.runner, decodeStep()).WillOnce(Return(tokenResult({10, 11})));
    EXPECT_CALL(*h.tokenizer, is_stop_token(10)).WillOnce(Return(false));
    EXPECT_CALL(*h.tokenizer, is_stop_token(11)).WillOnce(Return(false));
    EXPECT_CALL(*h.tokenizer, decode_token(10)).WillOnce(Return(" smooth"));
    EXPECT_CALL(*h.tokenizer, decode_token(11)).WillOnce(Return(" flow"));
    EXPECT_CALL(*h.runner, flushStageTimeline()).Times(1);
    EXPECT_CALL(*h.runner, shutdown()).Times(1);

    testing::internal::CaptureStdout();
    testing::internal::CaptureStderr();
    CompletionMode mode;
    EXPECT_EQ(mode.execute(h.ctx), 0);
    const std::string stderr_text = testing::internal::GetCapturedStderr();
    const std::string stdout_text = testing::internal::GetCapturedStdout();

    EXPECT_THAT(stdout_text, HasSubstr("Prompt:\nExplain laminar flow\n\nResponse:\n smooth flow"));
    EXPECT_THAT(stdout_text, Not(HasSubstr("[INFO")));
    EXPECT_THAT(stdout_text, Not(HasSubstr("Running prefill")));
    EXPECT_THAT(stderr_text, HasSubstr("[INFO"));
    EXPECT_THAT(stderr_text, HasSubstr("Running prefill (1 tokens)"));
}

TEST(Test__SingleShotChatMode, NonRootRankEntersWorkerLoopWithoutTokenizerUse)
{
    ModeHarness h(/*rank=*/1, /*world_size=*/2);
    h.ctx.config.single_shot_chat = true;

    EXPECT_CALL(*h.runner, setMPICoordinatedMode(true)).Times(1);
    EXPECT_CALL(*h.runner, runMPIWorkerLoop()).Times(1);
    EXPECT_CALL(*h.runner, shutdown()).Times(1);
    EXPECT_CALL(*h.runner, shutdownMPIWorkers()).Times(0);
    EXPECT_CALL(*h.runner, prefill(_)).Times(0);
    EXPECT_CALL(*h.runner, decodeStep()).Times(0);
    EXPECT_CALL(*h.tokenizer, hasChatTemplate()).Times(0);
    EXPECT_CALL(*h.tokenizer, encodeChat(_, _, _)).Times(0);

    SingleShotChatMode mode;
    EXPECT_EQ(mode.execute(h.ctx), 0);
}

TEST(Test__SingleShotChatMode, RootRankShutsDownWorkersOnSuccess)
{
    ModeHarness h(/*rank=*/0, /*world_size=*/2);
    h.ctx.config.single_shot_chat = true;
    h.ctx.config.system_prompt = "system";

    EXPECT_CALL(*h.runner, setMPICoordinatedMode(true)).Times(1);
    EXPECT_CALL(*h.tokenizer, hasChatTemplate()).WillOnce(Return(true));
    EXPECT_CALL(*h.tokenizer, encodeChat(_, true, "")).WillOnce(Return(std::vector<int>{7, 8}));
    EXPECT_CALL(*h.runner, prefill(ElementsAre(7, 8))).WillOnce(Return(true));
    EXPECT_CALL(*h.runner, setSamplingParams(_)).Times(1);
    EXPECT_CALL(*h.runner, decodeStep()).WillOnce(Return(tokenResult(42)));
    EXPECT_CALL(*h.tokenizer, is_stop_token(42)).Times(1).WillRepeatedly(Return(false));
    EXPECT_CALL(*h.tokenizer, decode_token(42)).WillOnce(Return(" answer"));
    EXPECT_CALL(*h.runner, flushStageTimeline()).Times(1);
    EXPECT_CALL(*h.runner, shutdownMPIWorkers()).Times(1);
    EXPECT_CALL(*h.runner, shutdown()).Times(1);

    SingleShotChatMode mode;
    EXPECT_EQ(mode.execute(h.ctx), 0);
}

TEST(Test__SingleShotChatMode, RootRankShutsDownWorkersWhenChatTemplateMissing)
{
    ModeHarness h(/*rank=*/0, /*world_size=*/2);
    h.ctx.config.single_shot_chat = true;

    EXPECT_CALL(*h.runner, setMPICoordinatedMode(true)).Times(1);
    EXPECT_CALL(*h.tokenizer, hasChatTemplate()).WillOnce(Return(false));
    EXPECT_CALL(*h.tokenizer, encodeChat(_, _, _)).Times(0);
    EXPECT_CALL(*h.runner, prefill(_)).Times(0);
    EXPECT_CALL(*h.runner, shutdownMPIWorkers()).Times(1);
    EXPECT_CALL(*h.runner, shutdown()).Times(1);

    SingleShotChatMode mode;
    EXPECT_EQ(mode.execute(h.ctx), 1);
}

TEST(Test__SingleShotChatMode, RootRankShutsDownWorkersWhenPrefillThrows)
{
    ModeHarness h(/*rank=*/0, /*world_size=*/2);
    h.ctx.config.single_shot_chat = true;

    EXPECT_CALL(*h.runner, setMPICoordinatedMode(true)).Times(1);
    EXPECT_CALL(*h.tokenizer, hasChatTemplate()).WillOnce(Return(true));
    EXPECT_CALL(*h.tokenizer, encodeChat(_, true, "")).WillOnce(Return(std::vector<int>{7}));
    EXPECT_CALL(*h.runner, prefill(ElementsAre(7)))
        .WillOnce(Throw(std::runtime_error("chat prefill threw")));
    EXPECT_CALL(*h.runner, setSamplingParams(_)).Times(0);
    EXPECT_CALL(*h.runner, decodeStep()).Times(0);
    EXPECT_CALL(*h.runner, flushStageTimeline()).Times(0);
    EXPECT_CALL(*h.runner, shutdownMPIWorkers()).Times(0);
    EXPECT_CALL(*h.runner, abortMPIWorkers("chat prefill threw")).Times(1);
    EXPECT_CALL(*h.runner, shutdown()).Times(1);

    SingleShotChatMode mode;
    EXPECT_EQ(mode.execute(h.ctx), 1);
}

TEST(Test__SingleShotChatMode, RootRankShutsDownWorkersWhenDecodeThrows)
{
    ModeHarness h(/*rank=*/0, /*world_size=*/2);
    h.ctx.config.single_shot_chat = true;

    EXPECT_CALL(*h.runner, setMPICoordinatedMode(true)).Times(1);
    EXPECT_CALL(*h.tokenizer, hasChatTemplate()).WillOnce(Return(true));
    EXPECT_CALL(*h.tokenizer, encodeChat(_, true, "")).WillOnce(Return(std::vector<int>{7}));
    EXPECT_CALL(*h.runner, prefill(ElementsAre(7))).WillOnce(Return(true));
    EXPECT_CALL(*h.runner, setSamplingParams(_)).Times(1);
    EXPECT_CALL(*h.runner, decodeStep())
        .WillOnce(Throw(std::runtime_error("chat decode threw")));
    EXPECT_CALL(*h.runner, flushStageTimeline()).Times(0);
    EXPECT_CALL(*h.runner, shutdownMPIWorkers()).Times(0);
    EXPECT_CALL(*h.runner, abortMPIWorkers("chat decode threw")).Times(1);
    EXPECT_CALL(*h.runner, shutdown()).Times(1);

    SingleShotChatMode mode;
    EXPECT_EQ(mode.execute(h.ctx), 1);
}

TEST(Test__SingleShotChatMode, SingleRankUsesDirectExecutionWithoutWorkerCoordination)
{
    ModeHarness h(/*rank=*/0, /*world_size=*/1);
    h.ctx.config.single_shot_chat = true;

    EXPECT_CALL(*h.runner, setMPICoordinatedMode(_)).Times(0);
    EXPECT_CALL(*h.runner, runMPIWorkerLoop()).Times(0);
    EXPECT_CALL(*h.runner, shutdownMPIWorkers()).Times(0);
    EXPECT_CALL(*h.tokenizer, hasChatTemplate()).WillOnce(Return(true));
    EXPECT_CALL(*h.tokenizer, encodeChat(_, true, "")).WillOnce(Return(std::vector<int>{7}));
    EXPECT_CALL(*h.runner, prefill(ElementsAre(7))).WillOnce(Return(true));
    EXPECT_CALL(*h.runner, setSamplingParams(_)).Times(1);
    EXPECT_CALL(*h.runner, decodeStep()).WillOnce(Return(tokenResult(42)));
    EXPECT_CALL(*h.tokenizer, is_stop_token(42)).Times(1).WillRepeatedly(Return(false));
    EXPECT_CALL(*h.tokenizer, decode_token(42)).WillOnce(Return(" answer"));
    EXPECT_CALL(*h.runner, shutdown()).Times(1);

    SingleShotChatMode mode;
    EXPECT_EQ(mode.execute(h.ctx), 0);
}

TEST(Test__SingleShotChatMode, EmitsAllTokensReturnedByMultiTokenDecodeStep)
{
    ModeHarness h(/*rank=*/0, /*world_size=*/1);
    h.ctx.config.single_shot_chat = true;
    h.ctx.config.n_predict = 2;

    EXPECT_CALL(*h.tokenizer, hasChatTemplate()).WillOnce(Return(true));
    EXPECT_CALL(*h.tokenizer, encodeChat(_, true, "")).WillOnce(Return(std::vector<int>{7}));
    EXPECT_CALL(*h.runner, prefill(ElementsAre(7))).WillOnce(Return(true));
    EXPECT_CALL(*h.runner, setSamplingParams(_)).Times(1);
    EXPECT_CALL(*h.runner, setDecodeStepTokenBudget(2)).Times(1);
    EXPECT_CALL(*h.runner, setDecodeStepTokenBudget(0)).Times(1);
    EXPECT_CALL(*h.runner, decodeStep()).WillOnce(Return(tokenResult({20, 21, 22})));
    EXPECT_CALL(*h.tokenizer, is_stop_token(20)).WillOnce(Return(false));
    EXPECT_CALL(*h.tokenizer, is_stop_token(21)).WillOnce(Return(false));
    EXPECT_CALL(*h.tokenizer, is_stop_token(22)).Times(0);
    EXPECT_CALL(*h.tokenizer, decode_token(20)).WillOnce(Return("H"));
    EXPECT_CALL(*h.tokenizer, decode_token(21)).WillOnce(Return("i"));
    EXPECT_CALL(*h.tokenizer, decode_token(22)).Times(0);
    EXPECT_CALL(*h.runner, shutdown()).Times(1);

    SingleShotChatMode mode;
    EXPECT_EQ(mode.execute(h.ctx), 0);
}

TEST(Test__SingleShotChatMode, UserTextStaysOnStdoutAndInfoLogsUseStderr)
{
    ScopedLogLevel log_level(LogLevel::INFO);
    ModeHarness h(/*rank=*/0, /*world_size=*/1);
    h.ctx.config.single_shot_chat = true;
    h.ctx.config.prompt = "Tell me a tiny story";
    h.ctx.config.n_predict = 2;

    EXPECT_CALL(*h.tokenizer, hasChatTemplate()).WillOnce(Return(true));
    EXPECT_CALL(*h.tokenizer, encodeChat(_, true, "")).WillOnce(Return(std::vector<int>{7}));
    EXPECT_CALL(*h.runner, prefill(ElementsAre(7))).WillOnce(Return(true));
    EXPECT_CALL(*h.runner, setSamplingParams(_)).Times(1);
    EXPECT_CALL(*h.runner, setDecodeStepTokenBudget(2)).Times(1);
    EXPECT_CALL(*h.runner, setDecodeStepTokenBudget(0)).Times(1);
    EXPECT_CALL(*h.runner, decodeStep()).WillOnce(Return(tokenResult({20, 21})));
    EXPECT_CALL(*h.tokenizer, is_stop_token(20)).WillOnce(Return(false));
    EXPECT_CALL(*h.tokenizer, is_stop_token(21)).WillOnce(Return(false));
    EXPECT_CALL(*h.tokenizer, decode_token(20)).WillOnce(Return(" Once"));
    EXPECT_CALL(*h.tokenizer, decode_token(21)).WillOnce(Return(" there"));
    EXPECT_CALL(*h.runner, flushStageTimeline()).Times(1);
    EXPECT_CALL(*h.runner, shutdown()).Times(1);

    testing::internal::CaptureStdout();
    testing::internal::CaptureStderr();
    SingleShotChatMode mode;
    EXPECT_EQ(mode.execute(h.ctx), 0);
    const std::string stderr_text = testing::internal::GetCapturedStderr();
    const std::string stdout_text = testing::internal::GetCapturedStdout();

    EXPECT_THAT(stdout_text, HasSubstr("Prompt:\nTell me a tiny story\n\nResponse:\n Once there"));
    EXPECT_THAT(stdout_text, Not(HasSubstr("[INFO")));
    EXPECT_THAT(stdout_text, Not(HasSubstr("Generating response")));
    EXPECT_THAT(stderr_text, HasSubstr("[INFO"));
    EXPECT_THAT(stderr_text, HasSubstr("Generating response (max 2 tokens)"));
}

int main(int argc, char **argv)
{
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (!initialized)
    {
        MPI_Init(&argc, &argv);
    }

    ::testing::InitGoogleMock(&argc, argv);
    const int result = RUN_ALL_TESTS();

    int finalized = 0;
    MPI_Finalized(&finalized);
    if (!finalized)
    {
        MPI_Finalize();
    }
    return result;
}
