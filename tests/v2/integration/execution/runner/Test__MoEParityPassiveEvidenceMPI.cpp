/**
 * @file Test__MoEParityPassiveEvidenceMPI.cpp
 * @brief Model-free native MPI lifecycle of original session passive evidence.
 *
 * The real session and MPI collectives execute. The existing runner mock models
 * only command closure and teardown, never model/source or native readiness.
 * Audits must join after the original terminal command and before local owners
 * retire; independent participant records remain distinct until the real sums.
 */
#include "utils/ScopedMoEParityCoordinatedSession.h"
#include "mocks/MockOrchestrationRunner.h"
#include "utils/MPIContext.h"
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <array>
#include <vector>
#include <stdexcept>
#include <optional>

namespace llaminar2::test::parity {
namespace {
/** @brief Test-only lifecycle observations, not model readiness. */
struct Lifetime { bool closed=false; bool retired=false; int audits=0; int retirement_attempts=0; };
/**
 * @brief Configure only the session's original command-loop/retirement interface.
 * @param runner Existing lifecycle mock; its default inference readiness stays false.
 * @param state Passive local observations.
 * @param rank Actual MPI rank.
 * @param authority Original selected command root, tested in both directions.
 * @param fail_retirement_rank Optional original rank throwing from its single local shutdown attempt.
 */
void configure(::testing::NiceMock<MockOrchestrationRunner>& runner,Lifetime& state,int rank,int authority,
    std::optional<int> fail_retirement_rank={})
{
    ON_CALL(runner,coordinatedRootRank()).WillByDefault(::testing::Return(authority));
    EXPECT_CALL(runner,setMPICoordinatedMode(true)).Times(1);
    EXPECT_CALL(runner,setMPICoordinatedMode(false)).Times(1);
    EXPECT_CALL(runner,shutdown()).WillOnce([&state,rank,fail_retirement_rank] {
        EXPECT_TRUE(state.closed);EXPECT_FALSE(state.retired);++state.retirement_attempts;
        if(fail_retirement_rank==rank)throw std::runtime_error("original local retirement failure");
        state.retired=true;
    });
    if(rank==authority)
    {
        EXPECT_CALL(runner,runMPIWorkerLoop()).Times(0);
        EXPECT_CALL(runner,shutdownMPIWorkers()).WillOnce([&state,authority] {
            int terminal=73;
            EXPECT_EQ(MPI_Bcast(&terminal,1,MPI_INT,authority,MPI_COMM_WORLD),MPI_SUCCESS);
            state.closed=true;
        });
    }
    else
    {
        EXPECT_CALL(runner,shutdownMPIWorkers()).Times(0);
        EXPECT_CALL(runner,runMPIWorkerLoop()).WillOnce([&state,authority] {
            int terminal=0;
            EXPECT_EQ(MPI_Bcast(&terminal,1,MPI_INT,authority,MPI_COMM_WORLD),MPI_SUCCESS);
            EXPECT_EQ(terminal,73);state.closed=true;
        });
    }
}
/** @return Actual two-rank identity; no accelerator/NUMA readiness is inferred. */
int originalRank()
{
    int rank=-1,size=0;MPI_Comm_rank(MPI_COMM_WORLD,&rank);MPI_Comm_size(MPI_COMM_WORLD,&size);
    EXPECT_EQ(size,2);return rank;
}
}
/** @test Both original ranks preserve coverage/totals/movement sum order before owner retirement. */
TEST(MoEParityPassiveEvidenceMPI, CollectiveAuditsJoinAfterShutdownBeforeRetirement)
{
    const int rank=originalRank();
    for(int authority:{0,1})
    {
        Lifetime state;::testing::NiceMock<MockOrchestrationRunner> runner;
        configure(runner,state,rank,authority);
        CoordinatedPassiveEvidenceObligation obligation;
        std::vector<int> sequence;
        auto audit=obligation.bind([&] {
            EXPECT_TRUE(state.closed);EXPECT_FALSE(state.retired);++state.audits;
            sequence.push_back(1);
            float ready=0,any_failed=-1;
            ASSERT_EQ(MPI_Allreduce(&ready,&any_failed,1,MPI_FLOAT,MPI_SUM,MPI_COMM_WORLD),MPI_SUCCESS);
            ASSERT_EQ(any_failed,0);
            for(int count:{49,2,4})
            {
                sequence.push_back(count);
                std::vector<float> local(count,float(rank+1)),global(count,0);
                ASSERT_EQ(MPI_Allreduce(local.data(),global.data(),count,MPI_FLOAT,MPI_SUM,MPI_COMM_WORLD),MPI_SUCCESS);
                for(float value:global)EXPECT_EQ(value,3.0f);
            }
        });
        {
            ScopedMoEParityCoordinatedSession session(runner,std::move(audit));
            obligation.requireServing();
            if(session.isAuthority())session.finishAuthority();else session.serveFollowerAndFinish();
            EXPECT_THROW(session.finishAuthority(),std::logic_error);
        }
        EXPECT_TRUE(state.retired);EXPECT_EQ(state.audits,1);
        EXPECT_EQ(sequence,(std::vector<int>{1,49,2,4}));
        EXPECT_NO_THROW(obligation.requireCompletedIfBound());
    }
}
/** @test Existing callback-free sessions retain original terminal/retirement behavior. */
TEST(MoEParityPassiveEvidenceMPI, OrdinarySessionWithoutDeferredAuditIsUnchanged)
{
    const int rank=originalRank();Lifetime state;::testing::NiceMock<MockOrchestrationRunner> runner;
    configure(runner,state,rank,0);
    {
        ScopedMoEParityCoordinatedSession session(runner);
        if(session.isAuthority())session.finishAuthority();else session.serveFollowerAndFinish();
    }
    EXPECT_TRUE(state.retired);EXPECT_EQ(state.audits,0);
}
/** @test A local-only callback error is agreed before every original rank retires, without reentry. */
TEST(MoEParityPassiveEvidenceMPI, LocalAuditFailureRetiresBothRanksExactlyOnce)
{
    const int rank=originalRank();Lifetime state;::testing::NiceMock<MockOrchestrationRunner> runner;
    configure(runner,state,rank,0);CoordinatedPassiveEvidenceObligation obligation;
    {
        ScopedMoEParityCoordinatedSession session(runner,obligation.bind([&] {
            EXPECT_TRUE(state.closed);EXPECT_FALSE(state.retired);++state.audits;
            // No internal collective is entered here. Terminal consensus cannot
            // rescue divergent user collectives, and this test makes no such claim.
            if(rank==0)throw std::runtime_error("original local audit failure");
        }));
        if(session.isAuthority())EXPECT_THROW(session.finishAuthority(),std::runtime_error);
        else EXPECT_THROW(session.serveFollowerAndFinish(),std::runtime_error);
    }
    EXPECT_TRUE(state.retired);EXPECT_EQ(state.audits,1);
    EXPECT_NO_THROW(obligation.requireCompletedIfBound());
}
/** @test A rank-local prerequisite failure joins before payload audits and leaves both peers able to retire. */
TEST(MoEParityPassiveEvidenceMPI, PrerequisiteFailureCannotEnterMismatchedPayloadCollectives)
{
    const int rank=originalRank();Lifetime state;::testing::NiceMock<MockOrchestrationRunner> runner;
    configure(runner,state,rank,0);CoordinatedPassiveEvidenceObligation obligation;
    int payload_audits=0;
    {
        ScopedMoEParityCoordinatedSession session(runner,obligation.bind([&] {
            EXPECT_TRUE(state.closed);EXPECT_FALSE(state.retired);++state.audits;
            float local_failed=rank==1?1.0f:0.0f,any_failed=0;
            ASSERT_EQ(MPI_Allreduce(&local_failed,&any_failed,1,MPI_FLOAT,MPI_SUM,MPI_COMM_WORLD),MPI_SUCCESS);
            EXPECT_EQ(any_failed,1.0f);
            if(any_failed!=0)return;
            ++payload_audits;
        }));
        if(session.isAuthority())session.finishAuthority();else session.serveFollowerAndFinish();
    }
    EXPECT_TRUE(state.retired);EXPECT_EQ(state.audits,1);EXPECT_EQ(payload_audits,0);
    EXPECT_NO_THROW(obligation.requireCompletedIfBound());
}
/** @test One-rank shutdown failure reaches a common terminal boundary without retry or false retirement. */
TEST(MoEParityPassiveEvidenceMPI, AsymmetricShutdownFailureAttemptsEachOriginalOwnerOnce)
{
    const int rank=originalRank();
    for(int fail_rank:{0,1})
    {
        Lifetime state;::testing::NiceMock<MockOrchestrationRunner> runner;
        configure(runner,state,rank,0,fail_rank);
        CoordinatedPassiveEvidenceObligation obligation;
        {
            ScopedMoEParityCoordinatedSession session(runner,obligation.bind([&] {
                EXPECT_TRUE(state.closed);EXPECT_FALSE(state.retired);++state.audits;
            }));
            if(session.isAuthority())EXPECT_THROW(session.finishAuthority(),std::runtime_error);
            else EXPECT_THROW(session.serveFollowerAndFinish(),std::runtime_error);
        }
        EXPECT_EQ(state.retirement_attempts,1);EXPECT_EQ(state.audits,1);
        EXPECT_EQ(state.retired,rank!=fail_rank); // A thrown cleanup is not certified as retired.
        EXPECT_NO_THROW(obligation.requireCompletedIfBound());
    }
}
/** @test Actual helper agrees original sparse IDs, rejects differing extents and IDs before coverage reductions. */
TEST(MoEParityPassiveEvidenceMPI, OriginalAuditLayerCountsAndIDsMustMatch)
{
    const int rank=originalRank();MPIContext original(rank,2,MPI_COMM_WORLD);
    const std::vector<int> matching{0,7,48};
    EXPECT_NO_THROW(requireCoordinatedPassiveEvidenceLayers(original,matching));
    EXPECT_NO_THROW(requireCoordinatedPassiveEvidenceLayers(original,{}));
    const std::vector<int> different_count=rank==0?std::vector<int>{0,7,48}:std::vector<int>{0,48};
    EXPECT_THROW(requireCoordinatedPassiveEvidenceLayers(original,different_count),std::logic_error);
    const std::vector<int> different_ids=rank==0?std::vector<int>{0,7,48}:std::vector<int>{0,8,48};
    EXPECT_THROW(requireCoordinatedPassiveEvidenceLayers(original,different_ids),std::logic_error);
    const std::vector<int> duplicate=rank==0?matching:std::vector<int>{0,7,7};
    EXPECT_THROW(requireCoordinatedPassiveEvidenceLayers(original,duplicate),std::logic_error);
    // All original peers remain at the same protocol ordinal after rejection.
    int local=1,total=0;
    ASSERT_EQ(MPI_Allreduce(&local,&total,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD),MPI_SUCCESS);
    EXPECT_EQ(total,2);
}
/** @test Missing/duplicate deferred obligations are explicit lifecycle failures, not silent skips. */
TEST(MoEParityPassiveEvidenceMPI, MissingAndDuplicateAuditObligationsReject)
{
    CoordinatedPassiveEvidenceObligation obligation;
    EXPECT_THROW(obligation.requireServing(),std::logic_error);
    EXPECT_NO_THROW(obligation.requireCompletedIfBound()); // Ordinary, never-deferred path.
    EXPECT_THROW(obligation.bind({}),std::logic_error);
    auto audit=obligation.bind([]{});
    EXPECT_THROW(obligation.requireCompletedIfBound(),std::logic_error);
    EXPECT_THROW(obligation.bind([]{}),std::logic_error);
    audit();EXPECT_NO_THROW(obligation.requireCompletedIfBound());
    EXPECT_THROW(audit(),std::logic_error);
}
}
