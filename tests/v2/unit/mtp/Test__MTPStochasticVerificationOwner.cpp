/**
 * @file Test__MTPStochasticVerificationOwner.cpp
 * @brief Device-free regression for the production stochastic admission seam.
 *
 * The installed runner contract is queried exactly as in production. These
 * tests do not emulate GPU arithmetic: backend integration owns that proof.
 * Their purpose is to keep expert-service ranks distinct from vocabulary peers
 * and reject incomplete resident verification/publication before execution.
 */
#include "execution/runner/MTPStochasticVerificationOwner.h"
#include "execution/local_execution/orchestrators/IInferenceRunner.h"
#include "execution/mpi_orchestration/RankExecutionPlan.h"

#include <gtest/gtest.h>

namespace llaminar2
{
    namespace
    {
        /** @brief One deliberately absent part of a runner's installed contract. */
        enum class MissingOwner { None, MirroredHead, Verifier, Publisher, TokenCoordination };

        /** @brief Immutable runner double; executing or reading logits is forbidden. */
        class OwnerRunner final : public IInferenceRunner
        {
        public:
            /** @brief Describe a complete or intentionally incomplete installed runner. */
            explicit OwnerRunner(DeviceId device, MissingOwner missing = MissingOwner::None)
                : device_(device), missing_(missing) {}
            /** @brief Admission must not execute model work. */
            bool forward(const int *, int) override { ADD_FAILURE(); return false; }
            /** @brief Admission must not materialize model data. */
            const float *logits() const override { ADD_FAILURE(); return nullptr; }
            /** @return Minimal nonempty vocabulary geometry. */
            int vocab_size() const override { return 16; }
            /** @brief Admission must not mutate cache state. */
            void clear_cache() override { ADD_FAILURE(); }
            /** @return No live position is consulted by this test. */
            int get_position() const override { ADD_FAILURE(); return 0; }
            /** @return The production graph interface under test. */
            ExecutionPath executionPath() const override { return ExecutionPath::GRAPH; }
            /** @return No model-specific admission is allowed. */
            const char *architecture() const override { return "ownership-test"; }
            /** @return Backend identity without querying hardware. */
            DeviceId primaryDeviceId() const override { return device_; }
            /** @return Whether the full terminal head is installed. */
            bool usesMirroredMTPHeadForVerifier() const override
                { return missing_ != MissingOwner::MirroredHead; }
            /** @return Whether device-local stochastic resources are installed. */
            bool supportsDeviceStochasticMTPVerification() const override
                { return device_.is_gpu() && missing_ != MissingOwner::Verifier; }
            /** @return Whether the exact device outcome owns live-state publication. */
            bool supportsDeviceResidentMTPSpecStatePublication() const override
                { return device_.is_gpu() && missing_ != MissingOwner::Publisher; }
            /** @return Whether global CPU vocabulary peers coordinate their outcome. */
            bool supportsMTPTokenCoordination() const override
                { return missing_ != MissingOwner::TokenCoordination; }
        private:
            DeviceId device_;
            MissingOwner missing_;
        };

        /** @brief Describe local participants without initializing a backend. */
        RankExecutionPlan localPlan(DeviceId device, int participants)
        {
            RankExecutionPlan plan;
            plan.rank = 7; // Authority is not coupled to rank zero.
            for (int index = 0; index < participants; ++index)
                plan.local_tp_devices.push_back(device.is_cuda()
                    ? GlobalDeviceAddress::cuda(index) : GlobalDeviceAddress::rocm(index));
            return plan;
        }
    }

    /** @brief Remote expert services do not turn a complete GPU head into global TP. */
    TEST(MTPStochasticVerificationOwner, OverlayContinuationOwnsSingleAndMirroredGPUOutcomes)
    {
        for (const auto device : {DeviceId::cuda(0), DeviceId::rocm(0)})
            for (const int participants : {1, 2, 4, 8})
            {
                const OwnerRunner runner(device);
                const auto plan = localPlan(device, participants);
                EXPECT_TRUE(mtpStochasticVerificationOwnerFailure(plan, runner,
                    MTPRankParticipation::ExpertTransactionContinuation).empty());
                EXPECT_TRUE(mtpStochasticVerificationOwnerFailure(plan, runner,
                    MTPRankParticipation::SingleRank).empty());
            }
    }

    /** @brief Removing any publisher or distribution owner cannot admit a fake GPU lane. */
    TEST(MTPStochasticVerificationOwner, RejectsIncompleteOverlayAndLocalOwners)
    {
        for (const auto device : {DeviceId::cuda(0), DeviceId::rocm(0)})
            for (const auto missing : {MissingOwner::Verifier, MissingOwner::Publisher})
                for (const int participants : {1, 2, 8})
                {
                    const OwnerRunner runner(device, missing);
                    EXPECT_FALSE(mtpStochasticVerificationOwnerFailure(
                        localPlan(device, participants), runner,
                        MTPRankParticipation::ExpertTransactionContinuation).empty());
                }
        for (const auto device : {DeviceId::cuda(0), DeviceId::rocm(0)})
            for (const auto role : {MTPRankParticipation::SingleRank,
                                    MTPRankParticipation::ExpertTransactionContinuation})
                EXPECT_FALSE(mtpStochasticVerificationOwnerFailure(localPlan(device, 2),
                    OwnerRunner(device, MissingOwner::MirroredHead), role).empty());
    }

    /** @brief MPI size alone never authorizes a global GPU or uncoordinated CPU sampler. */
    TEST(MTPStochasticVerificationOwner, CollectivePeersRequireActualGatheredCPUOwnership)
    {
        for (const auto device : {DeviceId::cpu(), DeviceId::cuda(0), DeviceId::rocm(0)})
        {
            RankExecutionPlan plan;
            EXPECT_FALSE(mtpStochasticVerificationOwnerFailure(plan, OwnerRunner(device),
                MTPRankParticipation::ModelCollectivePeer).empty());
            plan.global_tp_domain_id = 3;
            plan.global_tp_domain_size = 2;
            EXPECT_EQ(mtpStochasticVerificationOwnerFailure(plan, OwnerRunner(device),
                MTPRankParticipation::ModelCollectivePeer).empty(), device.is_cpu());
            EXPECT_FALSE(mtpStochasticVerificationOwnerFailure(plan,
                OwnerRunner(device, MissingOwner::TokenCoordination),
                MTPRankParticipation::ModelCollectivePeer).empty());
        }
    }

    /** @brief An expert follower cannot acquire authority from installed sampler helpers. */
    TEST(MTPStochasticVerificationOwner, ExpertFollowersNeverSample)
    {
        for (const auto device : {DeviceId::cpu(), DeviceId::cuda(0), DeviceId::rocm(0)})
            EXPECT_FALSE(mtpStochasticVerificationOwnerFailure({}, OwnerRunner(device),
                MTPRankParticipation::ExpertTransactionFollower).empty());
    }
}
