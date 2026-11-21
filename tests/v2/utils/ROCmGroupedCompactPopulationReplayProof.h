/**
 * @file ROCmGroupedCompactPopulationReplayProof.h
 * @brief Adversarial capture-construction replay for Compact's shared directory lease.
 *
 * These auxiliary native graphs are inspected and never launched. Successful
 * production packets come from the actual parent. Tests perturb one immutable
 * binding at a time and prove rejection records no unauthorized down work.
 * Numerical, poisoned/reset and producer-omission replay remains in the parent
 * fixture and exercises the real optimized kernels.
 */
#pragma once
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "kernels/rocm/gemm/ROCmMoEGroupedCompactPopulation.h"
#include <gtest/gtest.h>
#include <functional>
#include <optional>
#include <vector>

namespace llaminar2::test
{
/** @tparam Context Actual GPU context interface. @tparam Operation Capture-only callback.
 * @param context Production context. @param stream Exact native recording stream.
 * @param operation Submission sequence. @return Actual captured kernel inventory.
 * @throws std::runtime_error For failed native capture or inspection. */
template<class Context,class Operation>
std::vector<GPUGraphKernelNodeInfo> captureCompactPopulationProbe(
    Context &context,void *stream,Operation &&operation)
{
    auto graph = context.createGraphCapture(stream);
    {
        ScopedBackendGraphCapture capture(context,*graph,"Compact population ownership replay");
        if (!capture.begin()) throw std::runtime_error("Compact probe could not begin capture");
        operation();
        capture.finish();
    }
    std::vector<GPUGraphKernelNodeInfo> nodes;
    if (!graph->inspectKernelNodes(nodes)) throw std::runtime_error("Compact probe could not inspect capture");
    return nodes;
}

/** @tparam Context Actual GPU context interface.
 * @param context Production context. @param stream Exact native stream.
 * @param key Independent complete-operation fixture facts. @param workspace Canonical arena bindings.
 * @param gu Captured production GU packet. @param down Captured production down packet. */
template<class Context>
void replayCompactPopulationSubmission(Context &context,void *stream,
    rocm::MoEGroupedPopulationKey key,rocm::MoEGroupedPopulationWorkspace workspace,
    const rocm::MoEGroupedWmmaArguments &gu,const rocm::MoEGroupedWmmaArguments &down)
{
    using namespace rocm;
    using Transaction = MoEGroupedCompactPopulationTransaction;
    const auto exact = MoEGroupedPopulationWorkspace{workspace.selection,workspace.directory,Transaction::kDirectoryWords};
    const auto scope = [&](auto &owner,MoEGroupedPopulationWorkspace storage) {
        owner.emplace(key,storage,gu.group_counts,gu.group_offsets,gu.device,stream);
    };
    auto nodes = captureCompactPopulationProbe(context,stream,[&] {
        auto short_bank = exact; --short_bank.directory_words;
        EXPECT_THROW((Transaction(key,short_bank,gu.group_counts,gu.group_offsets,gu.device,stream)),std::invalid_argument);
    });
    ASSERT_TRUE(nodes.empty());

    nodes = captureCompactPopulationProbe(context,stream,[&] {
        Transaction owner(key,exact,gu.group_counts,gu.group_offsets,gu.device,stream);
        auto produced = owner.gateUp(key.gateup_physical,gu);
        ASSERT_TRUE(produced); ASSERT_EQ(produced->receipt().leaves(),1u);
        auto moved = std::move(*produced);
        EXPECT_THROW((void)owner.down(std::move(*produced),key.down_physical,down),std::logic_error);
        auto submitted = owner.down(std::move(moved),key.down_physical,down);
        ASSERT_TRUE(submitted); EXPECT_EQ(submitted->leaves(),1u);
        EXPECT_THROW((void)owner.down(std::move(moved),key.down_physical,down),std::logic_error);
        EXPECT_THROW((void)owner.gateUp(key.gateup_physical,gu),std::logic_error);
    });
    ASSERT_EQ(nodes.size(),3u); // One shared planner and one adaptive consumer per role.

    const int source = key.gateup_physical == 1u ? 0 : 13;
    const ROCmMoEGroupedPrefillRouteKey route_key{
        .gateup_codebook=static_cast<uint8_t>(source),.down_codebook=static_cast<uint8_t>(source),
        .hidden_size=key.hidden,.expert_width=key.intermediate,.expert_count=key.experts,
        .top_k=key.top_k,.rows=key.rows,.operation=key.operation};
    // Each mutation preserves a constructible packet but breaks one binding.
    // Rejection must happen before a foreign publication can be read on device.
    std::vector<std::function<void(MoEGroupedWmmaArguments &)>> mutations{
        [](auto &v) { ++v.group_counts; }, [](auto &v) { ++v.group_offsets; },
        [](auto &v) { ++v.directory; }, [](auto &v) { --v.directory_capacity; },
        [](auto &v) { --v.rows; }, [](auto &v) { --v.experts; },
        [](auto &v) { v.n += 32; }, [](auto &v) { v.k += 32; },
        [](auto &v) { ++v.device; },
        [&](auto &v) { v.admission = ROCmMoEGroupedRouteAdmission(route_key,gu.group_counts+1,gu.group_offsets); },
        [&](auto &v) { v.admission = ROCmMoEGroupedRouteAdmission(route_key,gu.group_counts,gu.group_offsets+1); },
        [&](auto &v) { auto k=route_key; ++k.expert_count; v.admission=ROCmMoEGroupedRouteAdmission(k,gu.group_counts,gu.group_offsets); },
        [&](auto &v) { auto k=route_key; --k.rows; v.admission=ROCmMoEGroupedRouteAdmission(k,gu.group_counts,gu.group_offsets); },
        [&](auto &v) { auto k=route_key; k.operation=MoEPrefillMeasuredOperation::Unmeasured;
            v.admission=ROCmMoEGroupedRouteAdmission(k,gu.group_counts,gu.group_offsets); },
        [](auto &v) { v.dot_admission = MoEGroupedDotAdmission::wideCandidate(
            v.second_descriptors ? MoEPrefillProjectionRole::GateUp : MoEPrefillProjectionRole::Down,
            v.n,v.n,v.k,MoEGroupedWmmaRowFloor::Sixteen); }
    };
    for (const auto &mutate : mutations)
    {
        nodes = captureCompactPopulationProbe(context,stream,[&] {
            Transaction owner(key,exact,gu.group_counts,gu.group_offsets,gu.device,stream);
            auto bad = gu; mutate(bad);
            EXPECT_FALSE(owner.gateUp(key.gateup_physical,bad));
            EXPECT_THROW((void)owner.gateUp(key.gateup_physical,gu),std::logic_error);
        });
        ASSERT_TRUE(nodes.empty());
        nodes = captureCompactPopulationProbe(context,stream,[&] {
            Transaction owner(key,exact,gu.group_counts,gu.group_offsets,gu.device,stream);
            auto produced = owner.gateUp(key.gateup_physical,gu); ASSERT_TRUE(produced);
            auto bad = down; mutate(bad);
            EXPECT_FALSE(owner.down(std::move(*produced),key.down_physical,bad));
            EXPECT_THROW((void)produced->receipt(),std::logic_error);
        });
        ASSERT_EQ(nodes.size(),2u); // Only the successful GU portion was recorded.
    }

    nodes = captureCompactPopulationProbe(context,stream,[&] {
        Transaction first(key,exact,gu.group_counts,gu.group_offsets,gu.device,stream);
        Transaction second(key,exact,gu.group_counts,gu.group_offsets,gu.device,stream);
        auto a = first.gateUp(key.gateup_physical,gu), b = second.gateUp(key.gateup_physical,gu);
        ASSERT_TRUE(a); ASSERT_TRUE(b);
        EXPECT_THROW((void)second.down(std::move(*a),key.down_physical,down),std::logic_error);
        EXPECT_THROW((void)first.down(std::move(*b),key.down_physical,down),std::logic_error);
        EXPECT_TRUE(first.down(std::move(*a),key.down_physical,down));
        EXPECT_TRUE(second.down(std::move(*b),key.down_physical,down));
    });
    ASSERT_EQ(nodes.size(),6u);

    nodes = captureCompactPopulationProbe(context,stream,[&] {
        std::optional<Transaction> owner; scope(owner,exact);
        const auto *address = &*owner;
        auto stale = owner->gateUp(key.gateup_physical,gu); ASSERT_TRUE(stale);
        owner.reset(); scope(owner,exact); ASSERT_EQ(&*owner,address);
        auto current = owner->gateUp(key.gateup_physical,gu); ASSERT_TRUE(current);
        EXPECT_THROW((void)owner->down(std::move(*stale),key.down_physical,down),std::logic_error);
        EXPECT_TRUE(owner->down(std::move(*current),key.down_physical,down));
    });
    ASSERT_EQ(nodes.size(),5u);

    Transaction crossed(key,exact,gu.group_counts,gu.group_offsets,gu.device,stream);
    std::optional<Transaction::GateUpSubmission> retained;
    nodes = captureCompactPopulationProbe(context,stream,[&] {
        auto produced = crossed.gateUp(key.gateup_physical,gu); ASSERT_TRUE(produced);
        retained.emplace(std::move(*produced));
    });
    ASSERT_EQ(nodes.size(),2u); ASSERT_TRUE(retained);
    nodes = captureCompactPopulationProbe(context,stream,[&] {
        EXPECT_THROW((void)crossed.down(std::move(*retained),key.down_physical,down),std::logic_error);
    });
    ASSERT_TRUE(nodes.empty());

    Transaction eager_down(key,exact,gu.group_counts,gu.group_offsets,gu.device,stream);
    std::optional<Transaction::GateUpSubmission> captured;
    nodes = captureCompactPopulationProbe(context,stream,[&] {
        auto produced = eager_down.gateUp(key.gateup_physical,gu); ASSERT_TRUE(produced);
        captured.emplace(std::move(*produced));
    });
    ASSERT_EQ(nodes.size(),2u); ASSERT_TRUE(captured);
    EXPECT_THROW((void)eager_down.down(std::move(*captured),key.down_physical,down),std::logic_error);
}
}
