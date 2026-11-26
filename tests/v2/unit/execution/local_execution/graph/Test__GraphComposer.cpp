/**
 * @file Test__GraphComposer.cpp
 * @brief Device-free contract and adversarial unit tests for GraphComposer,
 *        HeterogeneousOverlaySection, LocalMegakernelLowering, and typed ComputeGraph edges.
 *
 * Proves that invalid dependency wiring — duplicate nodes, missing producers,
 * foreign-graph handles, cycles, delaying an early CPU trigger behind a GPU
 * overlap branch, attaching manual host work to a follower graph, or inverting
 * post-CPU join edges — is rejected deterministically at cold graph construction
 * time without mutating the graph.
 */

#include <gtest/gtest.h>
#include "execution/local_execution/graph/ComputeGraph.h"
#include "execution/local_execution/graph/GraphComposer.h"
#include "execution/local_execution/graph/HeterogeneousOverlaySection.h"
#include "execution/local_execution/graph/LocalMegakernelLowering.h"
#include <stdexcept>
#include <string>

namespace llaminar2
{
namespace
{
    /** @brief Minimal device-free stage with configurable collective and manual-boundary contracts. */
    class ContractStage final : public IComputeStage
    {
    public:
        ContractStage(std::string stage_name,
                      DeviceId device,
                      bool collective = false,
                      bool manual_boundary = false)
            : IComputeStage(device),
              name_(std::move(stage_name)),
              collective_(collective),
              manual_boundary_(manual_boundary) {}

        bool execute(IDeviceContext *) override { return true; }
        std::string name() const override { return name_; }
        size_t estimatedFlops() const override { return 0; }
        StageBufferRequirements getBufferRequirements() const override { return {}; }
        ComputeStageType type() const override { return ComputeStageType::COPY; }
        bool isCollectiveStage() const override { return collective_; }
        bool isManualGraphBoundary() const override { return manual_boundary_; }
        bool isGraphCapturable() const override { return !manual_boundary_ && device().is_gpu(); }
        bool supportsBackend(ComputeBackendType) const override { return true; }
        StageDumpInfo buildDumpInfoImpl() const override { return {}; }

    private:
        std::string name_;
        bool collective_;
        bool manual_boundary_;
    };

    std::unique_ptr<IComputeStage> gpuStage(const std::string &name, DeviceId dev = DeviceId::cuda(0),
                                            bool collective = false)
    {
        return std::make_unique<ContractStage>(name, dev, collective, false);
    }

    std::unique_ptr<IComputeStage> manualCPUStage(const std::string &name)
    {
        return std::make_unique<ContractStage>(name, DeviceId::cpu(), false, true);
    }

    // =========================================================================
    // 1. GraphComposer typed declarations & fail-fast invariants
    // =========================================================================

    TEST(GraphComposerContract, DeclaresTypedEdgesAndRejectsInvalidConstruction)
    {
        ComputeGraph graph;
        GraphComposer composer(graph);
        const DeviceId gpu = DeviceId::cuda(0);

        const auto a = composer.add({.name = "a", .stage = gpuStage("a", gpu), .device = gpu});
        const auto b = composer.add({
            .name = "b",
            .stage = gpuStage("b", gpu),
            .device = gpu,
            .data_deps = {a},
        });
        const auto c = composer.add({
            .name = "c",
            .stage = gpuStage("c", gpu),
            .device = gpu,
            .data_deps = {a},
            .order_deps = {b},
        });
        const auto host = composer.add({
            .name = "host_dispatch",
            .stage = manualCPUStage("host_dispatch"),
            .device = DeviceId::cpu(),
            .data_deps = {a},
            .host_submission_deps = {c},
        });

        EXPECT_NO_THROW(graph.validateDeclarativeInvariants());
        EXPECT_TRUE(graph.hasDependency("c", "b"));
        EXPECT_TRUE(graph.hasPath("a", "host_dispatch"));
        EXPECT_FALSE(graph.hasPath("host_dispatch", "a"));

        const auto host_edges = graph.dependencyEdges(host.name());
        ASSERT_EQ(host_edges.size(), 2u);
        EXPECT_EQ(host_edges[0], (GraphDependencyEdge{"a", GraphDependencyKind::DataFlow}));
        EXPECT_EQ(host_edges[1], (GraphDependencyEdge{"c", GraphDependencyKind::HostSubmissionOrder}));

        // Duplicate node names must fail hard instead of silently overwriting.
        EXPECT_THROW(
            composer.add({.name = "b", .stage = gpuStage("b2", gpu), .device = gpu}),
            std::invalid_argument);

        // Foreign graph handle must fail hard.
        ComputeGraph other;
        GraphComposer other_composer(other);
        const auto foreign = other_composer.add({.name = "x", .stage = gpuStage("x", gpu), .device = gpu});
        EXPECT_THROW(
            composer.add({.name = "d", .stage = gpuStage("d", gpu), .device = gpu, .data_deps = {foreign}}),
            std::invalid_argument);

        // HostSubmissionOrder targeting a captured GPU stage must fail hard.
        EXPECT_THROW(
            composer.add({
                .name = "bad_gpu_consumer",
                .stage = gpuStage("bad_gpu_consumer", gpu),
                .device = gpu,
                .host_submission_deps = {c},
            }),
            std::invalid_argument);

        // Cycle introduction via connect() must fail hard without mutating the graph.
        const auto gen_before = graph.topologyGeneration();
        EXPECT_THROW(composer.connect(a, c, GraphDependencyKind::DataFlow), std::invalid_argument);
        EXPECT_EQ(graph.topologyGeneration(), gen_before);
        EXPECT_FALSE(graph.hasDependency("a", "c"));
    }

    // =========================================================================
    // 2. HeterogeneousOverlaySection structural invariants (authority + follower)
    // =========================================================================

    TEST(GraphComposerContract, HeterogeneousOverlayAuthorityAndFollowerSealCleanly)
    {
        const DeviceId gpu0 = DeviceId::cuda(0);
        const DeviceId gpu1 = DeviceId::cuda(1);

        // Authority graph: router -> early_trigger; local_experts + shared_ffn overlap CPU;
        // pre_cpu_cutpoint closes producer unit -> HostSubmissionOrder -> host_dispatch -> post_cpu_join.
        {
            ComputeGraph authority;
            authority.setNativeCaptureEnvelope(
                GraphNativeCaptureEnvelope::HeterogeneousTicketAuthorityTransaction);
            GraphComposer c(authority);

            const auto router = c.add({.name = "router", .stage = gpuStage("router", gpu0), .device = gpu0});
            const auto trigger = c.add({
                .name = "ticket_publish",
                .stage = gpuStage("ticket_publish", gpu0),
                .device = gpu0,
                .data_deps = {router},
            });
            const auto local_exp = c.add({
                .name = "local_experts",
                .stage = gpuStage("local_experts", gpu0),
                .device = gpu0,
                .data_deps = {router},
                .order_deps = {trigger},
            });
            const auto shared_ffn = c.add({
                .name = "shared_ffn",
                .stage = gpuStage("shared_ffn", gpu0),
                .device = gpu0,
                .data_deps = {router},
            });
            const auto host_dispatch = c.add({
                .name = "cpu_dispatch",
                .stage = manualCPUStage("cpu_dispatch"),
                .device = DeviceId::cpu(),
                .data_deps = {trigger},
            });
            const auto cpu_return = c.add({
                .name = "cpu_return",
                .stage = manualCPUStage("cpu_return"),
                .device = DeviceId::cpu(),
                .data_deps = {host_dispatch},
            });
            const auto ticket_consume = c.add({
                .name = "ticket_consume",
                .stage = gpuStage("ticket_consume", gpu0),
                .device = gpu0,
                .data_deps = {cpu_return, local_exp},
            });
            const auto ordered_reduce = c.add({
                .name = "ordered_reduce",
                .stage = gpuStage("ordered_reduce", gpu0),
                .device = gpu0,
                .data_deps = {ticket_consume, local_exp},
            });

            ASSERT_NO_THROW(sealHeterogeneousOverlayTicketSection(authority, {
                .router = router,
                .early_trigger = trigger,
                .pre_cpu_gpu_cutpoint = local_exp,
                .ticket_unit_identity = "layer0_pre_cpu_ticket_unit",
                .pre_cpu_overlap_branches = {shared_ffn},
                .manual_host_entry = host_dispatch,
                .post_cpu_join = ordered_reduce,
            }));

            EXPECT_NO_THROW(authority.validateDeclarativeInvariants());
            EXPECT_TRUE(authority.hasDependency("shared_ffn", "ticket_publish"));
            EXPECT_TRUE(authority.hasDependency("local_experts", "shared_ffn"));
            EXPECT_TRUE(authority.hasDependency("cpu_dispatch", "local_experts"));
            ASSERT_TRUE(authority.getNode("local_experts")->heterogeneous_ticket_unit_contract.has_value());
            EXPECT_EQ(
                authority.getNode("local_experts")->heterogeneous_ticket_unit_contract->identity,
                "layer0_pre_cpu_ticket_unit");
            EXPECT_EQ(
                authority.getNode("local_experts")->heterogeneous_ticket_unit_contract->disposition,
                GraphHeterogeneousTicketUnitDisposition::BeforeManualBoundary);
            // Early trigger must NEVER wait on shared_ffn!
            EXPECT_FALSE(authority.hasPath("shared_ffn", "ticket_publish"));
        }

        // Follower graph: symmetric cutpoint, NO early trigger and NO manual host work.
        {
            ComputeGraph follower;
            follower.setNativeCaptureEnvelope(
                GraphNativeCaptureEnvelope::HeterogeneousTicketFollowerTransaction);
            GraphComposer c(follower);

            const auto router = c.add({.name = "router", .stage = gpuStage("router", gpu1), .device = gpu1});
            const auto local_exp = c.add({
                .name = "local_experts",
                .stage = gpuStage("local_experts", gpu1),
                .device = gpu1,
                .data_deps = {router},
            });
            const auto shared_ffn = c.add({
                .name = "shared_ffn",
                .stage = gpuStage("shared_ffn", gpu1),
                .device = gpu1,
                .data_deps = {router},
            });
            const auto ordered_reduce = c.add({
                .name = "ordered_reduce",
                .stage = gpuStage("ordered_reduce", gpu1),
                .device = gpu1,
                .data_deps = {local_exp},
            });

            ASSERT_NO_THROW(sealHeterogeneousOverlayTicketSection(follower, {
                .router = router,
                .early_trigger = std::nullopt,
                .pre_cpu_gpu_cutpoint = local_exp,
                .ticket_unit_identity = "layer0_pre_cpu_ticket_unit",
                .pre_cpu_overlap_branches = {shared_ffn},
                .manual_host_entry = std::nullopt,
                .post_cpu_join = ordered_reduce,
            }));

            EXPECT_NO_THROW(follower.validateDeclarativeInvariants());
            EXPECT_TRUE(follower.hasDependency("local_experts", "shared_ffn"));
        }
    }

    TEST(GraphComposerContract, RejectsDelayedCPUTriggerAndFollowerManualWork)
    {
        const DeviceId gpu0 = DeviceId::cuda(0);

        // Defect regression: delaying early_trigger behind shared_ffn/shared_gate MUST fail hard!
        {
            ComputeGraph graph;
            graph.setNativeCaptureEnvelope(
                GraphNativeCaptureEnvelope::HeterogeneousTicketAuthorityTransaction);
            GraphComposer c(graph);

            const auto router = c.add({.name = "router", .stage = gpuStage("router", gpu0), .device = gpu0});
            const auto shared_ffn = c.add({
                .name = "shared_ffn",
                .stage = gpuStage("shared_ffn", gpu0),
                .device = gpu0,
                .data_deps = {router},
            });
            // Miswired: ticket_publish waits on shared_ffn (delays CPU wake-up!).
            const auto bad_trigger = c.add({
                .name = "ticket_publish",
                .stage = gpuStage("ticket_publish", gpu0),
                .device = gpu0,
                .data_deps = {router, shared_ffn},
            });
            const auto local_exp = c.add({
                .name = "local_experts",
                .stage = gpuStage("local_experts", gpu0),
                .device = gpu0,
                .data_deps = {router},
            });
            const auto host_dispatch = c.add({
                .name = "cpu_dispatch",
                .stage = manualCPUStage("cpu_dispatch"),
                .device = DeviceId::cpu(),
                .data_deps = {bad_trigger},
            });

            const auto gen_before = graph.topologyGeneration();
            EXPECT_THROW(
                sealHeterogeneousOverlayTicketSection(graph, {
                    .router = router,
                    .early_trigger = bad_trigger,
                    .pre_cpu_gpu_cutpoint = local_exp,
                    .ticket_unit_identity = "layer0_pre_cpu_ticket_unit",
                    .pre_cpu_overlap_branches = {shared_ffn},
                    .manual_host_entry = host_dispatch,
                }),
                std::logic_error);
            EXPECT_EQ(graph.topologyGeneration(), gen_before);
        }

        // Defect regression: follower graph with manual_host_entry MUST fail hard!
        {
            ComputeGraph follower;
            follower.setNativeCaptureEnvelope(
                GraphNativeCaptureEnvelope::HeterogeneousTicketFollowerTransaction);
            GraphComposer c(follower);
            const auto router = c.add({.name = "router", .stage = gpuStage("router", gpu0), .device = gpu0});
            const auto local_exp = c.add({
                .name = "local_experts",
                .stage = gpuStage("local_experts", gpu0),
                .device = gpu0,
                .data_deps = {router},
            });
            const auto host_dispatch = c.add({
                .name = "cpu_dispatch",
                .stage = manualCPUStage("cpu_dispatch"),
                .device = DeviceId::cpu(),
                .data_deps = {router},
            });

            EXPECT_THROW(
                sealHeterogeneousOverlayTicketSection(follower, {
                    .router = router,
                    .early_trigger = std::nullopt,
                    .pre_cpu_gpu_cutpoint = local_exp,
                    .ticket_unit_identity = "layer0_pre_cpu_ticket_unit",
                    .manual_host_entry = host_dispatch,
                }),
                std::logic_error);
        }

        // Defect regression: manual_host_entry waiting on post_cpu_join (e.g. shared_gate after AR) MUST fail!
        {
            ComputeGraph graph;
            graph.setNativeCaptureEnvelope(
                GraphNativeCaptureEnvelope::HeterogeneousTicketAuthorityTransaction);
            GraphComposer c(graph);
            const auto router = c.add({.name = "router", .stage = gpuStage("router", gpu0), .device = gpu0});
            const auto local_exp = c.add({
                .name = "local_experts",
                .stage = gpuStage("local_experts", gpu0),
                .device = gpu0,
                .data_deps = {router},
            });
            const auto post_join = c.add({
                .name = "shared_gate",
                .stage = gpuStage("shared_gate", gpu0),
                .device = gpu0,
                .data_deps = {local_exp},
            });
            const auto host_dispatch = c.add({
                .name = "cpu_dispatch",
                .stage = manualCPUStage("cpu_dispatch"),
                .device = DeviceId::cpu(),
                .data_deps = {post_join},
            });

            EXPECT_THROW(
                sealHeterogeneousOverlayTicketSection(graph, {
                    .router = router,
                    .pre_cpu_gpu_cutpoint = local_exp,
                    .ticket_unit_identity = "layer0_pre_cpu_ticket_unit",
                    .manual_host_entry = host_dispatch,
                    .post_cpu_join = post_join,
                }),
                std::logic_error);
        }
    }

    // =========================================================================
    // 3. LocalMegakernelLowering automatic export & capture-policy preservation
    // =========================================================================

    TEST(GraphComposerContract, LocalMegakernelLoweringPreservesExportsAndTicketCutpoint)
    {
        const DeviceId gpu = DeviceId::rocm(0);
        ComputeGraph graph;
        graph.setNativeCaptureEnvelope(
            GraphNativeCaptureEnvelope::HeterogeneousTicketAuthorityTransaction);
        GraphComposer c(graph);

        const auto prev_ar = c.add({.name = "prev_ar", .stage = gpuStage("prev_ar", gpu, true), .device = gpu});
        const auto norm = c.add({.name = "norm", .stage = gpuStage("norm", gpu), .device = gpu, .data_deps = {prev_ar}});
        const auto proj = c.add({.name = "proj", .stage = gpuStage("proj", gpu), .device = gpu, .data_deps = {norm}});
        const auto conv = c.add({.name = "conv", .stage = gpuStage("conv", gpu), .device = gpu, .data_deps = {proj}});
        const auto out = c.add({
            .name = "out_proj",
            .stage = gpuStage("out_proj", gpu),
            .device = gpu,
            .data_deps = {conv},
            .ticket_unit = GraphHeterogeneousTicketUnitContract{
                .identity = "layer0_unit",
                .disposition = GraphHeterogeneousTicketUnitDisposition::BeforeManualBoundary,
            },
        });
        // Independent outside reader of interior `norm` (mirrors hc_injection!)
        const auto side_reader = c.add({
            .name = "hc_injection",
            .stage = gpuStage("hc_injection", gpu),
            .device = gpu,
            .data_deps = {norm},
        });
        const auto next_ar = c.add({
            .name = "next_ar",
            .stage = gpuStage("next_ar", gpu, true),
            .device = gpu,
            .data_deps = {out},
        });

        const auto fused = lowerLocalMegakernelRegion(graph, {
            .members = {norm, proj, conv, out},
            .terminal = out,
            .replacement = gpuStage("gdn_megakernel", gpu),
        });

        EXPECT_EQ(fused.name(), "out_proj");
        EXPECT_EQ(graph.size(), 4u); // prev_ar, out_proj (megakernel), hc_injection, next_ar
        EXPECT_TRUE(graph.hasDependency("out_proj", "prev_ar"));
        EXPECT_TRUE(graph.hasDependency("hc_injection", "out_proj"));
        EXPECT_TRUE(graph.hasDependency("next_ar", "out_proj"));
        ASSERT_TRUE(graph.getNode("out_proj")->heterogeneous_ticket_unit_contract.has_value());
        EXPECT_EQ(
            graph.getNode("out_proj")->heterogeneous_ticket_unit_contract->identity,
            "layer0_unit");
        EXPECT_NO_THROW(graph.validateDeclarativeInvariants());

        // Attempting to absorb a collective stage into a local megakernel must fail cleanly.
        const auto gen_before = graph.topologyGeneration();
        EXPECT_THROW(
            lowerLocalMegakernelRegion(graph, {
                .members = {fused, next_ar},
                .terminal = next_ar,
                .replacement = gpuStage("bad_fused_ar", gpu),
            }),
            std::invalid_argument);
        EXPECT_EQ(graph.topologyGeneration(), gen_before);
        EXPECT_NE(graph.getNode(side_reader.name()), nullptr);
    }

} // namespace
} // namespace llaminar2
