/**
 * @file Test__EarlyCurrentLookup.cpp
 * @brief Device-free replay of the production graph-local lookup construction contract.
 *
 * Native source publication and sampled-token parent readiness remain explicit
 * integration obligations. This test never pretends its inert boundary is an
 * actual provider, executes a GPU operation or reads a model/request value.
 */
#include "execution/compute_stages/stages/qwen4exp/GraphLocalLookupFrame.h"
#include "mocks/MockComputeStage.h"
#include <gtest/gtest.h>
#include <algorithm>
using namespace llaminar2;
namespace {
/** @brief Inert boundary identity for construction-only rejection/ownership tests. */
class ConstructionBoundary final : public IRetainedGraphStorageBoundary {
public:
    DeviceId device() const noexcept override { return DeviceId::cuda(0); }
    std::uint64_t captureIdentity() const noexcept override { return 17; }
    void bindGenerationControl(RetainedStorageGenerationControl) override { throw std::logic_error("No native service"); }
    /** @brief Reject fabricated retirement for a construction-only identity. */
    void retireGenerationControl(RetainedStorageGenerationControl) override { throw std::logic_error("No native service"); }
    RetainedStorageGenerationControl generationControl() const override { return {}; }
    void beginAdmission(std::uint64_t,void*) override { throw std::logic_error("No native service"); }
    bool hasActiveInterval() const noexcept override { return false; }
    void requireArmedForLaunch() const override { throw std::logic_error("No native service"); }
    void recordTerminalClose(void*,const int*,const int*) const override { throw std::logic_error("No native service"); }
    void awaitTerminal() override { throw std::logic_error("No native service"); }
    RetainedStorageTerminalObservation terminalObservation() const override { throw std::logic_error("No native service"); }
};
/** @brief Declare a dependency-only node; no runtime work is submitted. */
void node(ComputeGraph& graph,const std::string& name){graph.addNode(name,
    std::make_unique<llaminar2::testing::MockComputeStage>(),DeviceId::cpu());}
}
/** @brief The consumed production handoff supplies both Publish and actual trunk dependencies. */
TEST(EarlyCurrentLookup, GraphHandoffRepeatedResetAndAdversarialInsertionReplay){
    for(unsigned replay=0;replay<20;++replay){
        ComputeGraph graph;graph.setRetainedStorageBoundary(std::make_shared<ConstructionBoundary>());
        for(const auto* name:{"consume","kv","trunk","publish","hash","query"})node(graph,name);
        graph.addDependency("publish","hash");graph.addDependency("trunk","publish");graph.addDependency("query","trunk");
        GraphLocalLookupFrame frame(graph,"publish",100+replay,200+replay);
        EXPECT_THROW(frame.requireConsumed(),std::logic_error);
        const auto dependencies=frame.consume(graph,100+replay,200+replay,{"trunk"});
        ASSERT_EQ(dependencies.size(),2u);
        for(const auto& edge:dependencies)graph.addDependency("consume",edge);
        graph.addDependency("kv","consume");
        const auto order=graph.getExecutionOrder();
        const auto at=[&](const char* name){return std::find(order.begin(),order.end(),name)-order.begin();};
        EXPECT_LT(at("hash"),at("publish"));EXPECT_LT(at("publish"),at("trunk"));
        EXPECT_LT(at("trunk"),at("consume"));EXPECT_LT(at("consume"),at("kv"));EXPECT_LT(at("trunk"),at("query"));
        EXPECT_NO_THROW(frame.requireConsumed());
        EXPECT_THROW(frame.consume(graph,100+replay,200+replay,{"trunk"}),std::invalid_argument);
    }
}
/** @brief Rejected construction does not consume the original capability or authorize another graph. */
TEST(EarlyCurrentLookup, GraphHandoffRejectsForeignQuerySourceAndIncompleteTrunk){
    ComputeGraph graph,foreign;
    graph.setRetainedStorageBoundary(std::make_shared<ConstructionBoundary>());
    foreign.setRetainedStorageBoundary(std::make_shared<ConstructionBoundary>());
    for(auto* g:{&graph,&foreign}){node(*g,"publish");node(*g,"trunk");}
    GraphLocalLookupFrame frame(graph,"publish",101,202);
    EXPECT_THROW(frame.consume(foreign,101,202,{"trunk"}),std::invalid_argument);
    EXPECT_THROW(frame.consume(graph,102,202,{"trunk"}),std::invalid_argument);
    EXPECT_THROW(frame.consume(graph,101,203,{"trunk"}),std::invalid_argument);
    EXPECT_THROW(frame.consume(graph,101,202,{}),std::invalid_argument);
    EXPECT_THROW(frame.consume(graph,101,202,{"missing"}),std::invalid_argument);
    EXPECT_THROW(frame.consume(graph,101,202,{"publish"}),std::invalid_argument);
    EXPECT_THROW(GraphLocalLookupFrame(graph,"missing",101,202),std::invalid_argument);
    EXPECT_NO_THROW(frame.consume(graph,101,202,{"trunk"}));
    EXPECT_NO_THROW(frame.requireConsumed());
}
