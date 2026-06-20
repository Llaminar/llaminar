#include <gtest/gtest.h>

#include "collective/ITPContext.h"
#include "execution/local_execution/graph/IGraphBuilder.h"
#include "models/GraphTypes.h"

#include <map>
#include <string>
#include <type_traits>
#include <utility>

namespace llaminar2::test
{
    namespace
    {

        class MockTPContext final : public ITPContext
        {
        public:
            explicit MockTPContext(TPScope scope) : scope_(scope) {}

            TPScope scope() const override { return scope_; }
            int degree() const override { return scope_ == TPScope::LOCAL ? 2 : 3; }
            int myIndex() const override { return 0; }
            CollectiveBackendType backend() const override { return CollectiveBackendType::HOST; }
            bool allreduce(TensorBase *) override { return true; }
            bool broadcast(TensorBase *, int = 0) override { return true; }
            bool allgather(const TensorBase *, TensorBase *) override { return true; }

        private:
            TPScope scope_ = TPScope::LOCAL;
        };

        class RecordingGraphBuilder final : public IGraphBuilder
        {
        public:
            ComputeGraph buildForwardGraph(const ForwardInput &, ForwardOutput &) override { return {}; }
            ComputeGraph buildLayerGraph(const LayerContext &) override { return {}; }

            const GraphConfig &config() const override { return config_; }
            void setWeights(const ModelWeights &weights) override { weights_ = weights; }
            void setBuffers(const ModelBuffers &buffers) override { buffers_ = buffers; }
            const ModelBuffers &buffers() const override { return buffers_; }

            void setTPContext(const std::string &domain_name, ITPContext *tp_ctx) override
            {
                config_.domain_tp_contexts[domain_name] = tp_ctx;
            }

        private:
            GraphConfig config_;
            ModelWeights weights_;
            ModelBuffers buffers_;
        };

    } // namespace

    TEST(Test__MoEContinuationTPContextPlumbing, GraphConfigDomainMapIsPolymorphicITPContext)
    {
        using DomainMap = std::remove_reference_t<decltype(std::declval<GraphConfig &>().domain_tp_contexts)>;
        static_assert(std::is_same_v<DomainMap, std::map<std::string, ITPContext *>>,
                      "GraphConfig::domain_tp_contexts must expose polymorphic ITPContext pointers");
        static_assert(std::is_invocable_v<decltype(&IGraphBuilder::setTPContext),
                                          IGraphBuilder *, const std::string &, ITPContext *>,
                      "IGraphBuilder::setTPContext must accept ITPContext*");

        MockTPContext local_ctx(TPScope::LOCAL);
        MockTPContext global_ctx(TPScope::GLOBAL);

        GraphConfig config;
        config.domain_tp_contexts["local_dense"] = &local_ctx;
        config.domain_tp_contexts["global_dense"] = &global_ctx;

        ASSERT_EQ(config.domain_tp_contexts.at("local_dense"), &local_ctx);
        ASSERT_EQ(config.domain_tp_contexts.at("global_dense"), &global_ctx);
        EXPECT_TRUE(config.domain_tp_contexts.at("local_dense")->isLocal());
        EXPECT_TRUE(config.domain_tp_contexts.at("global_dense")->isGlobal());
    }

    TEST(Test__MoEContinuationTPContextPlumbing, GraphBuilderReceivesLocalAndGlobalITPContexts)
    {
        MockTPContext local_ctx(TPScope::LOCAL);
        MockTPContext global_ctx(TPScope::GLOBAL);
        RecordingGraphBuilder builder;

        builder.setTPContext("local_continuation", &local_ctx);
        builder.setTPContext("global_continuation", &global_ctx);

        const auto &contexts = builder.config().domain_tp_contexts;
        ASSERT_EQ(contexts.at("local_continuation"), &local_ctx);
        ASSERT_EQ(contexts.at("global_continuation"), &global_ctx);
        EXPECT_TRUE(contexts.at("local_continuation")->isLocal());
        EXPECT_FALSE(contexts.at("global_continuation")->isLocal());
        EXPECT_TRUE(contexts.at("global_continuation")->isGlobal());
    }

} // namespace llaminar2::test