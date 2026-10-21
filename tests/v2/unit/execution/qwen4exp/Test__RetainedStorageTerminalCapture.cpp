/**
 * @file Test__RetainedStorageTerminalCapture.cpp
 * @brief Device-free proof of exact stream retention during storage graph teardown.
 *
 * Native graph destruction observes a weak stream lifetime, so the regression
 * fails if graph/stream destruction order regresses. Releasing the generation
 * owner's original stream models the formerly unsafe topology-release edge.
 */
#include "execution/local_execution/graph/RetainedStorageTerminalCapture.h"
#include <gtest/gtest.h>
#include <memory>
#include <vector>
using namespace llaminar2;
namespace
{
/** @brief Native-shaped graph whose destruction needs its still-live exact stream. */
class BorrowedGraph final : public IGPUGraphCapture
{
public:
    /** @brief Capture an opaque stream lifetime without retaining it inside the test graph. */
    BorrowedGraph(const std::shared_ptr<void> &stream,std::vector<int> &trace):stream_(stream),address_(stream.get()),trace_(&trace){}
    /** @brief Independent lifetime oracle: native destruction must precede stream destruction. */
    ~BorrowedGraph() override{EXPECT_FALSE(stream_.expired());trace_->push_back(1);}
    /** @brief Exact immutable borrowed stream. */
    void *executionStream() const noexcept override{return address_;}
    /** @brief Unused device-free construction operation. */
    bool beginCapture() override{return false;}
    /** @brief Unused device-free construction operation. */
    bool endCapture() override{return false;}
    /** @brief Unit never materializes a native executable. */
    bool instantiate() override{return false;}
    /** @brief Unit never launches device work. */
    bool launch() override{return false;}
    /** @brief Topology cannot update in this lifetime-only oracle. */
    GraphUpdateResult tryUpdate() override{return GraphUpdateResult::Failed;}
    /** @brief No executable update capability is advertised. */
    bool supportsExecutableUpdate() const noexcept override{return false;}
    /** @brief No native executable exists. */
    bool hasExecutable() const override{return false;}
    /** @brief No device memory is owned. */
    std::size_t residentMemoryBytes() const noexcept override{return 0;}
    /** @brief One diagnostic node suffices for immutable identity. */
    std::size_t nodeCount() const override{return 1;}
    /** @brief No device state is touched during reset. */
    void reset() override{}
    /** @brief Diagnostic fixture backend. */
    const char *backendName() const override{return "lifetime-unit";}
private:
    std::weak_ptr<void> stream_;void *address_;std::vector<int> *trace_;
};
}
/** @test Parent stream ownership can retire first without dangling terminal graph destruction. */
TEST(RetainedStorageTerminalCapture, ExactStreamOutlivesTerminalGraphAfterParentRetires)
{
    std::vector<int> trace;
    auto stream=std::shared_ptr<void>(new int(7),[&](void *p){trace.push_back(2);delete static_cast<int *>(p);});
    std::weak_ptr<void> weak=stream;
    RetainedStorageTerminalCapture terminal;terminal.bind(stream,std::make_unique<BorrowedGraph>(stream,trace));
    stream.reset();EXPECT_FALSE(weak.expired());EXPECT_TRUE(trace.empty());
    terminal.reset();EXPECT_TRUE(weak.expired());EXPECT_EQ(trace,(std::vector<int>{1,2}));
    terminal.reset();EXPECT_EQ(trace.size(),2u);
}
/** @test A recording cannot silently borrow an unrelated stream or replace live topology. */
TEST(RetainedStorageTerminalCapture, RejectsWrongStreamAndLiveReplacement)
{
    std::vector<int> trace;auto first=std::make_shared<int>(1),second=std::make_shared<int>(2);
    RetainedStorageTerminalCapture terminal;
    EXPECT_THROW(terminal.bind(second,std::make_unique<BorrowedGraph>(first,trace)),std::invalid_argument);
    terminal.bind(first,std::make_unique<BorrowedGraph>(first,trace));
    EXPECT_THROW(terminal.bind(first,std::make_unique<BorrowedGraph>(first,trace)),std::invalid_argument);
    EXPECT_EQ(terminal->executionStream(),first.get());
}
/** @test Moving empty/live owners preserves exact borrowed stream lifetime and retires replaced graphs first. */
TEST(RetainedStorageTerminalCapture, MoveRetiresEachGraphBeforeItsOwnStream)
{
    RetainedStorageTerminalCapture empty,movedEmpty(std::move(empty));
    EXPECT_FALSE(empty);EXPECT_FALSE(movedEmpty);
    std::vector<int> firstTrace,secondTrace;
    auto first=std::shared_ptr<void>(new int(1),[&](void *p){firstTrace.push_back(2);delete static_cast<int *>(p);});
    auto second=std::shared_ptr<void>(new int(2),[&](void *p){secondTrace.push_back(2);delete static_cast<int *>(p);});
    RetainedStorageTerminalCapture source,target;
    source.bind(first,std::make_unique<BorrowedGraph>(first,firstTrace));
    target.bind(second,std::make_unique<BorrowedGraph>(second,secondTrace));
    first.reset();second.reset();
    RetainedStorageTerminalCapture moved(std::move(source));
    EXPECT_FALSE(source);EXPECT_TRUE(moved);EXPECT_TRUE(firstTrace.empty());
    target=std::move(moved);EXPECT_FALSE(moved);EXPECT_TRUE(target);
    EXPECT_EQ(secondTrace,(std::vector<int>{1,2}));EXPECT_TRUE(firstTrace.empty());
    target=std::move(target);EXPECT_TRUE(target);EXPECT_TRUE(firstTrace.empty());
    movedEmpty=std::move(target);EXPECT_FALSE(target);EXPECT_TRUE(movedEmpty);
    movedEmpty.reset();EXPECT_EQ(firstTrace,(std::vector<int>{1,2}));
    movedEmpty=RetainedStorageTerminalCapture{};EXPECT_FALSE(movedEmpty);
}
