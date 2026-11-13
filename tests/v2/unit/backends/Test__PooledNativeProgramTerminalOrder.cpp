/**
 * @file Test__PooledNativeProgramTerminalOrder.cpp
 * @brief Replay the consumed terminal helper with an actually asynchronous marker model.
 */
#include "fixtures/backends/PooledNativeProgramTerminalOrder.h"
#include <gtest/gtest.h>
#include <stdexcept>
#include <string>
#include <vector>
namespace llaminar2::test {namespace {
/** @brief Device-free SDK model: publication stays pending until the next original join. */
struct Queue {
    unsigned submitted=0,completed=0,joins=0;bool storageFailure=false;
    std::vector<std::string> calls;
};
struct Boundary {
    Queue& queue;
    void awaitTerminal(){queue.calls.push_back("storage await");if(queue.storageFailure)throw std::runtime_error("original storage failure");}
};
struct Parent {
    Queue& queue;Boundary* boundary;
    void join(){queue.calls.push_back("original join");++queue.joins;queue.completed=queue.submitted;}
};
struct Side {
    Queue& queue;
    void publishRequestTerminal(){queue.calls.push_back("record exact marker");++queue.submitted;}
    void observeRequestTerminal(){queue.calls.push_back("observe exact marker");if(queue.completed!=queue.submitted)throw std::logic_error("new marker still pending");}
};
}
TEST(PooledNativeProgramTerminalOrder, OriginalJoinCompletesNewMarkerAcrossResetAndRepeat){
    Queue queue;Boundary boundary{queue};Parent parent{queue,&boundary};Side side{queue};
    for(unsigned epoch=0;epoch<20;++epoch){queue.calls.clear();retireOriginalProviderRequest(parent,side);
        EXPECT_EQ(queue.calls,(std::vector<std::string>{"record exact marker","original join","storage await","observe exact marker"}));
        EXPECT_EQ(queue.joins,epoch+1);EXPECT_EQ(queue.submitted,queue.completed);
    }
}
TEST(PooledNativeProgramTerminalOrder, FormerPostJoinPublicationLeavesTheNewMarkerPending){
    Queue queue;Boundary boundary{queue};Parent parent{queue,&boundary};Side side{queue};
    parent.join();boundary.awaitTerminal();side.publishRequestTerminal();
    EXPECT_THROW(side.observeRequestTerminal(),std::logic_error);EXPECT_LT(queue.completed,queue.submitted);
}
TEST(PooledNativeProgramTerminalOrder, OriginalStorageFailureStillRetiresAlreadyJoinedGpuReaders){
    Queue queue;queue.storageFailure=true;Boundary boundary{queue};Parent parent{queue,&boundary};Side side{queue};
    try{retireOriginalProviderRequest(parent,side);FAIL();}
    catch(const std::runtime_error& failure){EXPECT_EQ(std::string(failure.what()),"original storage failure");}
    EXPECT_EQ(queue.calls,(std::vector<std::string>{"record exact marker","original join","storage await","observe exact marker"}));
    EXPECT_EQ(queue.joins,1u);EXPECT_EQ(queue.submitted,queue.completed);
}
}
