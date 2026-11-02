/**
 * @file PooledNativeProgramTerminalOrder.h
 * @brief The actual provider's one terminal enqueue/join/observation order.
 *
 * Recording an event queues a new operation even on an otherwise empty stream.
 * The existing parent join must follow that publication before readiness is
 * observed. This fixture helper adds neither a synchronization nor a poll.
 */
#pragma once
#include <exception>
namespace llaminar2::test {
/**
 * @brief Retire one original request using its existing native/storage boundary.
 * @param original Actual fixture parent, whose terminal graph was just enqueued.
 * @param side Original lane authority, or the explicitly ordinary fixture's no-op authority.
 * @throws The original storage error after the already joined GPU readers retire.
 */
template<class OriginalParent,class OriginalSide>
void retireOriginalProviderRequest(OriginalParent& original,OriginalSide& side){
    side.publishRequestTerminal();
    original.join();
    try{original.boundary->awaitTerminal();}
    catch(...){
        const auto failure=std::current_exception();
        // GPU completion is already proven by the SAME existing parent join;
        // a storage error cannot leave its separate GPU ownership armed.
        try{side.observeRequestTerminal();}catch(...){std::terminate();}
        std::rethrow_exception(failure);
    }
    side.observeRequestTerminal();
}
}
