/**
 * @file GPUGraphMemoryContractProof.h
 * @brief Symmetric native-node rejection proof for bounded GPU graph admission.
 *
 * Small helper reservations cover flat kernel/copy/memset metadata, not event
 * ownership, child graphs, or control flow. This model-free proof records an
 * actual event node and checks the production capture contract without ever
 * allocating an executable or launching that graph.
 */

#pragma once

#include "backends/IGPUGraphCapture.h"
#include "backends/IWorkerGPUContext.h"
#include <gtest/gtest.h>
#include <memory>
#include <string>

namespace llaminar2::test
{
    /**
     * @brief Reject an event node from the cheaper physical helper class.
     * @param context Owning GPU worker; caller executes on its worker thread.
     * @param record_external Backend recorder returning true on successful
     * external event capture, given the event and exact producer stream.
     *
     * The event outlives the native definition. No pending launch exists, so
     * cleanup needs no stream fence and cannot affect concurrent inference.
     */
    template <typename RecordExternalEvent>
    inline void proveBoundedHelperRejectsEventGraph(
        IWorkerGPUContext &context, RecordExternalEvent record_external)
    {
        void *const stream = context.defaultStream();
        ASSERT_NE(stream, nullptr);
        auto event_deleter = [&](void *event) { context.destroyEvent(event); };
        std::unique_ptr<void, decltype(event_deleter)> event(
            context.createEvent(), event_deleter);
        ASSERT_NE(event, nullptr);
        auto capture = context.createGraphCapture(stream);
        ASSERT_NE(capture, nullptr);
        ASSERT_TRUE(capture->beginCapture());
        // Ordinary captured events express dependency edges and may add no node.
        // External recording deliberately retains the event as native metadata,
        // which is the unsupported ownership class this negative proof exercises.
        ASSERT_TRUE(record_external(event.get(), stream));
        ASSERT_TRUE(capture->endCapture());
        ASSERT_GT(capture->nodeCount(), 0u);
        std::string error;
        EXPECT_FALSE(capture->validateExecutableMemoryClass(
            GPUGraphExecutableClass::BoundedFlatHelper, &error));
        EXPECT_NE(error.find("forbidden nested/control node"), std::string::npos)
            << error;
        EXPECT_TRUE(capture->validateExecutableMemoryClass(
            GPUGraphExecutableClass::General, &error));
        EXPECT_TRUE(error.empty());
        EXPECT_FALSE(capture->hasExecutable());
    }
} // namespace llaminar2::test
