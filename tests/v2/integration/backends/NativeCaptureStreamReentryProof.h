/**
 * @file NativeCaptureStreamReentryProof.h
 * @brief Acyclic captured work may revisit streams whose membership graph cycles.
 *
 * A publishes to B, B publishes to C, then C publishes back to A. These
 * timestamped dependencies are a linear operation DAG, although the runtime's
 * stream-membership edges form A->B->C->A. Native EndCapture must retire each
 * membership before following its links so repeated visits terminate.
 * The same original worker, streams and events survive repeated capture/reset
 * and changing-input replay. Raw copies are the test infrastructure boundary;
 * their persistent device bytes have one explicit physical-memory claim.
 */
#pragma once
#include "backends/IBackend.h"
#include "backends/IGPUGraphCapture.h"
#include "backends/IWorkerGPUContext.h"
#include "planning/PhysicalMemoryAuthority.h"
#include <gtest/gtest.h>
#include <array>
#include <memory>
#include <stdexcept>
#include <vector>

namespace llaminar2::test
{
    /**
     * @brief Reenter a non-origin ancestor, then join the original capture stream.
     * @param context Original GPU worker; caller executes on its owning thread.
     * @param backend Matching native allocation/copy authority.
     * @param device Exact original physical device, never a synthetic participant.
     * Test-only terminal observations finish before graphs, events, or payload
     * storage retire; no synchronization is inserted inside captured execution.
     */
    inline void proveNativeCaptureStreamReentry(IWorkerGPUContext& context,
        IBackend& backend, DeviceId device)
    {
        ASSERT_TRUE(context.ownsCurrentThread());
        ASSERT_EQ(context.deviceOrdinal(),device.ordinal);
        constexpr size_t extent=4096,regions=7;
        PhysicalMemoryPlanBuilder builder;
        builder.add({0,device,backend.deviceMemoryTotal(device.ordinal),
            backend.deviceMemoryFree(device.ordinal)},PhysicalMemoryOwner::ExecutionWorkspace,regions*extent);
        auto memory=std::make_shared<PhysicalMemoryAuthority>(
            std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(builder.build()),0);
        {
            auto claim=memory->claimNewAllocation(device,PhysicalMemoryOwner::ExecutionWorkspace,regions*extent);
            const auto release=[&](void* pointer){if(pointer)backend.free(pointer,device.ordinal);};
            std::unique_ptr<void,decltype(release)> storage(backend.allocate(regions*extent,device.ordinal),release);
            ASSERT_NE(storage,nullptr);
            auto* bytes=static_cast<unsigned char*>(storage.get());
            const std::array streams{context.defaultStream(),
                context.getOrCreateAuxiliaryStream("capture_reentry_B"),
                context.getOrCreateAuxiliaryStream("capture_reentry_C"),
                context.getOrCreateAuxiliaryStream("capture_reentry_D")};
            for(size_t i=0;i<streams.size();++i){
                ASSERT_NE(streams[i],nullptr);
                for(size_t j=0;j<i;++j)ASSERT_NE(streams[i],streams[j]);
            }
            const auto destroy=[&](void* event){if(event)context.destroyEvent(event);};
            std::array<std::unique_ptr<void,decltype(destroy)>,5> events{
                std::unique_ptr<void,decltype(destroy)>(context.createEvent(),destroy),
                std::unique_ptr<void,decltype(destroy)>(context.createEvent(),destroy),
                std::unique_ptr<void,decltype(destroy)>(context.createEvent(),destroy),
                std::unique_ptr<void,decltype(destroy)>(context.createEvent(),destroy),
                std::unique_ptr<void,decltype(destroy)>(context.createEvent(),destroy)};
            for(const auto& event:events)ASSERT_NE(event,nullptr);
            auto capture=context.createGraphCapture(streams[0]);
            ASSERT_NE(capture,nullptr);
            // Host oracle buffers also outlive any accepted diagnostic D2H.
            std::vector<unsigned char> expected(extent),actual(extent);
            struct TerminalRetirement {
                IWorkerGPUContext& context;
                const std::array<void*,4>& streams;
                /** @brief Assertion unwinding still drains original readers before storage. */
                ~TerminalRetirement(){for(void* stream:streams)EXPECT_TRUE(context.synchronizeStreamChecked(stream));}
            } terminal{context,streams};
            const auto copy=[&](size_t to,size_t from,void* stream){
                if(!backend.deviceCopyAsync(bytes+to*extent,bytes+from*extent,extent,device.ordinal,stream))
                    throw std::runtime_error("Stream reentry proof D2D failed");
            };
            for(int generation=0;generation<3;++generation){
                SCOPED_TRACE(generation);
                ASSERT_TRUE(capture->beginCapture());
                copy(1,0,streams[0]);
                context.recordEvent(events[0].get(),streams[0]);
                context.waitEvent(events[0].get(),streams[1]);
                copy(2,1,streams[1]);
                context.recordEvent(events[1].get(),streams[1]);
                context.waitEvent(events[1].get(),streams[2]);
                copy(3,2,streams[2]);
                context.recordEvent(events[2].get(),streams[2]);
                context.waitEvent(events[2].get(),streams[3]);
                copy(4,3,streams[3]);
                context.recordEvent(events[3].get(),streams[3]);
                // The returning stream is non-origin and is not the direct
                // parent of streams[3]. HIP therefore adds 3->1 while retaining
                // 1->2->3. The captured operation DAG still only points forward.
                context.waitEvent(events[3].get(),streams[1]);
                copy(5,4,streams[1]);
                context.recordEvent(events[4].get(),streams[1]);
                // Join the actual origin only after all descendants finish.
                context.waitEvent(events[4].get(),streams[0]);
                copy(6,5,streams[0]);
                ASSERT_TRUE(capture->endCapture());
                ASSERT_EQ(capture->nodeCount(),6u);
                ASSERT_TRUE(capture->instantiate());
                for(int replay=0;replay<20;++replay){
                    SCOPED_TRACE(replay);
                    for(size_t i=0;i<extent;++i)expected[i]=static_cast<unsigned char>((i*31+replay*17+generation*59)%251);
                    ASSERT_TRUE(backend.memset(bytes+extent,0xff,6*extent,device.ordinal,streams[0]));
                    ASSERT_TRUE(backend.hostToDeviceOnStream(bytes,expected.data(),extent,device.ordinal,streams[0]));
                    ASSERT_TRUE(capture->launch());
                    ASSERT_TRUE(backend.deviceToHostOnStream(actual.data(),bytes+6*extent,extent,device.ordinal,streams[0]));
                    ASSERT_TRUE(context.synchronizeStreamChecked(streams[0]));
                    ASSERT_EQ(actual,expected);
                }
                capture->reset();
                EXPECT_FALSE(capture->hasExecutable());
                EXPECT_EQ(capture->nodeCount(),0u);
            }
        }
        EXPECT_EQ(memory->claimedBytes(device,PhysicalMemoryOwner::ExecutionWorkspace,
            PhysicalMemoryMaterializationKind::NewAllocation),0u);
    }
}
