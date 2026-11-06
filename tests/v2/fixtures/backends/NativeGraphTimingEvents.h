/**
 * @file NativeGraphTimingEvents.h
 * @brief Fixture-owned event timing around an unchanged complete retained graph.
 *
 * Events are created outside capture and timing, on the original admitted
 * worker. Input preparation and result oracles remain outside the measured
 * interval. This diagnostic helper is not a production ordering authority.
 */
#pragma once
#include "backends/BackendManager.h"
#include "backends/IBackend.h"
#include "backends/IWorkerGPUContext.h"
#include <cmath>
#include <stdexcept>
namespace llaminar2::test {
/** @brief One reusable start/stop pair with exact-stream ownership and terminal release. */
class NativeGraphTimingEvents final {
    IWorkerGPUContext& worker_; IBackend* backend_; DeviceId device_;
    void* start_=nullptr; void* stop_=nullptr; void* stream_=nullptr;
    /** @brief Fail at the original event boundary instead of inventing a duration. */
    static void require(bool value,const char* message){if(!value)throw std::runtime_error(message);}
public:
    /**
     * @brief Create only the two diagnostic events on the original fixture worker.
     * @param worker Actual already-admitted worker; caller retains it through destruction.
     * @param device Exact participant whose stream the events will bracket.
     */
    NativeGraphTimingEvents(IWorkerGPUContext& worker,DeviceId device)
        :worker_(worker),backend_(getBackendFor(device)),device_(device)
    {
        require(backend_ && device_.is_gpu(),"Native timing requires the original GPU backend");
        worker_.submitAndWait([&]{try{
            start_=backend_->createTimingEvent(device_.ordinal);require(start_,"Native start event unavailable");
            stop_=backend_->createTimingEvent(device_.ordinal);require(stop_,"Native stop event unavailable");
        }catch(...){if(start_)backend_->destroyEvent(start_,device_.ordinal);start_=nullptr;throw;}});
    }
    /** @brief Event ownership is unique; copying would create double retirement. */
    NativeGraphTimingEvents(const NativeGraphTimingEvents&)=delete;
    NativeGraphTimingEvents& operator=(const NativeGraphTimingEvents&)=delete;
    /** @brief Complete all diagnostic work before releasing either event outside capture. */
    ~NativeGraphTimingEvents(){worker_.submitAndWait([&]{
        if(stream_ && !backend_->synchronizeStream(stream_,device_.ordinal))std::terminate();
        if(stop_)backend_->destroyEvent(stop_,device_.ordinal);
        if(start_)backend_->destroyEvent(start_,device_.ordinal);
    });}
    /** @brief Record after untimed input publication. @param stream Exact graph execution stream. */
    void begin(void* stream){require(stream && (!stream_ || stream_==stream),"Timing events require one original exact stream");stream_=stream;
        require(backend_->recordEvent(start_,device_.ordinal,stream),"Native timing start failed");}
    /** @brief Record after the complete graph launch. @param stream Same original execution stream. */
    void end(void* stream){require(stream && stream==stream_,"Native timing end replaced its original source stream");
        require(backend_->recordEvent(stop_,device_.ordinal,stream),"Native timing stop failed");}
    /** @return Elapsed microseconds after the caller has joined the complete native interval. */
    double microseconds(){float ms=0;worker_.submitAndWait([&]{
        require(backend_->eventElapsedTimeMs(start_,stop_,device_.ordinal,&ms),"Native timing interval unavailable");});
        require(std::isfinite(ms)&&ms>=0,"Invalid native event duration");return static_cast<double>(ms)*1000;}
};
} // namespace llaminar2::test
