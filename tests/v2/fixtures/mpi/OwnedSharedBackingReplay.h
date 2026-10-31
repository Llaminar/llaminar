/**
 * @file OwnedSharedBackingReplay.h
 * @brief Device-free IMPIContext packet replay for the real startup decoder.
 *
 * Only startup message delivery is modeled. The production startup state
 * machine, directory authentication and real POSIX file operations are used.
 * These fixtures never construct native certification or replace an owner.
 */
#pragma once
#include "mocks/MockMPIContext.h"
#include "execution/moe/MoEOverlaySharedBackingStartup.h"
#include <deque>
#include <functional>
#include <map>
#include <mutex>
#include <cstring>
#include <memory>
#include <vector>
namespace llaminar2::test {
    /** @brief One exact copied MPI message, retained until its receive acquires it. */
    struct OwnedSharedBackingReplayMessage {
        int source,target,tag;
        std::vector<std::byte> bytes;
    };
    /** @brief Explicit two/many-rank replay link; no production file/byte authority lives here. */
    struct OwnedSharedBackingReplayLink final {
        std::mutex mutex;
        std::deque<OwnedSharedBackingReplayMessage> messages;
        std::function<void(OwnedSharedBackingReplayMessage&)> before_send;
    };
    /** @brief Existing mock context plus truthful ownership of pending startup packets. */
    class OwnedSharedBackingReplayContext final : public MockMPIContext {
        struct Request final { bool send;void *destination;std::size_t capacity;int source,tag; };
    public:
        OwnedSharedBackingReplayContext(int rank,int size):MockMPIContext(rank,size){}
        /** @brief Bind this replay to its explicit shared startup link before construction. */
        void bind(std::shared_ptr<OwnedSharedBackingReplayLink> link) {link_=std::move(link);}
        MPI_Request isend(const void *data,size_t count,MPI_Datatype type,int target,int tag) const override
        {
            if(tag!=kMoEOverlaySharedBackingStartupTag)return MockMPIContext::isend(data,count,type,target,tag);
            if(!link_ || type!=MPI_BYTE)throw std::logic_error("Startup replay lacks its actual typed byte link");
            (void)MockMPIContext::isend(data,count,type,target,tag);
            OwnedSharedBackingReplayMessage message{rank(),target,tag,std::vector<std::byte>(count)};
            std::memcpy(message.bytes.data(),data,count);
            if(link_->before_send)link_->before_send(message);
            {std::lock_guard lock(link_->mutex);link_->messages.push_back(std::move(message));}
            return reinterpret_cast<MPI_Request>(new Request{true,nullptr,0,rank(),tag});
        }
        MPI_Request irecv(void *data,size_t count,MPI_Datatype type,int source,int tag) const override
        {
            if(tag!=kMoEOverlaySharedBackingStartupTag)return MockMPIContext::irecv(data,count,type,source,tag);
            if(!link_ || type!=MPI_BYTE)throw std::logic_error("Startup replay lacks its actual typed byte link");
            (void)MockMPIContext::irecv(data,count,type,source,tag);
            return reinterpret_cast<MPI_Request>(new Request{false,data,count,source,tag});
        }
        bool test(MPI_Request *handle,MPI_Status *status) const override
        {
            if(!handle || *handle==MPI_REQUEST_NULL)return MockMPIContext::test(handle,status);
            MPI_Request counter=MPI_REQUEST_NULL;(void)MockMPIContext::test(&counter,nullptr);
            auto *request=reinterpret_cast<Request*>(*handle);
            if(!request->send) {
                std::lock_guard lock(link_->mutex);
                auto found=std::find_if(link_->messages.begin(),link_->messages.end(),[&](const auto &message){
                    return message.target==rank() && message.source==request->source && message.tag==request->tag;});
                if(found==link_->messages.end())return false;
                if(found->bytes.size()>request->capacity)throw std::runtime_error("Startup replay native truncation");
                std::memcpy(request->destination,found->bytes.data(),found->bytes.size());
                received_bytes_=static_cast<int>(found->bytes.size());link_->messages.erase(found);
            }
            if(status) {status->MPI_SOURCE=request->source;status->MPI_TAG=request->tag;status->MPI_ERROR=MPI_SUCCESS;}
            delete request;*handle=MPI_REQUEST_NULL;return true;
        }
        int getCount(const MPI_Status &,MPI_Datatype type) const override
        {if(type!=MPI_BYTE)throw std::logic_error("Unexpected replay count datatype");return received_bytes_;}
    private:
        std::shared_ptr<OwnedSharedBackingReplayLink> link_;
        mutable int received_bytes_=0;
    };
    /** @brief Join independently built mock endpoints under their explicit existing channel identity. */
    inline std::shared_ptr<OwnedSharedBackingReplayLink> ownedSharedBackingReplayLink(const std::string &channel)
    {
        static std::mutex mutex;
        static std::map<std::string,std::weak_ptr<OwnedSharedBackingReplayLink>> links;
        std::lock_guard lock(mutex);auto result=links[channel].lock();
        if(!result) {result=std::make_shared<OwnedSharedBackingReplayLink>();links[channel]=result;}
        return result;
    }
}
