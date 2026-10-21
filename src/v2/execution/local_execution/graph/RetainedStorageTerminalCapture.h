/**
 * @file RetainedStorageTerminalCapture.h
 * @brief Lifetime-safe terminal recording that retains its exact borrowed stream.
 *
 * A generation parent may be retired before its immutable-storage close source.
 * The source must still have a live native stream while its graph destructor
 * executes. This typed owner retains the original shared stream lifetime and
 * always destroys the recording first; no caller can release those resources
 * in the opposite order or bind a graph to an unrelated stream.
 */
#pragma once
#include "../../../backends/IGPUGraphCapture.h"
#include <memory>
#include <stdexcept>
#include <utility>
namespace llaminar2
{
/** @brief One immutable terminal graph and the exact stream lifetime it borrows. */
class RetainedStorageTerminalCapture final
{
public:
    /** @brief Start empty, before terminal topology is materialized. */
    RetainedStorageTerminalCapture()=default;
    /** @brief Retire the recording before the last stream owner can disappear. */
    ~RetainedStorageTerminalCapture(){reset();}
    RetainedStorageTerminalCapture(const RetainedStorageTerminalCapture &)=delete;
    RetainedStorageTerminalCapture &operator=(const RetainedStorageTerminalCapture &)=delete;
    /** @brief Transfer both resources, retaining the borrowed stream throughout graph movement. */
    RetainedStorageTerminalCapture(RetainedStorageTerminalCapture &&other) noexcept
        :stream_(std::move(other.stream_)),capture_(std::move(other.capture_)){}
    /** @brief Retire the old graph before its stream, then take the source pair.
     * @param other Complete terminal binding; left empty after a non-self move.
     * @return This owner, including an unchanged valid binding after self-move.
     */
    RetainedStorageTerminalCapture &operator=(RetainedStorageTerminalCapture &&other) noexcept
    {
        if(this!=&other){reset();stream_=std::move(other.stream_);capture_=std::move(other.capture_);}
        return *this;
    }
    /** @brief Bind an empty owner to one exact recording and its original stream.
     * @param stream Strong native stream lifetime, supplied by the generation owner.
     * @param capture Recording whose executionStream is exactly stream.get().
     * @throws std::invalid_argument For missing, mismatched or replacement topology.
     */
    void bind(std::shared_ptr<void> stream,std::unique_ptr<IGPUGraphCapture> capture)
    {
        if(stream_||capture_||!stream||!capture||capture->executionStream()!=stream.get())
            throw std::invalid_argument("Retained storage terminal requires one exact live stream and cannot replace its graph");
        stream_=std::move(stream);capture_=std::move(capture);
    }
    /** @brief Retire graph resources while the exact native stream is still alive. */
    void reset() noexcept{capture_.reset();stream_.reset();}
    /** @brief Return whether one terminal recording is retained. */
    explicit operator bool() const noexcept{return bool(capture_);}
    /** @brief Borrow the recording for setup composition or authenticated terminal submission. */
    IGPUGraphCapture *get() const noexcept{return capture_.get();}
    /** @brief Access recording operations without exposing independent ownership. */
    IGPUGraphCapture *operator->() const noexcept{return capture_.get();}
private:
    std::shared_ptr<void> stream_; ///< Destroyed after capture_, including exceptional unwind.
    std::unique_ptr<IGPUGraphCapture> capture_;
};
} // namespace llaminar2
