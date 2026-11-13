/** @file PreparedDevicePeerCopy.cpp
 * @brief Delegate frozen byte-copy submission to the public transfer/coherence
 * authority. No allocation, context lookup, producer query or communicator work
 * is performed here.
 */
#include "PreparedDevicePeerCopy.h"
#include "TransferEngine.h"
namespace llaminar2 {
void PreparedDevicePeerCopy::enqueue(std::size_t source_offset,
                                     std::size_t destination_offset,
                                     std::size_t live_bytes) const {
  TransferEngine::enqueuePreparedDevicePeerCopy(*this, source_offset,
                                                destination_offset, live_bytes);
}
} // namespace llaminar2
