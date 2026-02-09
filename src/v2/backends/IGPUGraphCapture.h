#pragma once

#include <cstddef>
#include <memory>
#include <string>

namespace llaminar2
{

    /// Result of an in-place graph executable update
    enum class GraphUpdateResult
    {
        Success,            ///< Graph executable updated in-place
        NeedsReinstantiate, ///< Topology changed, needs full re-instantiation
        Failed              ///< Update failed, fallback to non-graph path
    };

    /// Abstract interface for GPU graph capture and replay.
    /// Abstracts over HIP Graphs (ROCm) and CUDA Graphs (NVIDIA).
    ///
    /// Lifecycle:
    ///   1. beginCapture() — stream enters capture mode
    ///   2. (launch kernels into the captured stream)
    ///   3. endCapture() — stop capture, produce graph object
    ///   4. instantiate() — compile graph into executable
    ///   5. launch() — replay the entire captured workload
    ///
    /// For subsequent decode steps (same topology, different params):
    ///   1. beginCapture() → launch kernels → endCapture()
    ///   2. tryUpdate() — in-place update of the executable
    ///   3. launch()
    ///
    /// If tryUpdate() returns NeedsReinstantiate:
    ///   1. instantiate() to create a new executable from the latest capture
    ///   2. launch()
    class IGPUGraphCapture
    {
    public:
        virtual ~IGPUGraphCapture() = default;

        /// Begin stream capture. All subsequent kernel launches on the associated
        /// stream will be recorded into a graph rather than executed.
        /// @return true on success
        virtual bool beginCapture() = 0;

        /// End stream capture and produce a graph object from the recorded operations.
        /// @return true on success
        virtual bool endCapture() = 0;

        /// Compile the captured graph into an executable.
        /// Must be called after endCapture() before the first launch().
        /// @return true on success
        virtual bool instantiate() = 0;

        /// Launch (replay) the instantiated graph executable on the associated stream.
        /// @return true on success
        virtual bool launch() = 0;

        /// Attempt to update the existing executable in-place with a newly captured graph.
        /// Call this after endCapture() on subsequent iterations where the graph topology
        /// is unchanged but kernel parameters differ.
        /// @return GraphUpdateResult indicating success, need for reinstantiation, or failure
        virtual GraphUpdateResult tryUpdate() = 0;

        /// @return true if an instantiated executable exists and is ready for launch()
        virtual bool hasExecutable() const = 0;

        /// @return Number of nodes in the last captured graph (0 if no capture done)
        virtual size_t nodeCount() const = 0;

        /// Destroy all captured graph and executable resources.
        /// Safe to call multiple times or on an empty object.
        virtual void reset() = 0;

        /// @return Human-readable backend name ("HIP" or "CUDA")
        virtual const char *backendName() const = 0;

    protected:
        IGPUGraphCapture() = default;
        // Non-copyable
        IGPUGraphCapture(const IGPUGraphCapture &) = delete;
        IGPUGraphCapture &operator=(const IGPUGraphCapture &) = delete;
    };

} // namespace llaminar2
