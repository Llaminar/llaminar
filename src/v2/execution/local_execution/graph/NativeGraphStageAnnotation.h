/**
 * @file NativeGraphStageAnnotation.h
 * @brief Setup-only stage attribution for opt-in native graph diagnostics.
 *
 * A standalone observer can correlate the exact native nodes recorded by one
 * canonical compute stage with that stage's model-owned name. The engine never
 * allocates timing storage, inserts GPU events, reads GPU state or waits here.
 * An absent observer leaves ordinary capture unchanged. An installed observer
 * owns its diagnostic metadata and event lifetimes through native executable
 * retirement; it cannot substitute model work or become an inference owner.
 *
 * The versioned weak ABI keeps the observer out of the serving dependency
 * closure. Calls occur only during setup recording, never graph replay. The
 * explicit device/stream and scoped begin/complete/abort lifecycle prohibit
 * guessed streams, dangling stage attribution and exceptional partial records.
 */
#pragma once

#include "../../../backends/DeviceId.h"
#include <cstdint>
#include <stdexcept>

extern "C"
{
    /**
     * @brief Optional observer ABI, implemented by a standalone diagnostic DSO.
     * @param phase 0 begins a stage, 1 seals it, 2 aborts its partial annotation.
     * @param backend 1 denotes CUDA and 2 denotes ROCm.
     * @param ordinal Exact native device owning the recording stream.
     * @param stream Exact non-null stream already bound by capture admission.
     * @param name Borrowed stable compute-node name, valid through the call.
     * @param type Borrowed canonical compute-stage type name.
     */
    void llaminar_native_graph_stage_annotation_v1(
        std::uint32_t phase, std::uint32_t backend, int ordinal,
        void *stream, const char *name, const char *type) noexcept
        __attribute__((weak));
}

namespace llaminar2
{
    /// Exact versioned callback ABI; no mutable inference value crosses it.
    using NativeGraphStageAnnotationObserver = decltype(
        &llaminar_native_graph_stage_annotation_v1);

    /** @return Installed standalone observer, or null when annotation is disabled. */
    inline NativeGraphStageAnnotationObserver nativeGraphStageAnnotationObserver() noexcept
    {
        return llaminar_native_graph_stage_annotation_v1;
    }

    /** @brief Own one complete setup-time stage annotation without GPU operations. */
    class NativeGraphStageAnnotationScope final
    {
    public:
        /**
         * @brief Begin attribution on the admitted native recording stream.
         * @param observer Non-null installed diagnostic observer.
         * @param device Exact CUDA/ROCm device owning the stage.
         * @param stream Exact non-null capture stream, never a default substitute.
         * @param name Stable model compute-node name.
         * @param type Canonical compute-stage type name.
         * @throws std::invalid_argument If attribution ownership is incomplete.
         */
        NativeGraphStageAnnotationScope(
            NativeGraphStageAnnotationObserver observer, DeviceId device,
            void *stream, const char *name, const char *type)
            : observer_(observer), device_(device), stream_(stream),
              name_(name), type_(type)
        {
            if (!observer_ || (!device_.is_cuda() && !device_.is_rocm()) ||
                device_.ordinal < 0 || !stream_ || !name_ || !*name_ || !type_ || !*type_)
                throw std::invalid_argument("Native graph stage annotation requires an observer, exact GPU/stream and stage identity");
            publish(Phase::Begin);
        }

        /** @brief Retire an exceptional/failed stage without certifying its partial nodes. */
        ~NativeGraphStageAnnotationScope()
        {
            if (state_ == State::Open) publish(Phase::Abort);
        }

        NativeGraphStageAnnotationScope(const NativeGraphStageAnnotationScope &) = delete;
        NativeGraphStageAnnotationScope &operator=(const NativeGraphStageAnnotationScope &) = delete;

        /**
         * @brief Seal attribution only after the canonical stage succeeds.
         * @throws std::logic_error If a caller attempts to seal the scope twice.
         */
        void complete()
        {
            if (state_ != State::Open)
                throw std::logic_error("Native graph stage annotation cannot complete twice");
            publish(Phase::Complete);
            state_ = State::Complete;
        }

    private:
        enum class Phase : std::uint32_t { Begin, Complete, Abort };
        enum class State : std::uint8_t { Open, Complete };

        /** @brief Publish immutable stage identity at one setup-only lifecycle boundary. */
        void publish(Phase phase) const noexcept
        {
            observer_(static_cast<std::uint32_t>(phase), device_.is_cuda() ? 1u : 2u,
                device_.ordinal, stream_, name_, type_);
        }

        NativeGraphStageAnnotationObserver observer_;
        DeviceId device_;
        void *stream_;
        const char *name_;
        const char *type_;
        State state_ = State::Open;
    };
}
