/**
 * @file Test__MappedHostTransferRegion.cpp
 * @brief Device-free contract tests for shared mapped activation channels.
 *
 * These tests exercise the public TransferEngine authority with independent
 * CUDA- and ROCm-shaped backend spies. They prove that one shared allocation
 * is registered once per backend family, resolves one alias per exact device,
 * carries 64-bit waits/publications on the caller's exact streams, never
 * synchronizes as a side effect, and rolls partial setup back before releasing
 * the external mapping lifetime.
 */

#include "backends/IBackend.h"
#include "transfer/TransferEngine.h"
#include "../../mocks/MockBackend.h"

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

using namespace llaminar2;
using namespace llaminar2::test;

namespace
{
    /** @brief Records one exact-stream timeline operation submitted by the engine. */
    struct TimelineCall
    {
        enum class Kind : std::uint8_t
        {
            Wait,
            Publish,
        };

        Kind kind = Kind::Wait;
        void *stream = nullptr;
        void *signal = nullptr;
        std::uint64_t value = 0u;
        int device_ordinal = -1;
    };

    /** @brief One progress-kernel launch observed below TransferEngine. */
    struct MappedProgressCall
    {
        void *destination_alias = nullptr;
        const void *source = nullptr;
        size_t bytes = 0u;
        int device_ordinal = -1;
        void *stream = nullptr;
    };

    /**
     * @brief Backend spy for setup registration and non-blocking timeline calls.
     *
     * Aliases deliberately differ by backend family and ordinal. This makes a
     * mistaken reuse of the host address or the registration device's alias
     * observable without needing a real accelerator in a unit test.
     */
    class MappedTimelineBackend final : public MockBackend
    {
    public:
        MappedTimelineBackend(
            DeviceType type,
            std::uintptr_t alias_bias,
            const bool *lifetime_released)
            : MockBackend(type),
              alias_bias_(alias_bias),
              lifetime_released_(lifetime_released)
        {
        }

        /** @brief Register the family once, unless failure injection is active. */
        bool registerExternalMappedHostMemory(
            void *ptr,
            size_t bytes,
            int registration_device_id) override
        {
            ++register_count;
            registered_host = ptr;
            registered_bytes = bytes;
            registration_ordinal = registration_device_id;
            return registration_succeeds;
        }

        /** @brief Return the deterministic alias belonging to one exact ordinal. */
        bool externalMappedHostDevicePointer(
            void *host_ptr,
            int device_id,
            void **device_ptr) override
        {
            ++alias_count;
            if (!device_ptr || host_ptr != registered_host ||
                !registration_succeeds)
            {
                return false;
            }
            *device_ptr = aliasFor(host_ptr, device_id);
            return true;
        }

        /** @brief Record that unregistration precedes mapping lifetime release. */
        bool unregisterExternalMappedHostMemory(
            void *ptr,
            int registration_device_id) override
        {
            ++unregister_count;
            unregister_host = ptr;
            unregister_ordinal = registration_device_id;
            unregister_saw_live_lifetime =
                lifetime_released_ && !*lifetime_released_;
            return true;
        }

        /** @brief The spy intentionally advertises exact-stream timeline support. */
        bool supportsStreamTimelineSignal64(int device_id) const override
        {
            return device_id >= 0;
        }

        /** @brief Record a wait without mutating or waiting on the host. */
        bool streamWaitTimelineSignal64(
            void *stream,
            void *signal,
            std::uint64_t value,
            int device_id) override
        {
            timeline_calls.push_back({
                .kind = TimelineCall::Kind::Wait,
                .stream = stream,
                .signal = signal,
                .value = value,
                .device_ordinal = device_id,
            });
            return timeline_calls_succeed;
        }

        /** @brief Record a fenced publication without synchronizing the host. */
        bool streamPublishTimelineSignal64(
            void *stream,
            void *signal,
            std::uint64_t value,
            int device_id) override
        {
            timeline_calls.push_back({
                .kind = TimelineCall::Kind::Publish,
                .stream = stream,
                .signal = signal,
                .value = value,
                .device_ordinal = device_id,
            });
            return timeline_calls_succeed;
        }

        /** @brief Record the exact mapped alias and stream without GPU work. */
        bool deviceToMappedHostByKernelOnStream(
            void *dst,
            const void *src,
            size_t bytes,
            int device_id,
            void *stream) override
        {
            progress_calls.push_back({
                .destination_alias = dst,
                .source = src,
                .bytes = bytes,
                .device_ordinal = device_id,
                .stream = stream,
            });
            return progress_calls_succeed;
        }

        /** @return Deterministic aligned alias for an exact backend/device pair. */
        [[nodiscard]] void *aliasFor(void *host_ptr, int device_id) const
        {
            return reinterpret_cast<void *>(
                reinterpret_cast<std::uintptr_t>(host_ptr) + alias_bias_ +
                (static_cast<std::uintptr_t>(device_id) + 1u) * 0x1000u);
        }

        bool registration_succeeds = true;
        bool timeline_calls_succeed = true;
        bool progress_calls_succeed = true;
        size_t register_count = 0u;
        size_t alias_count = 0u;
        size_t unregister_count = 0u;
        void *registered_host = nullptr;
        size_t registered_bytes = 0u;
        int registration_ordinal = -1;
        void *unregister_host = nullptr;
        int unregister_ordinal = -1;
        bool unregister_saw_live_lifetime = false;
        std::vector<TimelineCall> timeline_calls;
        std::vector<MappedProgressCall> progress_calls;

    private:
        std::uintptr_t alias_bias_ = 0u;
        const bool *lifetime_released_ = nullptr;
    };

    /** @brief Makes release ordering visible after the last retained owner drops. */
    struct LifetimeProbe
    {
        explicit LifetimeProbe(bool *released) : released_(released) {}

        ~LifetimeProbe()
        {
            if (released_)
                *released_ = true;
        }

    private:
        bool *released_ = nullptr;
    };

    MappedTimelineBackend *g_cuda_backend = nullptr;
    MappedTimelineBackend *g_rocm_backend = nullptr;

    /** @brief Resolve backend authority by endpoint type, never by protocol role. */
    IBackend *resolveMappedTimelineBackend(DeviceId device)
    {
        if (device.type == DeviceType::CUDA)
            return g_cuda_backend;
        if (device.type == DeviceType::ROCm)
            return g_rocm_backend;
        return nullptr;
    }

    /** @brief Installs independent family spies for one focused test. */
    class Test__MappedHostTransferRegion : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            lifetime_released_ = false;
            cuda_ = std::make_unique<MappedTimelineBackend>(
                DeviceType::CUDA, 0x100000u, &lifetime_released_);
            rocm_ = std::make_unique<MappedTimelineBackend>(
                DeviceType::ROCm, 0x200000u, &lifetime_released_);
            g_cuda_backend = cuda_.get();
            g_rocm_backend = rocm_.get();
        }

        void TearDown() override
        {
            g_cuda_backend = nullptr;
            g_rocm_backend = nullptr;
        }

        alignas(64) std::array<std::byte, 1024> pages_{};
        bool lifetime_released_ = false;
        std::unique_ptr<MappedTimelineBackend> cuda_;
        std::unique_ptr<MappedTimelineBackend> rocm_;
        TransferEngine engine_{&resolveMappedTimelineBackend};
    };
} // namespace

TEST_F(
    Test__MappedHostTransferRegion,
    RegistersOncePerFamilyAndReleasesMappingOnlyAfterBothUnregister)
{
    auto lifetime = std::make_shared<LifetimeProbe>(&lifetime_released_);
    const std::array devices{
        DeviceId::rocm(1),
        DeviceId::cuda(1),
        DeviceId::cpu(),
        DeviceId::rocm(0),
        DeviceId::cuda(0),
    };

    auto region = engine_.registerExternalMappedHostRegion(
        pages_.data(), pages_.size(), devices, lifetime);

    ASSERT_TRUE(region->isBound());
    EXPECT_EQ(region->sizeBytes(), pages_.size());
    EXPECT_EQ(cuda_->register_count, 1u);
    EXPECT_EQ(rocm_->register_count, 1u);
    EXPECT_EQ(cuda_->alias_count, 2u);
    EXPECT_EQ(rocm_->alias_count, 2u);
    EXPECT_EQ(region->deviceAlias(DeviceId::cpu()), pages_.data());
    EXPECT_EQ(
        region->deviceAlias(DeviceId::cuda(1)),
        cuda_->aliasFor(pages_.data(), 1));
    EXPECT_EQ(
        region->deviceAlias(DeviceId::rocm(0)),
        rocm_->aliasFor(pages_.data(), 0));
    EXPECT_EQ(cuda_->getSyncCount(), 0u);
    EXPECT_EQ(cuda_->getStreamSyncCount(), 0u);
    EXPECT_EQ(rocm_->getSyncCount(), 0u);
    EXPECT_EQ(rocm_->getStreamSyncCount(), 0u);

    // Leave the region as the sole lifetime owner before teardown.
    lifetime.reset();
    EXPECT_FALSE(lifetime_released_);
    region.reset();

    EXPECT_EQ(cuda_->unregister_count, 1u);
    EXPECT_EQ(rocm_->unregister_count, 1u);
    EXPECT_TRUE(cuda_->unregister_saw_live_lifetime);
    EXPECT_TRUE(rocm_->unregister_saw_live_lifetime);
    EXPECT_TRUE(lifetime_released_);
}

TEST_F(
    Test__MappedHostTransferRegion,
    EitherBackendRoleUsesItsExactAliasStreamAndMonotonicValueWithoutSync)
{
    auto lifetime = std::make_shared<LifetimeProbe>(&lifetime_released_);
    const std::array devices{DeviceId::cuda(0), DeviceId::rocm(0)};
    auto region = engine_.registerExternalMappedHostRegion(
        pages_.data(), pages_.size(), devices, lifetime);

    constexpr size_t signal_offset = 128u;
    void *const cuda_stream = reinterpret_cast<void *>(0xCAFE0001u);
    void *const rocm_stream = reinterpret_cast<void *>(0xBEEF0002u);

    // CUDA may be the continuation and ROCm the follower...
    engine_.enqueueMappedTimelinePublish64(
        *region, signal_offset, 0x10001u, DeviceId::cuda(0), cuda_stream);
    engine_.enqueueMappedTimelineWait64(
        *region, signal_offset, 0x10001u, DeviceId::rocm(0), rocm_stream);
    // ...or the planner may reverse those semantic roles later.
    engine_.enqueueMappedTimelinePublish64(
        *region, signal_offset, 0x20002u, DeviceId::rocm(0), rocm_stream);
    engine_.enqueueMappedTimelineWait64(
        *region, signal_offset, 0x20002u, DeviceId::cuda(0), cuda_stream);

    ASSERT_EQ(cuda_->timeline_calls.size(), 2u);
    EXPECT_EQ(cuda_->timeline_calls[0].kind, TimelineCall::Kind::Publish);
    EXPECT_EQ(cuda_->timeline_calls[0].stream, cuda_stream);
    EXPECT_EQ(
        cuda_->timeline_calls[0].signal,
        static_cast<void *>(
            static_cast<std::byte *>(cuda_->aliasFor(pages_.data(), 0)) +
            signal_offset));
    EXPECT_EQ(cuda_->timeline_calls[0].value, 0x10001u);
    EXPECT_EQ(cuda_->timeline_calls[1].kind, TimelineCall::Kind::Wait);
    EXPECT_EQ(cuda_->timeline_calls[1].stream, cuda_stream);
    EXPECT_EQ(cuda_->timeline_calls[1].value, 0x20002u);

    ASSERT_EQ(rocm_->timeline_calls.size(), 2u);
    EXPECT_EQ(rocm_->timeline_calls[0].kind, TimelineCall::Kind::Wait);
    EXPECT_EQ(rocm_->timeline_calls[0].stream, rocm_stream);
    EXPECT_EQ(
        rocm_->timeline_calls[0].signal,
        static_cast<void *>(
            static_cast<std::byte *>(rocm_->aliasFor(pages_.data(), 0)) +
            signal_offset));
    EXPECT_EQ(rocm_->timeline_calls[0].value, 0x10001u);
    EXPECT_EQ(rocm_->timeline_calls[1].kind, TimelineCall::Kind::Publish);
    EXPECT_EQ(rocm_->timeline_calls[1].stream, rocm_stream);
    EXPECT_EQ(rocm_->timeline_calls[1].value, 0x20002u);

    EXPECT_EQ(cuda_->getSyncCount(), 0u);
    EXPECT_EQ(cuda_->getStreamSyncCount(), 0u);
    EXPECT_EQ(rocm_->getSyncCount(), 0u);
    EXPECT_EQ(rocm_->getStreamSyncCount(), 0u);
}

TEST_F(
    Test__MappedHostTransferRegion,
    KernelTimelineBindingsPreserveExactEndpointTypeAndDoNotEnqueueWork)
{
    auto lifetime = std::make_shared<LifetimeProbe>(&lifetime_released_);
    const std::array devices{DeviceId::cuda(0), DeviceId::rocm(0)};
    auto region = engine_.registerExternalMappedHostRegion(
        pages_.data(), pages_.size(), devices, lifetime);

    constexpr size_t wait_offset = 128u;
    constexpr size_t publication_offset = 192u;
    const auto wait = engine_.bindMappedTimelineKernelWait64(
        *region, wait_offset, 0x10001u, DeviceId::rocm(0));
    const auto publication = engine_.bindMappedTimelineKernelPublish64(
        *region, publication_offset, 0x20002u, DeviceId::cuda(0));

    ASSERT_TRUE(wait.valid());
    EXPECT_EQ(wait.device(), DeviceId::rocm(0));
    EXPECT_EQ(wait.value(), 0x10001u);
    EXPECT_EQ(
        wait.deviceSignal(),
        static_cast<const std::uint64_t *>(static_cast<void *>(
            static_cast<std::byte *>(rocm_->aliasFor(pages_.data(), 0)) +
            wait_offset)));
    ASSERT_TRUE(publication.valid());
    EXPECT_EQ(publication.device(), DeviceId::cuda(0));
    EXPECT_EQ(publication.value(), 0x20002u);
    EXPECT_EQ(
        publication.deviceSignal(),
        static_cast<std::uint64_t *>(static_cast<void *>(
            static_cast<std::byte *>(cuda_->aliasFor(pages_.data(), 0)) +
            publication_offset)));

    /* Binding is capture setup, not execution: packet kernels receive the
     * typed result later on their exact stream. No backend wait/publication or
     * synchronization is allowed at this boundary. */
    EXPECT_TRUE(cuda_->timeline_calls.empty());
    EXPECT_TRUE(rocm_->timeline_calls.empty());
    EXPECT_EQ(cuda_->getSyncCount(), 0u);
    EXPECT_EQ(cuda_->getStreamSyncCount(), 0u);
    EXPECT_EQ(rocm_->getSyncCount(), 0u);
    EXPECT_EQ(rocm_->getStreamSyncCount(), 0u);

    EXPECT_THROW(
        {
            [[maybe_unused]] const auto invalid_wait =
                engine_.bindMappedTimelineKernelWait64(
                    *region,
                    wait_offset + 1u,
                    1u,
                    DeviceId::cuda(0));
        },
        std::invalid_argument);
    EXPECT_THROW(
        {
            [[maybe_unused]] const auto invalid_publication =
                engine_.bindMappedTimelineKernelPublish64(
                    *region,
                    publication_offset,
                    1u,
                    DeviceId::cuda(1));
        },
        std::invalid_argument);
}

TEST_F(
    Test__MappedHostTransferRegion,
    ProgressKernelUsesExactMappedAliasBoundsDeviceAndStreamWithoutSync)
{
    auto lifetime = std::make_shared<LifetimeProbe>(&lifetime_released_);
    const std::array devices{DeviceId::cuda(0), DeviceId::rocm(0)};
    auto region = engine_.registerExternalMappedHostRegion(
        pages_.data(), pages_.size(), devices, lifetime);
    alignas(64) std::array<std::byte, 512> source{};
    void *const stream = reinterpret_cast<void *>(0xCAFE1234u);

    engine_.enqueuePersistentDeviceRegionToMappedHostByKernel(
        source.data(),
        source.size(),
        64u,
        *region,
        128u,
        256u,
        DeviceId::cuda(0),
        stream);

    ASSERT_EQ(cuda_->progress_calls.size(), 1u);
    const auto &call = cuda_->progress_calls.front();
    EXPECT_EQ(
        call.destination_alias,
        static_cast<void *>(
            static_cast<std::byte *>(cuda_->aliasFor(pages_.data(), 0)) +
            128u));
    EXPECT_EQ(call.source, source.data() + 64u);
    EXPECT_EQ(call.bytes, 256u);
    EXPECT_EQ(call.device_ordinal, 0);
    EXPECT_EQ(call.stream, stream);
    EXPECT_TRUE(rocm_->progress_calls.empty());
    EXPECT_EQ(cuda_->getSyncCount(), 0u);
    EXPECT_EQ(cuda_->getStreamSyncCount(), 0u);

    EXPECT_THROW(
        engine_.enqueuePersistentDeviceRegionToMappedHostByKernel(
            source.data(),
            source.size(),
            400u,
            *region,
            0u,
            256u,
            DeviceId::cuda(0),
            stream),
        std::out_of_range);
    EXPECT_THROW(
        engine_.enqueuePersistentDeviceRegionToMappedHostByKernel(
            source.data(),
            source.size(),
            0u,
            *region,
            900u,
            256u,
            DeviceId::cuda(0),
            stream),
        std::out_of_range);
    EXPECT_THROW(
        engine_.enqueuePersistentDeviceRegionToMappedHostByKernel(
            source.data(),
            source.size(),
            0u,
            *region,
            0u,
            256u,
            DeviceId::cuda(0),
            nullptr),
        std::invalid_argument);
    EXPECT_EQ(cuda_->progress_calls.size(), 1u);
}

TEST_F(
    Test__MappedHostTransferRegion,
    RejectsDefaultStreamsUnalignedSignalsUndeclaredDevicesAndDuplicateEndpoints)
{
    auto lifetime = std::make_shared<LifetimeProbe>(&lifetime_released_);
    const std::array devices{DeviceId::cuda(0), DeviceId::rocm(0)};
    auto region = engine_.registerExternalMappedHostRegion(
        pages_.data(), pages_.size(), devices, lifetime);

    EXPECT_THROW(
        engine_.enqueueMappedTimelineWait64(
            *region, 64u, 1u, DeviceId::cuda(0), nullptr),
        std::invalid_argument);
    EXPECT_THROW(
        engine_.enqueueMappedTimelinePublish64(
            *region,
            65u,
            1u,
            DeviceId::rocm(0),
            reinterpret_cast<void *>(0x1234u)),
        std::invalid_argument);
    EXPECT_THROW(
        engine_.enqueueMappedTimelineWait64(
            *region,
            64u,
            1u,
            DeviceId::cuda(1),
            reinterpret_cast<void *>(0x1234u)),
        std::invalid_argument);

    const std::array duplicates{DeviceId::cuda(0), DeviceId::cuda(0)};
    EXPECT_THROW(
        {
            [[maybe_unused]] auto duplicate_region =
                engine_.registerExternalMappedHostRegion(
                    pages_.data(), pages_.size(), duplicates, lifetime);
        },
        std::invalid_argument);
    EXPECT_TRUE(cuda_->timeline_calls.empty());
    EXPECT_TRUE(rocm_->timeline_calls.empty());
}

TEST_F(
    Test__MappedHostTransferRegion,
    PartialCrossBackendRegistrationRollsBackBeforeLifetimeRelease)
{
    rocm_->registration_succeeds = false;
    auto lifetime = std::make_shared<LifetimeProbe>(&lifetime_released_);
    const std::array devices{DeviceId::cuda(0), DeviceId::rocm(0)};

    EXPECT_THROW(
        {
            [[maybe_unused]] auto failed_region =
                engine_.registerExternalMappedHostRegion(
                    pages_.data(), pages_.size(), devices, lifetime);
        },
        std::runtime_error);

    EXPECT_EQ(cuda_->register_count, 1u);
    EXPECT_EQ(cuda_->unregister_count, 1u);
    EXPECT_TRUE(cuda_->unregister_saw_live_lifetime);
    EXPECT_EQ(rocm_->register_count, 1u);
    EXPECT_EQ(rocm_->unregister_count, 0u);
    EXPECT_FALSE(lifetime_released_);
    lifetime.reset();
    EXPECT_TRUE(lifetime_released_);
}
