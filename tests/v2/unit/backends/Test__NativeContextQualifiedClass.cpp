/**
 * @file Test__NativeContextQualifiedClass.cpp
 * @brief Replay the installed native-worker class selector without creating a GPU context.
 *
 * Class data describes reviewed software, architecture and creator geometry.
 * Physical UUIDs and visibility identify the live owner, so a certificate must
 * retain them without restricting a portable class to the calibration host.
 * These tests exercise only public selection; native source-lifecycle tests
 * separately prove creation, same-PMA admission and physical retirement.
 */
#include "backends/NativeContextQualifiedLimits.h"
#include "fixtures/backends/RecordedNativeContextBirthKey.h"
#include <gtest/gtest.h>
#include <array>

namespace llaminar2::test {
namespace {
/** @return A complete recorded key in one installed class, with a distinct live owner. */
NativeContextBirthKey recorded(const detail::NativeContextQualifiedClass& row){return recordedNativeContextBirthKey(row.backend,row.population);}

/** @brief Change one digest while keeping it syntactically complete. */
void changeDigest(std::string& value) {
    ASSERT_EQ(value.size(), 64u);
    value.front() = value.front() == 'a' ? 'b' : 'a';
}
}

/** @test The selected certificate preserves the reviewed bounds and original live identity. */
TEST(NativeContextQualifiedClass, InstalledRowsPreserveReviewedLimitsAndLiveOwner) {
    ASSERT_FALSE(detail::kNativeContextQualifiedClasses.empty());
    for (const auto& row : detail::kNativeContextQualifiedClasses) {
        const auto key = recorded(row);
        const auto contract = NativeContextOpaqueCatalog::require(key);
        EXPECT_EQ(contract.key(), key);
        EXPECT_EQ(contract.hostUpperBytes(), row.host_upper_bytes);
        EXPECT_EQ(contract.deviceUpperBytes(), row.device_upper_bytes);
        EXPECT_EQ(contract.lifetime(), row.physical_lifetime);
        EXPECT_EQ(contract.evidenceSha256(), row.evidence_sha256);
    }
}

/** @test Ordinal remapping and another UUID do not turn a portable class into a host allowlist. */
TEST(NativeContextQualifiedClass, PhysicalRemappingRetainsActualIdentity) {
    ASSERT_FALSE(detail::kNativeContextQualifiedClasses.empty());
    for (const auto& row : detail::kNativeContextQualifiedClasses) {
        auto key = recorded(row);
        key.device = row.backend == DeviceType::CUDA ? DeviceId::cuda(1) : DeviceId::rocm(1);
        key.physical_uuid = "recorded-physical-device-two";
        key.visibility_identity = "recorded-visibility-two";
        const auto contract = NativeContextOpaqueCatalog::require(key);
        EXPECT_EQ(contract.key(), key);
        EXPECT_NE(contract.key().physical_uuid, recorded(row).physical_uuid);
    }
}

/** @test Opaque units are per original selected worker, independent of unrelated visible devices. */
TEST(NativeContextQualifiedClass, ValidVisiblePopulationUsesSamePerWorkerAllowance) {
    ASSERT_FALSE(detail::kNativeContextQualifiedClasses.empty());
    for (const auto& row : detail::kNativeContextQualifiedClasses) {
        auto key = recorded(row);
        key.visible_device_count = 1024;
        EXPECT_EQ(NativeContextOpaqueCatalog::require(key).hostUpperBytes(),row.host_upper_bytes);
    }
}

/** @test Native provider builds remain metadata; supported loader/backend ABI owns compatibility. */
TEST(NativeContextQualifiedClass, CompatibleRuntimeMetadataUsesSameAllowance) {
    ASSERT_FALSE(detail::kNativeContextQualifiedClasses.empty());
    const std::array fields{
        &NativeContextBirthKey::backend_sha256,
        &NativeContextBirthKey::runtime_sha256, &NativeContextBirthKey::driver_sha256,
        &NativeContextBirthKey::runtime_policy_sha256};
    for (const auto& row : detail::kNativeContextQualifiedClasses) {
        for (const auto field : fields) {
            auto key = recorded(row);
            changeDigest(key.*field);
            EXPECT_EQ(NativeContextOpaqueCatalog::require(key).hostUpperBytes(),row.host_upper_bytes);
        }
        for (const auto field : {&NativeContextBirthKey::runtime_version,
                                 &NativeContextBirthKey::driver_version,
                                 &NativeContextBirthKey::pthread_defaults_identity}) {
            auto key = recorded(row);
            (key.*field) += ".changed";
            EXPECT_EQ(NativeContextOpaqueCatalog::require(key).hostUpperBytes(),row.host_upper_bytes);
        }
        if (row.backend == DeviceType::ROCm) {
            auto key = recorded(row);
            changeDigest(key.hsa_sha256);
            EXPECT_EQ(NativeContextOpaqueCatalog::require(key).hostUpperBytes(),row.host_upper_bytes);
        }
    }
}

/** @test Current exact explicit geometry stays in the returned original key without changing opaque units. */
TEST(NativeContextQualifiedClass, CurrentExplicitGeometryUsesSameOpaqueAllowance) {
    ASSERT_FALSE(detail::kNativeContextQualifiedClasses.empty());
    const std::array fields{
        &NativeContextBirthKey::compute_units, &NativeContextBirthKey::native_wave_width,
        &NativeContextBirthKey::worker_stack_bytes, &NativeContextBirthKey::worker_guard_bytes,
        &NativeContextBirthKey::context_object_bytes};
    for (const auto& row : detail::kNativeContextQualifiedClasses) {
        for (const auto field : fields) {
            auto key = recorded(row);
            ++(key.*field);
            EXPECT_EQ(NativeContextOpaqueCatalog::require(key).hostUpperBytes(),row.host_upper_bytes);
        }
        auto key = recorded(row);
        key.context_object_alignment *= 2;
        EXPECT_EQ(NativeContextOpaqueCatalog::require(key).hostUpperBytes(),row.host_upper_bytes);
        key = recorded(row);
        key.architecture += ".changed";
        EXPECT_EQ(NativeContextOpaqueCatalog::require(key).hostUpperBytes(),row.host_upper_bytes);
    }
}

/** @test An unknown population and a changed creator family cannot reuse a reviewed class. */
TEST(NativeContextQualifiedClass, PopulationAndCreatorCountsAreClosed) {
    ASSERT_FALSE(detail::kNativeContextQualifiedClasses.empty());
    for (const auto& row : detail::kNativeContextQualifiedClasses) {
        auto key = recorded(row);
        key.population = static_cast<NativeContextBirthPopulation>(-1);
        EXPECT_THROW(NativeContextOpaqueCatalog::require(key), std::invalid_argument);
        for (const auto field : {&NativeContextBirthKey::primary_streams,
                                 &NativeContextBirthKey::primary_blas_handles,
                                 &NativeContextBirthKey::secondary_blas_handles,
                                 &NativeContextBirthKey::dependency_events}) {
            key = recorded(row);
            ++(key.*field);
            EXPECT_THROW(NativeContextOpaqueCatalog::require(key), std::invalid_argument);
        }
        key = recorded(row);
        ++key.source_program_version;
        EXPECT_THROW(NativeContextOpaqueCatalog::require(key), std::invalid_argument);
    }
}
}
