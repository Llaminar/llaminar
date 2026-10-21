/**
 * @file Test__Q8ResidualAllFormats.cpp
 * @brief Tiny device-free real prepared-stage coverage of every native TwoTerm
 * source format.
 *
 * Original Q8_0 fixtures and their accuracy threshold are untouched. This small
 * three-row/five-column sweep exercises real native factor, minimum and IQ
 * paths. Larger geometries and thread/ISA sweeps are explicit integration
 * cases.
 */
#include "../../utils/Q8ResidualAllFormatCPUHarness.h"
#include "kernels/common/GPUQ8ResidualNative.h"
#include <climits>
using namespace llaminar2;
/** @test The canonical inventory, not a second hand-maintained list, feeds
 * every actual CPU stage. */
TEST(Q8ResidualAllFormats, TinyActualPreparedStage) {
  for (const auto &format : test::quantizedVerifierFormats()) {
    test::q8_all_formats::proveCPU(format, 3, 5, 256, 1);
    ASSERT_FALSE(::testing::Test::HasFailure());
  }
}
/** @test Pure metadata rejects corrupt original identities and physical
 * allocation unions before any GPU operation. */
TEST(Q8ResidualAllFormats, NativeDescriptorRejectsForeignAndMalformedPlanes) {
  alignas(256) std::array<uint8_t, 4096> storage{};
  for (const auto &format : test::quantizedVerifierFormats()) {
    SCOPED_TRACE(format.label);
    const auto *source = native_vnni_formats::forSourceIdentity(
        format.source_codebook_id, format.source_is_superblock);
    ASSERT_NE(source, nullptr);
    const auto physical = canonicalDeviceVnniFormat(*source);
    DeviceNativeVNNIMatrixDesc d{};
    d.payload = storage.data();
    d.scales = storage.data() + 256;
    d.mins = physical.is_asymmetric ? storage.data() + 512 : nullptr;
    d.emins = physical.has_emins ? storage.data() + 768 : nullptr;
    d.n = 5;
    d.k = 256;
    d.blocks_per_row = 8;
    d.codebook_id = physical.codebook_id;
    d.source_codebook_id = format.source_codebook_id;
    d.source_is_superblock = format.source_is_superblock;
    d.source_identity_present = 1;
    EXPECT_NO_THROW((void)requireQ8ResidualNativeMatrix(d, 3, 5, 256));
    for (int mutation = 0; mutation < 15; ++mutation) {
      SCOPED_TRACE(mutation);
      auto bad = d;
      switch (mutation) {
      case 0:
        bad.source_identity_present = 0;
        break;
      case 1:
        bad.source_codebook_id = 255;
        break;
      case 2:
        bad.codebook_id = 255;
        break;
      case 3:
        bad.mins = physical.is_asymmetric ? nullptr : storage.data() + 512;
        break;
      case 4:
        bad.emins = physical.has_emins ? nullptr : storage.data() + 768;
        break;
      case 5:
        bad.blocks_per_row = 7;
        break;
      case 6:
        bad.n = 6;
        break;
      case 7:
        bad.k = 288;
        break;
      case 8:
        bad.reserved = 1;
        break;
      case 9:
        bad.reserved_tail = 1;
        break;
      case 10:
        bad.source_identity_present = 2;
        break;
      case 11:
        bad.source_is_superblock = 2;
        break;
      case 12:
        bad.allocation_has_mins = 1;
        break;
      case 13:
        bad.allocation_payload_bytes_per_block = 1;
        break;
      case 14:
        bad.codebook_id = d.codebook_id == 0 ? 17 : 0;
        break;
      }
      EXPECT_THROW((void)requireQ8ResidualNativeMatrix(bad, 3, 5, 256),
                   std::invalid_argument);
    }
    const auto reusable = reusableDeviceVnniAllocationFormat(*source);
    d.allocation_payload_bytes_per_block = reusable.payload_bytes_per_block;
    d.allocation_has_mins = reusable.has_mins;
    d.allocation_has_emins = reusable.has_emins;
    d.mins = reusable.has_mins ? storage.data() + 512 : nullptr;
    d.emins = reusable.has_emins ? storage.data() + 768 : nullptr;
    EXPECT_NO_THROW((void)requireQ8ResidualNativeMatrix(d, 3, 5, 256));
    auto bad = d;
    bad.allocation_has_emins ^= 1;
    EXPECT_THROW((void)requireQ8ResidualNativeMatrix(bad, 3, 5, 256),
                 std::invalid_argument);
    bad = d;
    bad.allocation_payload_bytes_per_block =
        physical.payload_bytes_per_block - 1;
    EXPECT_THROW((void)requireQ8ResidualNativeMatrix(bad, 3, 5, 256),
                 std::invalid_argument);
  }
}

/** @test Large declared capacity is legal; unrepresentable actual spans fail
 * without allocating. */
TEST(Q8ResidualAllFormats, NativeDeclaredCapacitySpanCannotOverflow) {
  alignas(256) std::array<uint8_t, 4096> storage{};
  DeviceNativeVNNIMatrixDesc d{};
  d.payload = storage.data();
  d.scales = storage.data() + 256;
  d.n = 5;
  d.k = 256;
  d.blocks_per_row = 8;
  d.codebook_id = 19;
  d.source_codebook_id = 19;
  d.source_is_superblock = 0;
  d.source_identity_present = 1;
  d.allocation_payload_bytes_per_block = 255;
  // Supersets are original allocation authority, not a source-specific minimal
  // size.
  d.allocation_has_mins = 1;
  d.allocation_has_emins = 1;
  d.mins = storage.data() + 512;
  d.emins = storage.data() + 768;
  EXPECT_NO_THROW((void)requireQ8ResidualNativeMatrix(d, 1, 5, 256));
  d.n = INT_MAX;
  d.k = INT_MAX - 31;
  d.blocks_per_row = d.k / 32;
  EXPECT_THROW((void)requireQ8ResidualNativeMatrix(d, 1, d.n, d.k),
               std::invalid_argument);
}
/** @test Original32-element source blocks run at the minimum supported K. */
TEST(Q8ResidualAllFormats, OriginalBlock32ActualPreparedStage) {
  for (const auto &format : test::quantizedVerifierFormats()) {
    if (format.source_is_superblock)
      continue;
    test::q8_all_formats::proveCPU(format, 3, 5, 32, 1);
    ASSERT_FALSE(::testing::Test::HasFailure());
  }
}
