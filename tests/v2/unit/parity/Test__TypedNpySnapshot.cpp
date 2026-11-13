/**
 * @file Test__TypedNpySnapshot.cpp
 * @brief Exact integer NumPy roundtrips and dtype-authentication regressions.
 *
 * Private, tiny files exercise the canonical cnpy adapter without models or
 * accelerators. Floating parity keeps its original values; integer identifiers
 * above both IEEE mantissa limits retain every bit and exact mismatch counts.
 */
#include "../../utils/TypedNpySnapshot.h"
#include <gtest/gtest.h>
#include <array>
#include <filesystem>
#include <unistd.h>

namespace llaminar2::test
{
/** @brief One private NumPy file retired before the test process exits. */
class TypedNpyFile
{
public:
    /** @brief Allocate a unique fixture pathname without consuming shared models. */
    TypedNpyFile()
    {
        char name[] = "/tmp/llaminar-typed-npy-XXXXXX";
        const int fd = ::mkstemp(name);
        if (fd < 0) throw std::runtime_error("typed NumPy fixture creation failed");
        ::close(fd);
        path = name;
    }
    /** @brief Remove the fixture regardless of assertion outcome. */
    ~TypedNpyFile() { std::filesystem::remove(path); }
    std::string path;
};
/** @test INT64 is authenticated as integer rather than interpreting eight bytes as double. */
TEST(TypedNpySnapshot, ExactInt64RoundTripAndMismatchCount)
{
    TypedNpyFile file;
    const std::array<std::int64_t, 4> ids{16777217, 9007199254740993LL,
        std::numeric_limits<std::int64_t>::max(), -1};
    cnpy::npy_save(file.path, ids.data(), {2, 2}, "w");
    EXPECT_EQ(inspectNpySnapshot(file.path).scalar, NpySnapshotScalar::INT64);
    const auto values = loadTypedNpySnapshot(file.path);
    EXPECT_EQ(values.integers(), (std::vector<std::int64_t>(ids.begin(), ids.end())));
    EXPECT_THROW(values.data(), std::logic_error);
    auto altered = ids;
    altered[1] += 1;
    EXPECT_EQ(integerSnapshotMismatches(altered, values.integers()), 1u);
    EXPECT_EQ(integerSnapshotMismatches(std::span(ids).first(3), values.integers()), 1u);
}
/** @test FP32 and FP64 numerical snapshots preserve the historical FP32 comparison contract. */
TEST(TypedNpySnapshot, FloatingPayloadsKeepTheirNumericInterpretation)
{
    TypedNpyFile file;
    const std::array<float, 3> floats{-1.25f, 0.0f, 3.5f};
    cnpy::npy_save(file.path, floats.data(), {3}, "w");
    EXPECT_EQ(loadTypedNpySnapshot(file.path).floats(), (std::vector<float>(floats.begin(), floats.end())));
    const std::array<double, 3> doubles{-1.25, 0.0, 3.5};
    cnpy::npy_save(file.path, doubles.data(), {3}, "w");
    EXPECT_EQ(inspectNpySnapshot(file.path).scalar, NpySnapshotScalar::FP64);
    EXPECT_EQ(loadTypedNpySnapshot(file.path).floats(), (std::vector<float>(floats.begin(), floats.end())));
}
/** @test Unsupported integer widths and incomplete payloads fail before cnpy can reinterpret bytes. */
TEST(TypedNpySnapshot, InvalidScalarAndTruncatedExtentFailClosed)
{
    TypedNpyFile file;
    const std::array<std::int32_t, 2> ids{17, 29};
    cnpy::npy_save(file.path, ids.data(), {2}, "w");
    EXPECT_THROW(loadTypedNpySnapshot(file.path), std::invalid_argument);
    const std::array<std::int64_t, 2> wide{16777217, 9007199254740993LL};
    cnpy::npy_save(file.path, wide.data(), {2}, "w");
    std::filesystem::resize_file(file.path, std::filesystem::file_size(file.path) - 1);
    EXPECT_THROW(loadTypedNpySnapshot(file.path), std::invalid_argument);
}
} // namespace llaminar2::test

namespace llaminar2::test
{
/** @test Endian, ordering and unsupported dependency-header versions reject before payload loading. */
TEST(TypedNpySnapshot, UnsupportedHeaderContractsRejectBeforeCanonicalLoader)
{
    TypedNpyFile file;
    const std::array<std::int64_t, 2> ids{16777217, 9007199254740993LL};
    for (int defect = 0; defect < 3; ++defect)
    {
        cnpy::npy_save(file.path, ids.data(), {2}, "w");
        std::ifstream source(file.path, std::ios::binary);
        std::string bytes((std::istreambuf_iterator<char>(source)), {});
        source.close();
        if (defect == 0) bytes[bytes.find("<i8")] = '>';
        if (defect == 1) bytes.replace(bytes.find("False"), 5, "True ");
        if (defect == 2) bytes[6] = 2;
        std::ofstream target(file.path, std::ios::binary | std::ios::trunc);
        target.write(bytes.data(), bytes.size());
        target.close();
        EXPECT_THROW(loadTypedNpySnapshot(file.path), std::invalid_argument);
    }
}
} // namespace llaminar2::test
