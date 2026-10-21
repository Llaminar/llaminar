/**
 * @file Test__GGUFIntegerArrays.cpp
 * @brief Exact metadata-array regressions including production GGUF parser subtype retention.
 *
 * Tiny generated files contain metadata only. These tests prevent 45-bit PLE
 * multipliers from passing through float or a guessed 32-bit array element type.
 */
#include "../models/qwen4exp/Qwen4ExpMetadataFixture.h"
#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>
#include <limits>
#include <unistd.h>

namespace llaminar2::test
{
    namespace
    {
        /** @brief Own a unique metadata-only temporary GGUF and remove it on every test exit. */
        class IntegerMetadataFile final
        {
        public:
            /** @brief Reserve a unique path without relying on a process-global fixture name. */
            IntegerMetadataFile()
            {
                std::string pattern = (std::filesystem::temp_directory_path() / "llaminar-integer-metadata-XXXXXX").string();
                const int descriptor = mkstemp(pattern.data());
                if (descriptor < 0)
                    throw std::runtime_error("cannot create GGUF metadata fixture");
                close(descriptor);
                path = pattern;
            }
            /** @brief Retire only the file owned by this fixture. */
            ~IntegerMetadataFile() { std::error_code error; std::filesystem::remove(path, error); }
            std::filesystem::path path;
        };

        /** @brief Serialize a fixed-width GGUF field using the loader's native file representation. */
        template <typename T>
        void field(std::ofstream &stream, T value)
        {
            stream.write(reinterpret_cast<const char *>(&value), sizeof(value));
        }

        /** @brief Serialize one length-prefixed GGUF string. */
        void stringField(std::ofstream &stream, const std::string &value)
        {
            field(stream, uint64_t{value.size()});
            stream.write(value.data(), value.size());
        }
    }

    TEST(GGUFIntegerArrays, PreservesEveryIntegerSubtypeThroughProductionParser)
    {
        const std::vector<std::pair<std::string, GGUFValue>> arrays{
            {"uint8", qwen4Array<uint8_t>(GGUFValueType::UINT8, {0, 255})},
            {"int8", qwen4Array<int8_t>(GGUFValueType::INT8, {-128, 127})},
            {"uint16", qwen4Array<uint16_t>(GGUFValueType::UINT16, {0, 65535})},
            {"int16", qwen4Array<int16_t>(GGUFValueType::INT16, {-32768, 32767})},
            {"uint32", qwen4Array<uint32_t>(GGUFValueType::UINT32, {0, UINT32_MAX})},
            {"int32", qwen4Array<int32_t>(GGUFValueType::INT32, {INT32_MIN, INT32_MAX})},
            {"uint64", qwen4Array<uint64_t>(GGUFValueType::UINT64, {(uint64_t{1} << 45) + 7, UINT64_MAX})},
            {"int64", qwen4Array<int64_t>(GGUFValueType::INT64, {INT64_MIN, INT64_MAX})}};
        IntegerMetadataFile fixture;
        {
            std::ofstream stream(fixture.path, std::ios::binary);
            field(stream, uint32_t{0x46554747});
            field(stream, uint32_t{3});
            field(stream, uint64_t{0});
            field(stream, uint64_t{arrays.size() + 1});
            stringField(stream, "general.architecture");
            field(stream, static_cast<uint32_t>(GGUFValueType::STRING));
            stringField(stream, "metadata-test");
            for (const auto &[name, value] : arrays)
            {
                stringField(stream, name);
                field(stream, static_cast<uint32_t>(GGUFValueType::ARRAY));
                field(stream, static_cast<uint32_t>(*value.array_element_type));
                field(stream, value.array_length);
                stream.write(reinterpret_cast<const char *>(value.data.data()), value.data.size());
            }
        }
        ModelLoader loader;
        loader.setUseMmap(false);
        ASSERT_TRUE(loader.loadModel(fixture.path.string()));
        for (const auto &[name, expected] : arrays)
        {
            SCOPED_TRACE(name);
            const auto &actual = loader.getModel().metadata.at(name);
            EXPECT_EQ(actual.array_element_type, expected.array_element_type);
            EXPECT_EQ(actual.array_length, expected.array_length);
            EXPECT_EQ(actual.data, expected.data);
        }
        EXPECT_EQ(loader.getModel().metadata.at("uint64").asUInt64Array(),
                  (std::vector<uint64_t>{(uint64_t{1} << 45) + 7, UINT64_MAX}));
        EXPECT_EQ(loader.getModel().metadata.at("int64").asInt64Array(),
                  (std::vector<int64_t>{INT64_MIN, INT64_MAX}));
    }

    TEST(GGUFIntegerArrays, ChecksSignedUnsignedConversionWithoutRounding)
    {
        EXPECT_EQ(qwen4Array<int32_t>(GGUFValueType::INT32, {0, INT32_MAX}).asUInt64Array(),
                  (std::vector<uint64_t>{0, INT32_MAX}));
        EXPECT_EQ(qwen4Array<uint32_t>(GGUFValueType::UINT32, {UINT32_MAX}).asInt64Array(),
                  (std::vector<int64_t>{UINT32_MAX}));
        EXPECT_THROW(qwen4Array<int64_t>(GGUFValueType::INT64, {-1}).asUInt64Array(), std::invalid_argument);
        EXPECT_THROW(qwen4Array<uint64_t>(GGUFValueType::UINT64, {uint64_t{1} << 63}).asInt64Array(), std::invalid_argument);
        EXPECT_EQ(qwen4Scalar(GGUFValueType::UINT64, UINT64_MAX).asUInt64Checked(), UINT64_MAX);
        EXPECT_THROW(qwen4Scalar(GGUFValueType::INT64, int64_t{-1}).asUInt64Checked(), std::invalid_argument);
    }

    TEST(GGUFIntegerArrays, RejectsMissingSubtypeWrongKindsAndMalformedLengths)
    {
        auto value = qwen4Array<uint64_t>(GGUFValueType::UINT64, {1, 2});
        value.array_element_type.reset();
        EXPECT_THROW(value.asUInt64Array(), std::invalid_argument);
        value = qwen4Array<uint64_t>(GGUFValueType::UINT64, {1, 2});
        value.data.pop_back();
        EXPECT_THROW(value.asUInt64Array(), std::invalid_argument);
        value = qwen4Array<uint64_t>(GGUFValueType::UINT64, {1});
        value.array_length = UINT64_MAX;
        EXPECT_THROW(value.asUInt64Array(), std::invalid_argument);
        for (auto type : {GGUFValueType::FLOAT32, GGUFValueType::FLOAT64,
                          GGUFValueType::BOOL, GGUFValueType::STRING, GGUFValueType::ARRAY})
        {
            value = qwen4Array<uint32_t>(type, {1});
            EXPECT_THROW(value.asUInt64Array(), std::invalid_argument);
        }
        value = qwen4Scalar(GGUFValueType::UINT64, uint64_t{1});
        EXPECT_THROW(value.asInt64Array(), std::invalid_argument);
        EXPECT_TRUE(qwen4Array<uint64_t>(GGUFValueType::UINT64, {}).asUInt64Array().empty());
    }
}
