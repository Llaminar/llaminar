/**
 * @file Test__BPETokenizerByteEncoding.cpp
 * @brief Unit tests for BPETokenizer byte encoding/decoding
 * @author David Sanftenberg
 * @date 2025-11-28
 *
 * Tests the GPT-2 style byte-level encoding/decoding that underlies
 * the BPE tokenizer. These are pure unit tests that don't require
 * model loading.
 */

#include <gtest/gtest.h>
#include <string>
#include <vector>
#include <unordered_map>
#include <iomanip>

namespace
{
    // Replicate the GPT-2 byte encoder logic for testing
    class ByteEncoderTest
    {
    public:
        ByteEncoderTest()
        {
            initializeByteEncoder();
        }

        std::string encode(unsigned char byte) const
        {
            return byte_encoder_[byte];
        }

        int decode(const std::string &encoded) const
        {
            auto it = byte_decoder_.find(encoded);
            if (it != byte_decoder_.end())
            {
                return it->second;
            }
            return -1;
        }

        const std::vector<std::string> &encoder() const { return byte_encoder_; }
        const std::unordered_map<std::string, int> &decoder() const { return byte_decoder_; }

    private:
        void initializeByteEncoder()
        {
            byte_encoder_.resize(256);
            int n = 0;
            for (int b = 0; b < 256; ++b)
            {
                if (b >= 33 && b <= 126)
                {
                    // Printable ASCII: single-byte identity mapping
                    byte_encoder_[b] = std::string(1, static_cast<char>(b));
                    byte_decoder_[std::string(1, static_cast<char>(b))] = b;
                }
                else if ((b >= 161 && b <= 172) || (b >= 174 && b <= 255))
                {
                    // Latin-1 Supplement: identity mapping but needs UTF-8 encoding
                    int codepoint = b;
                    char utf8[3];
                    utf8[0] = static_cast<char>(0xC0 | ((codepoint >> 6) & 0x1F));
                    utf8[1] = static_cast<char>(0x80 | (codepoint & 0x3F));
                    utf8[2] = '\0';
                    std::string encoded(utf8, 2);
                    byte_encoder_[b] = encoded;
                    byte_decoder_[encoded] = b;
                }
                else
                {
                    // Non-printable: map to Unicode U+0100 + n
                    int codepoint = 256 + n;
                    char utf8[3];
                    utf8[0] = static_cast<char>(0xC0 | ((codepoint >> 6) & 0x1F));
                    utf8[1] = static_cast<char>(0x80 | (codepoint & 0x3F));
                    utf8[2] = '\0';
                    std::string encoded(utf8, 2);
                    byte_encoder_[b] = encoded;
                    byte_decoder_[encoded] = b;
                    n++;
                }
            }
        }

        std::vector<std::string> byte_encoder_;
        std::unordered_map<std::string, int> byte_decoder_;
    };

    // =============================================================================
    // Printable ASCII Tests (bytes 33-126)
    // =============================================================================

    TEST(ByteEncodingTest, PrintableASCII_Identity)
    {
        ByteEncoderTest encoder;

        // Printable ASCII should map to single-byte strings
        for (int b = 33; b <= 126; ++b)
        {
            std::string encoded = encoder.encode(static_cast<unsigned char>(b));
            EXPECT_EQ(encoded.size(), 1) << "Byte " << b << " should encode to single byte";
            EXPECT_EQ(static_cast<unsigned char>(encoded[0]), b)
                << "Byte " << b << " should be identity mapped";
        }
    }

    TEST(ByteEncodingTest, PrintableASCII_RoundTrip)
    {
        ByteEncoderTest encoder;

        for (int b = 33; b <= 126; ++b)
        {
            std::string encoded = encoder.encode(static_cast<unsigned char>(b));
            int decoded = encoder.decode(encoded);
            EXPECT_EQ(decoded, b) << "Byte " << b << " should round-trip correctly";
        }
    }

    TEST(ByteEncodingTest, PrintableASCII_SpecificChars)
    {
        ByteEncoderTest encoder;

        // Test specific important characters
        EXPECT_EQ(encoder.encode('!'), "!");  // 33
        EXPECT_EQ(encoder.encode('"'), "\""); // 34
        EXPECT_EQ(encoder.encode('+'), "+");  // 43
        EXPECT_EQ(encoder.encode('.'), ".");  // 46
        EXPECT_EQ(encoder.encode('0'), "0");  // 48
        EXPECT_EQ(encoder.encode('A'), "A");  // 65
        EXPECT_EQ(encoder.encode('a'), "a");  // 97
        EXPECT_EQ(encoder.encode('~'), "~");  // 126
    }

    // =============================================================================
    // Latin-1 Supplement Tests (bytes 161-172, 174-255)
    // =============================================================================

    TEST(ByteEncodingTest, Latin1Supplement_UTF8Encoding)
    {
        ByteEncoderTest encoder;

        // Latin-1 Supplement bytes should encode as 2-byte UTF-8
        for (int b = 161; b <= 172; ++b)
        {
            std::string encoded = encoder.encode(static_cast<unsigned char>(b));
            EXPECT_EQ(encoded.size(), 2) << "Byte " << b << " should encode to 2-byte UTF-8";

            // Verify UTF-8 structure: 110xxxxx 10xxxxxx
            unsigned char b1 = static_cast<unsigned char>(encoded[0]);
            unsigned char b2 = static_cast<unsigned char>(encoded[1]);
            EXPECT_EQ(b1 & 0xE0, 0xC0) << "First byte should start with 110";
            EXPECT_EQ(b2 & 0xC0, 0x80) << "Second byte should start with 10";
        }

        for (int b = 174; b <= 255; ++b)
        {
            std::string encoded = encoder.encode(static_cast<unsigned char>(b));
            EXPECT_EQ(encoded.size(), 2) << "Byte " << b << " should encode to 2-byte UTF-8";
        }
    }

    TEST(ByteEncodingTest, Latin1Supplement_RoundTrip)
    {
        ByteEncoderTest encoder;

        for (int b = 161; b <= 172; ++b)
        {
            std::string encoded = encoder.encode(static_cast<unsigned char>(b));
            int decoded = encoder.decode(encoded);
            EXPECT_EQ(decoded, b) << "Byte " << b << " should round-trip correctly";
        }

        for (int b = 174; b <= 255; ++b)
        {
            std::string encoded = encoder.encode(static_cast<unsigned char>(b));
            int decoded = encoder.decode(encoded);
            EXPECT_EQ(decoded, b) << "Byte " << b << " should round-trip correctly";
        }
    }

    TEST(ByteEncodingTest, Latin1Supplement_SpecificBytes)
    {
        ByteEncoderTest encoder;

        // Byte 0xE4 (228) = ä (U+00E4) = UTF-8 C3 A4
        std::string encoded_e4 = encoder.encode(0xE4);
        EXPECT_EQ(encoded_e4.size(), 2);
        EXPECT_EQ(static_cast<unsigned char>(encoded_e4[0]), 0xC3);
        EXPECT_EQ(static_cast<unsigned char>(encoded_e4[1]), 0xA4);

        // Byte 0xE5 (229) = å (U+00E5) = UTF-8 C3 A5
        std::string encoded_e5 = encoder.encode(0xE5);
        EXPECT_EQ(encoded_e5.size(), 2);
        EXPECT_EQ(static_cast<unsigned char>(encoded_e5[0]), 0xC3);
        EXPECT_EQ(static_cast<unsigned char>(encoded_e5[1]), 0xA5);

        // Byte 0xBD (189) = ½ (U+00BD) = UTF-8 C2 BD
        std::string encoded_bd = encoder.encode(0xBD);
        EXPECT_EQ(encoded_bd.size(), 2);
        EXPECT_EQ(static_cast<unsigned char>(encoded_bd[0]), 0xC2);
        EXPECT_EQ(static_cast<unsigned char>(encoded_bd[1]), 0xBD);
    }

    // =============================================================================
    // Non-Printable Byte Tests (0-32, 127-160, 173)
    // =============================================================================

    TEST(ByteEncodingTest, NonPrintable_MapsToU0100Plus)
    {
        ByteEncoderTest encoder;

        // Non-printable bytes map to U+0100 + n
        // First non-printable is byte 0, should map to U+0100
        std::string encoded_0 = encoder.encode(0);
        EXPECT_EQ(encoded_0.size(), 2);
        // U+0100 = UTF-8 C4 80
        EXPECT_EQ(static_cast<unsigned char>(encoded_0[0]), 0xC4);
        EXPECT_EQ(static_cast<unsigned char>(encoded_0[1]), 0x80);

        // Byte 32 (space) should map to U+0120 = UTF-8 C4 A0
        std::string encoded_space = encoder.encode(' ');
        EXPECT_EQ(encoded_space.size(), 2);
        EXPECT_EQ(static_cast<unsigned char>(encoded_space[0]), 0xC4);
        EXPECT_EQ(static_cast<unsigned char>(encoded_space[1]), 0xA0);
    }

    TEST(ByteEncodingTest, NonPrintable_RoundTrip)
    {
        ByteEncoderTest encoder;

        // Test all non-printable ranges
        for (int b = 0; b <= 32; ++b)
        {
            std::string encoded = encoder.encode(static_cast<unsigned char>(b));
            int decoded = encoder.decode(encoded);
            EXPECT_EQ(decoded, b) << "Byte " << b << " should round-trip correctly";
        }

        // Byte 127 (DEL)
        EXPECT_EQ(encoder.decode(encoder.encode(127)), 127);

        // Bytes 128-160
        for (int b = 128; b <= 160; ++b)
        {
            std::string encoded = encoder.encode(static_cast<unsigned char>(b));
            int decoded = encoder.decode(encoded);
            EXPECT_EQ(decoded, b) << "Byte " << b << " should round-trip correctly";
        }

        // Byte 173 (soft hyphen)
        EXPECT_EQ(encoder.decode(encoder.encode(173)), 173);
    }

    // =============================================================================
    // UTF-8 Multi-Byte Sequence Tests (Chinese, Emoji, etc.)
    // =============================================================================

    TEST(ByteEncodingTest, ChineseCharacter_ByteSequence)
    {
        ByteEncoderTest encoder;

        // 你 = UTF-8: E4 BD A0
        std::string ni = "你";
        ASSERT_EQ(ni.size(), 3);

        unsigned char b1 = static_cast<unsigned char>(ni[0]); // 0xE4
        unsigned char b2 = static_cast<unsigned char>(ni[1]); // 0xBD
        unsigned char b3 = static_cast<unsigned char>(ni[2]); // 0xA0

        EXPECT_EQ(b1, 0xE4);
        EXPECT_EQ(b2, 0xBD);
        EXPECT_EQ(b3, 0xA0);

        // Each byte should encode and round-trip
        EXPECT_EQ(encoder.decode(encoder.encode(b1)), b1);
        EXPECT_EQ(encoder.decode(encoder.encode(b2)), b2);
        EXPECT_EQ(encoder.decode(encoder.encode(b3)), b3);
    }

    TEST(ByteEncodingTest, ChineseCharacter_FullRoundTrip)
    {
        ByteEncoderTest encoder;

        // Test encoding "你好" byte by byte and reconstructing
        std::string original = "你好";
        std::string encoded_combined;

        for (unsigned char c : original)
        {
            encoded_combined += encoder.encode(c);
        }

        // Now decode back
        std::string decoded;
        size_t i = 0;
        while (i < encoded_combined.size())
        {
            unsigned char b1 = static_cast<unsigned char>(encoded_combined[i]);

            if ((b1 & 0xE0) == 0xC0 && i + 1 < encoded_combined.size())
            {
                // 2-byte UTF-8 sequence
                std::string utf8_char = encoded_combined.substr(i, 2);
                int byte_val = encoder.decode(utf8_char);
                if (byte_val >= 0)
                {
                    decoded += static_cast<char>(byte_val);
                }
                i += 2;
            }
            else
            {
                // Single byte
                std::string single(1, encoded_combined[i]);
                int byte_val = encoder.decode(single);
                if (byte_val >= 0)
                {
                    decoded += static_cast<char>(byte_val);
                }
                i++;
            }
        }

        EXPECT_EQ(decoded, original) << "Chinese text should round-trip correctly";
    }

    TEST(ByteEncodingTest, Emoji_ByteSequence)
    {
        ByteEncoderTest encoder;

        // 😀 = UTF-8: F0 9F 98 80 (4 bytes)
        std::string emoji = "😀";
        ASSERT_EQ(emoji.size(), 4);

        // Each byte should round-trip
        for (unsigned char c : emoji)
        {
            EXPECT_EQ(encoder.decode(encoder.encode(c)), c)
                << "Byte 0x" << std::hex << (int)c << " should round-trip";
        }
    }

    // =============================================================================
    // Decoder Map Integrity Tests
    // =============================================================================

    TEST(ByteEncodingTest, DecoderHas256Entries)
    {
        ByteEncoderTest encoder;
        EXPECT_EQ(encoder.decoder().size(), 256)
            << "Decoder should have exactly 256 entries (one per byte)";
    }

    TEST(ByteEncodingTest, EncoderDecoderBijection)
    {
        ByteEncoderTest encoder;

        // Each byte should have a unique encoding
        std::unordered_map<std::string, int> seen;
        for (int b = 0; b < 256; ++b)
        {
            std::string encoded = encoder.encode(static_cast<unsigned char>(b));
            auto it = seen.find(encoded);
            EXPECT_EQ(it, seen.end())
                << "Encoding for byte " << b << " collides with byte " << it->second;
            seen[encoded] = b;
        }
    }

    TEST(ByteEncodingTest, AllBytesRoundTrip)
    {
        ByteEncoderTest encoder;

        for (int b = 0; b < 256; ++b)
        {
            std::string encoded = encoder.encode(static_cast<unsigned char>(b));
            int decoded = encoder.decode(encoded);
            EXPECT_EQ(decoded, b) << "Byte " << b << " should round-trip";
        }
    }

} // anonymous namespace

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
