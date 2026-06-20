#include <gtest/gtest.h>

#include "execution/prefix_cache/BlockHash.h"

#include <vector>

using namespace llaminar2;

TEST(Test__PrefixBlockHash, ByteHashIsDeterministicAndContentSensitive)
{
    const char a[] = "prefix-a";
    const char b[] = "prefix-b";

    EXPECT_EQ(hashPrefixBytes(a, sizeof(a)), hashPrefixBytes(a, sizeof(a)));
    EXPECT_NE(hashPrefixBytes(a, sizeof(a)), hashPrefixBytes(b, sizeof(b)));
    EXPECT_NE(hashPrefixBytes(a, sizeof(a), 7), hashPrefixBytes(a, sizeof(a), 8));
}

TEST(Test__PrefixBlockHash, TokenHashUsesTokenBytes)
{
    const std::vector<int32_t> first = {1, 2, 3, 4};
    const std::vector<int32_t> second = {1, 2, 3, 5};

    EXPECT_EQ(hashPrefixTokens(first), hashPrefixTokens(first));
    EXPECT_NE(hashPrefixTokens(first), hashPrefixTokens(second));
    EXPECT_EQ(prefixHashHex(0xabcull), "0000000000000abc");
}
