#include <gtest/gtest.h>
#include "../../../external/onednn/third_party/xbyak/xbyak.h"

using namespace Xbyak;

class TestJit : public CodeGenerator
{
public:
    TestJit()
    {
        // Function signature: void func(int32_t* input, int8_t* output)
        // rdi: input
        // rsi: output

        // Load 16 int32 values
        vmovdqu32(zmm0, ptr[rdi]);

        // Pack to 16 int8 values using vpmovdb
        vpmovdb(xmm1, zmm0);

        // Store result
        vmovdqu(ptr[rsi], xmm1);

        ret();
    }
};

class TestJitSaturating : public CodeGenerator
{
public:
    TestJitSaturating()
    {
        // Function signature: void func(int32_t* input, int8_t* output)
        // rdi: input
        // rsi: output

        // Load 16 int32 values
        vmovdqu32(zmm0, ptr[rdi]);

        // Pack to 16 int8 values using vpmovsdb (Signed Saturation)
        vpmovsdb(xmm1, zmm0);

        // Store result
        vmovdqu(ptr[rsi], xmm1);

        ret();
    }
};

TEST(XbyakTest, VPMOVDB_Behavior)
{
    TestJit jit;
    auto func = jit.getCode<void (*)(int32_t *, int8_t *)>();

    int32_t input[16];
    int8_t output[16];

    // Test value 130 (0x82)
    // If truncating: 0x82 -> -126
    // If saturating: 127
    for (int i = 0; i < 16; ++i)
        input[i] = 130;

    func(input, output);

    printf("Input: 130\n");
    printf("Output: %d\n", (int)output[0]);

    if (output[0] == -126)
    {
        printf("vpmovdb is TRUNCATING\n");
    }
    else if (output[0] == 127)
    {
        printf("vpmovdb is SATURATING\n");
    }
    else
    {
        printf("vpmovdb is UNKNOWN: %d\n", (int)output[0]);
    }
}

TEST(XbyakTest, VPMOVSDB_Behavior)
{
    TestJitSaturating jit;
    auto func = jit.getCode<void (*)(int32_t *, int8_t *)>();

    int32_t input[16];
    int8_t output[16];

    // Test value 130 (0x82)
    for (int i = 0; i < 16; ++i)
        input[i] = 130;

    func(input, output);

    printf("Input: 130\n");
    printf("Output: %d\n", (int)output[0]);

    if (output[0] == 127)
    {
        printf("vpmovsdb is SATURATING\n");
    }
    else
    {
        printf("vpmovsdb is NOT SATURATING: %d\n", (int)output[0]);
    }
}
