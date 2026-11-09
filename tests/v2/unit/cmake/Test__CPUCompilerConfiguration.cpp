/**
 * @file Test__CPUCompilerConfiguration.cpp
 * @brief Execute genuine separately compiled C/C++ configuration witnesses in ordinary and installed gates.
 *
 * Compiler macros describe the actual executable, not requested cache strings.
 * This complements the real CMake cache-state regression and authenticates
 * optimized Release/Integration versus Debug without compiling at test runtime.
 */
#include <gtest/gtest.h>

/** @brief Read optimization/assertion bits from the separately compiled actual C TU.
 * @return Bit0 for compiler optimization and bit1 for NDEBUG.
 */
extern "C" unsigned llaminar_test_c_compiler_configuration(void);

#ifndef LLAMINAR_TEST_EXPECT_OPTIMIZED_CONFIGURATION
#error "Compiler-configuration witness requires the typed CMake configuration expectation"
#endif

TEST(CPUCompilerConfiguration,RealCAndCXXMatchDeclaredBuildType)
{
    constexpr unsigned expected=LLAMINAR_TEST_EXPECT_OPTIMIZED_CONFIGURATION?3u:0u;
    unsigned cpp=0;
#if defined(__OPTIMIZE__)
    cpp|=1u;
#endif
#if defined(NDEBUG)
    cpp|=2u;
#endif
    EXPECT_EQ(llaminar_test_c_compiler_configuration(),expected);
    EXPECT_EQ(cpp,expected);
}
