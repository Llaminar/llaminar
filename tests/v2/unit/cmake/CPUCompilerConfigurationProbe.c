/**
 * @file CPUCompilerConfigurationProbe.c
 * @brief Preserve the actual C compiler's optimization/assertion configuration in a test binary.
 *
 * C and C++ are independent compiler envelopes. Returning preprocessor-observed
 * bits from this genuine C TU prevents a C++-only build label from certifying C
 * helpers. Runtime consumers require neither a compiler SDK nor executable /tmp.
 */

/** @brief Report genuine C optimization/NDEBUG bits, rather than a CMake-advertised value.
 * @return Bit0 for compiler optimization and bit1 for disabled standard assertions.
 */
unsigned llaminar_test_c_compiler_configuration(void);
unsigned llaminar_test_c_compiler_configuration(void)
{
    unsigned bits=0;
#if defined(__OPTIMIZE__)
    bits|=1u;
#endif
#if defined(NDEBUG)
    bits|=2u;
#endif
    return bits;
}
