/**
 * @file stub.c
 * @brief Empty translation unit for the absent-dependency link stub.
 *
 * The missing-dependency fixture links this stub and records its SONAME as a
 * NEEDED entry. The file has a different name so the loader cannot resolve that
 * dependency in the build tree. Test runner packaging discards the stub.
 */
