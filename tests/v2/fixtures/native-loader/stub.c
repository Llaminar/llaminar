/**
 * @file stub.c
 * @brief Empty translation unit for the absent-dependency link stub.
 *
 * The stub exists only during the fixture link so the missing-dependency
 * fixture records a NEEDED entry for a library this tree owns. It is removed
 * after the link and never ships, which makes the recorded dependency
 * genuinely unresolvable in every installed image.
 */
