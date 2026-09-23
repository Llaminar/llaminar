/**
 * @file Test__PipelineOrdinaryGeneration.cpp
 * @brief Real captured ordinary generation and prefix lifecycle proofs.
 *
 * The shared fixture owns real kernels and immutable weights; test registration
 * stays in the canonical pipeline preflight target across translation units.
 */
#include "PipelineGenerationTestSupport.h"

namespace llaminar2::test
{
#include "Test__OrdinaryDGOGeneration.inc"
}
