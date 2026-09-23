/**
 * @file Test__PipelineMTPMainForward.cpp
 * @brief Real captured speculative pipeline transactions and independent token proof.
 *
 * The shared fixture owns real kernels and immutable weights; test registration
 * stays in the canonical pipeline preflight target across translation units.
 */
#include "PipelineGenerationTestSupport.h"
#include "config/OrchestrationStartupPolicy.h"

namespace llaminar2::test
{
#include "Test__PipelineMTPMainForward.inc"
}
