/**
 * @file Tensors.h
 * @brief Tensor classes - convenience header
 *
 * This file provides the tensor implementation classes by including TensorClasses.h.
 * Code can include either "Tensors.h" or "TensorClasses.h" to access TensorBase
 * and all concrete tensor types (FP32Tensor, Q4_0Tensor, etc.).
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#pragma once

// Include the tensor class implementations
#include "TensorClasses.h"

// TurboQuant KV cache tensor types (separate headers due to TurboQuantContext dependency)
#include "TQ4Tensor.h"
#include "TQ8Tensor.h"
