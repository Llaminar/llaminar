/**
 * @file Test__Q16_1TensorExtent.cpp
 * @brief Q16_1 storage extents are checked, never wrapped.
 *
 * The mutable constructors computed rows x blocks_per_row x block_bytes in
 * unchecked size_t. A row count such as (2^61 + 2) / 17 with one 64-element
 * block per row wraps 136-byte blocks to a 16-byte buffer while the tensor
 * still publishes the full shape, so later head reads leave the allocation.
 */
#include "tensors/TensorClasses.h"

#include <gtest/gtest.h>
#include <limits>

using namespace llaminar2;

/** @test Wrapping block and byte products are rejected for every block size. */
TEST(Q16_1TensorExtent, WrappingExtentsAreRejected)
{
    const size_t wrapping_rows = ((size_t{1} << 61) + 2) / 17;
    EXPECT_THROW(Q16_1Tensor({wrapping_rows, 64}, Q16BlockSize::BLOCK_64), std::overflow_error);
    const size_t huge = std::numeric_limits<size_t>::max();
    for (auto block : {Q16BlockSize::BLOCK_32, Q16BlockSize::BLOCK_64, Q16BlockSize::BLOCK_128})
    {
        EXPECT_THROW(Q16_1Tensor({huge / 2, 4096}, block), std::overflow_error);
        EXPECT_THROW(Q16_1Tensor({2, huge}, block), std::overflow_error);
    }
    EXPECT_THROW(Q16_1Tensor({huge / 2, 4096}), std::overflow_error);
}

/** @test Representable shapes still construct with live storage. */
TEST(Q16_1TensorExtent, RepresentableExtentsConstruct)
{
    const Q16_1Tensor narrow({3, 65}, Q16BlockSize::BLOCK_64);
    EXPECT_NE(narrow.raw_data(), nullptr);
    const Q16_1Tensor wide({2, 33});
    EXPECT_NE(wide.raw_data(), nullptr);
}
