/**
 * @file OriginalQ8ResidualProjectionSource.h
 * @brief Bind saved native projection bytes through the original typed tensor factory.
 *
 * Q8_1 has a distinct existing raw-byte factory API because it is also an
 * activation format. Selecting that API preserves its integer-sum fields;
 * generic quantized construction intentionally lacks that switch entry.
 * Every other source keeps the original generic factory, including Q8_K's
 * full FP32-scale superblocks. This helper never quantizes or rewrites bytes.
 */
#pragma once
#include "tensors/TensorFactory.h"

namespace llaminar2::test::q8_all_formats
{
/**
 * @brief Reconstruct one original saved native source without changing its codebook.
 * @param factory Existing metadata-only CPU tensor factory.
 * @param type Exact authored native source type.
 * @param shape Original source matrix dimensions.
 * @param raw Original native source bytes, including all compensation fields.
 * @return Tensor owning those exact native bytes through its supported factory.
 * @throws std::exception If the selected original factory rejects type or geometry.
 */
inline std::unique_ptr<TensorBase> originalProjectionSource(
    TensorFactory &factory, TensorType type, const std::vector<std::size_t> &shape,
    const std::vector<std::uint8_t> &raw)
{
    if(type==TensorType::Q8_1)return factory.createQ8_1(shape,raw);
    return factory.createQuantized(type,shape,raw);
}
}
