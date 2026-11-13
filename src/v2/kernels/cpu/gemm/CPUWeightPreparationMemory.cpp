/**
 * @file CPUWeightPreparationMemory.cpp
 * @brief Exact native CPU packer payload geometry for admitted startup samples.
 *
 * Encoding selection and interleaved strides come from the production packer.
 * Its one retained root uses the common native allocator's physical geometry.
 * Setup demand describes the actual bounded append workspace;
 * it is not a multiplier over GGUF bytes or an anonymous safety reserve.
 */
#include "CPUWeightPreparationMemory.h"
#include "CPUNativeQuantisedWeightPacker.h"
#include "memory/CPUStorageRoot.h"
#include <climits>
#include <limits>
#include <stdexcept>

namespace llaminar2
{
    CPUWeightPreparationMemory CPUWeightPreparationMemory::sourceNative(
        std::string_view format, size_t n, size_t k)
    {
        const auto multiply = [](size_t a, size_t b) {
            if (b && a > std::numeric_limits<size_t>::max() / b)
                throw std::overflow_error("CPU preparation byte extent overflow");
            return a * b;
        };
        // The packer uses signed-int padded geometry. Reject before its N+63
        // or K+31 arithmetic can wrap, even on a host with 64-bit size_t.
        if (!n || !k || n > size_t(INT_MAX - 63) || k > size_t(INT_MAX - 31))
            throw std::invalid_argument("CPU preparation requires representable positive matrix geometry");
        if (format == "F32" || format == "F16" || format == "BF16")
        {
            const auto elements=multiply(n,k);
            const auto payload=multiply(elements,format=="F32"?sizeof(float):sizeof(uint16_t));
            const auto physical=format=="F32"
                ? CPUStoragePlacement::local().physicalBytes<float>(elements)
                : CPUStoragePlacement::local().physicalBytes<uint16_t>(elements);
            return {physical,0,payload};
        }

        const auto *native = native_quantised_formats::forQuantType(format);
        if (!native || k % (native->is_superblock ? 256u : 32u))
            throw std::invalid_argument("CPU preparation requires a supported, block-aligned source format");
        using namespace cpu::native_quantised;
        const auto footprint = preparedFootprintForFormat(*native);
        const size_t units = multiply((n + 63) / 64, k / 32);
        const size_t interleaved = multiply(units, footprint.weight_bytes_per_n_chunk_k_block);
        const auto retained=CPUStoragePlacement::local().physicalBytes<uint8_t>(interleaved);
        // The immediate path borrows the caller's original native tensor and
        // uses the same bounded append workspace for every source codebook.
        const auto workspace=immediateNativePackScratchGeometry(*native,static_cast<int>(n),
            static_cast<int>(k),CPUStoragePlacement::local());
        return {retained,workspace.physical_bytes,interleaved};
    }
}
