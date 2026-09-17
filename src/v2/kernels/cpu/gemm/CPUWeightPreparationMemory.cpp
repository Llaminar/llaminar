/**
 * @file CPUWeightPreparationMemory.cpp
 * @brief Exact native CPU packer payload geometry for admitted startup samples.
 *
 * Encoding selection and interleaved strides come from the production packer.
 * The named temporary terms mirror its source-oriented passes. They are not a
 * multiplier over GGUF bytes or an anonymous safety reserve. Tests compare the
 * retained demand with prepared engines across the entire native-format catalog.
 */
#include "CPUWeightPreparationMemory.h"
#include "CPUNativeVNNIWeightPacker.h"
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
            return {multiply(multiply(n, k), format == "F32" ? sizeof(float) : sizeof(uint16_t)), 0};

        const auto *native = native_vnni_formats::forQuantType(format);
        if (!native || k % (native->is_superblock ? 256u : 32u))
            throw std::invalid_argument("CPU preparation requires a supported, block-aligned source format");
        using namespace cpu::native_vnni;
        const auto footprint = preparedFootprintForFormat(native->codebook_id, native->is_asymmetric);
        const size_t units = multiply((n + 63) / 64, k / 32);
        const size_t blocks = multiply(units, 64);
        const size_t interleaved = multiply(units, footprint.weight_bytes_per_n_chunk_k_block);
        switch (footprint.encoding)
        {
        case CPUNativeVNNIEncoding::NibbleLUT:
        {
            // A retained scalar-oracle payload plus one temporary source-order
            // payload, and two scale/min pairs, coexist with final interleaving.
            const size_t payload = multiply(blocks, native->payload_bytes);
            if (payload > std::numeric_limits<size_t>::max() - interleaved)
                throw std::overflow_error("CPU retained payload extent overflow");
            return {interleaved + payload,
                multiply(blocks, native->payload_bytes + 4 * sizeof(uint16_t))};
        }
        case CPUNativeVNNIEncoding::Q6KNativeDualScale:
            return {interleaved,
                multiply(blocks, native->payload_bytes + 2 * sizeof(uint16_t))};
        case CPUNativeVNNIEncoding::ExpandedInt8:
        case CPUNativeVNNIEncoding::CompactMultiScale:
            // These paths already pack bounded stack blocks directly into the
            // final allocation, without a matrix-sized decoded shadow.
            return {interleaved, 0};
        }
        throw std::logic_error("CPU preparation returned an unknown physical encoding");
    }
}
