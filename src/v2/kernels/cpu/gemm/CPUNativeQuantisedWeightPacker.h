/**
 * @file CPUNativeQuantisedWeightPacker.h
 * @brief Repacks tensor weights from native quantized formats into a CPU-cache-friendly
 *        layout for NativeQuantised GEMV/GEMM with inline decode + vpdpbusd accumulation.
 *
 * ## Design
 *
 * Unlike the existing CPUQuantisedGemmKernel path (which pre-decodes ALL formats to INT8),
 * this packer preserves the native weight bytes (Q4_0 nibbles, IQ4_NL nibbles, etc.) and
 * reorganizes them for efficient streaming during GEMV/GEMM. Native signed Q8
 * weights retain their byte payload plus one FP16 scale per block.
 *
 * The kernel decodes blocks inline at compute time, trading a small decode cost for
 * 2-4× less memory traffic on memory-bound GEMV (M=1).
 *
 * ## Packed Layout
 *
 * Weights are arranged in N-blocks of 64 columns for AVX-512 SIMD:
 *
 * The sole retained execution root is [N_chunks][K_blocks][interleaved unit].
 * Native source payload/scale/min words are read through the same bounded
 * native-group append primitive used by streaming preparation. Its scratch is
 * capped by the common source policy and owned by WeightLoadStaging; no whole
 * source-order nibble/Q6 matrix is materialized. Legacy archive payload sections
 * remain import-only and retire at engine publication.
 *
 * This layout ensures:
 * - Sequential K-block streaming within an N-chunk → good prefetch/cache line use
 * - 64 contiguous scales per K-block → aligned ZMM loads
 * - Native payload preserved → 2× less memory for Q4_0 vs INT8
 */

#pragma once

#include "memory/CPUWeightStoragePlacement.h"
#include "loaders/NativeWeightLoadStaging.h"

#include <cstdint>
#include <span>
#include <limits>
#include <cstring>
#include <stdexcept>
#include <vector>

#ifdef __AVX512F__
#include <immintrin.h>
#endif

#include "CPUNativeQuantisedDecode.h"
#include "CPUNativeQuantisedEncoding.h"
#include "CPUNativeQuantisedWeightView.h"
#include "CPUNativeQuantisedPackedWeights.h"
#include "CPUNativeQuantisedPreparedFootprint.h"
#include "kernels/common/MoEProjectionNumericalContract.h"
#include "tensors/AlignedVector.h"
#include "tensors/BlockStructures.h"
#include "tensors/FP16Utils.h"
#include "tensors/TensorClasses.h"
#include "tensors/NativeQuantisedFormatInfo.h"
#include "tensors/NativeQuantisedQ6Scale.h"
#include "tensors/NativeQuantisedIQ2SScale.h"
#include "tensors/NativeQuantisedIQ4XSScale.h"
#include "tensors/MXFP4Format.h"
#include "tensors/NativeQuantisedIQ3SScale.h"
#include "kernels/common/DeviceFP32NumericalContract.h"
#include "utils/Logger.h"

namespace llaminar2::cpu::native_quantised
{

    /**
     * @brief CPU-packed native quantised weights.
     *
     * All buffers are allocated and filled by packWeightsCPUNativeQuantised().
     *
     * ## Layout
     *
     * Newly prepared weights retain one execution representation. Source-order
     * nibble bytes are temporary setup storage; scalar diagnostics reconstruct
     * their bounded block from the same interleaved bytes as AVX2/AVX512.
     * The optional payload member remains solely for archive compatibility.
     *
     * **native_interleaved**: VNNI-interleaved bytes with inline metadata:
     *    `[N_chunks][bpr][ groups×4 ZMMs×64B | 128B comp | 128B scales | 128B mins ]`
     *    Each 64-byte ZMM holds 16 columns × 4 consecutive native payload bytes.
     *    Metadata (comp, scales, mins) is embedded at the end of each K-block
     *    so the entire working set is a single sequential memory stream per thread.
     *    This improves L3 cache utilization for large-K shapes like FFN_Down.
     */
    /**
     * @brief Return the raw source block size for a native-quantised format.
     *
     * Several source formats intentionally share an execution codebook after
     * preparation, so codebook id alone is not sufficient.  The superblock bit
     * disambiguates IQ4_NL/IQ4_XS, Q4_1/Q4_K, and Q5_1/Q5_K.  This metadata is
     * used only by transferred/deferred prepared weights; eager kernels retain
     * their already interleaved representation.
     */
    inline size_t native_block_bytes_for_format(uint8_t codebook_id, bool is_superblock)
    {
        switch (codebook_id)
        {
        case 0:  return 18;  // Q4_0Block
        case 4:  return is_superblock ? 136 : 18; // IQ4_XSBlock / IQ4_NLBlock
        case 5:  return is_superblock ? 144 : 20; // Q4_KBlock / Q4_1Block
        case 6:  return 22;  // Q5_0Block
        case 7:  return is_superblock ? 176 : 24; // Q5_KBlock / Q5_1Block
        case 8:  return 210; // Q6_KBlock
        case 9:  return 110; // Q3_KBlock
        case 10: return 84;  // Q2_KBlock
        case 11: return 110; // IQ3_SBlock
        case 12: return 98;  // IQ3_XXSBlock
        case 13: return 82;  // IQ2_SBlock
        case 14: return 74;  // IQ2_XSBlock
        case 15: return 66;  // IQ2_XXSBlock
        case 16: return 50;  // IQ1_SBlock
        case 17: return 56;  // IQ1_MBlock
        case 19: return 34;  // Q8_0Block
        case 20: return 36;  // Q8_1Block
        case 21: return sizeof(Q8_KBlock); // Original FP32-scale Q8_K source block
        case kNativeQuantisedMXFP4Codebook: return sizeof(MXFP4Block); // E8M0 + 16 code bytes
        default: return 0;
        }
    }



#include "CPUNativeQuantisedWindowPacking.inc"

    /**
     * @brief Exact capped workspace for the immediate consumer of already-resident native bytes.
     * @param format Same canonical append codec, with no second format table.
     * @param n Useful final rows. @param k Original source columns.
     * @param placement Same original allocator placement as final prepared storage.
     * @return Workspace for at most one 64-row chunk, further bounded by the common startup cap.
     */
    inline NativeWeightLoadStagingGeometry immediateNativePackScratchGeometry(
        const NativeQuantisedFormatInfo& format,int n,int k,CPUStoragePlacement placement)
    {
        if(n<=0 || k<=0 || k>std::numeric_limits<int>::max()-31)
            throw std::invalid_argument("Immediate native pack requires representable positive source geometry");
        const auto groups=(size_t(k)+31)/32;
        const auto native_groups=(groups+(format.is_superblock?7:0))/(format.is_superblock?8:1);
        return resolveNativeWeightLoadStaging({NativePackGroupScratch::checked(format).bytes,0,0,
            std::min(size_t(n),size_t{64})*native_groups},placement,configuredNativeWeightLoadStagingBudgetBytes());
    }

    /**
     * @brief Pack tensor weights into CPUNativeQuantisedPackedWeights.
     *
     * Uses the IINT8Unpackable interface to extract native block data
     * from any supported quantized tensor format.
     *
     * @param weights     Source tensor (must implement IINT8Unpackable + vnniFormatInfo)
     * @param out         Output packed weights
     * @param row_start   First row to pack (for TP slicing), default 0
     * @param row_end     One-past-last row, default -1 (all rows)
     * @param placement   Final destination first-touch policy for every encoding.
     * @return true on success
     */

    inline bool packWeightsCPUNativeQuantised(const TensorBase *weights,
                                         CPUNativeQuantisedPackedWeights &out,
                                         int row_start = 0, int row_end = -1,
                                         CPUWeightStoragePlacement placement = CPUWeightStoragePlacement::local())
    {
        // Validate
        int full_N = weights->shape()[0];
        int K = weights->shape()[1];
        if (row_end < 0)
            row_end = full_N;
        if (row_start < 0)
            row_start = 0;
        if (row_end > full_N)
            row_end = full_N;
        if (row_start >= row_end)
        {
            LOG_ERROR("[CPUNativeQuantised] Invalid row range: [" << row_start << ", " << row_end << ")");
            return false;
        }

        // Get IINT8Unpackable interface
        const IINT8Unpackable *unpackable = dynamic_cast<const IINT8Unpackable *>(weights);
        if (!unpackable)
        {
            LOG_ERROR("[CPUNativeQuantised] Tensor does not implement IINT8Unpackable");
            return false;
        }

        // Get native format info
        const NativeQuantisedFormatInfo *fmt = unpackable->vnniFormatInfo();
        if (!fmt)
        {
            LOG_ERROR("[CPUNativeQuantised] Tensor does not provide vnniFormatInfo");
            return false;
        }

        const int N = row_end - row_start;
        const int blocks_per_row = (K + 31) / 32;
        // The source already owns native bytes. Only the same append
        // primitive's workspace is transient; never make a native copy.
        const auto original_memory=placement.physicalMemoryAuthority();
        std::optional<PhysicalMemoryCPUWeightSourceLease> source_scope;
        if(original_memory)
            source_scope.emplace(original_memory->acquireCPUWeightSourceTransaction(placement.allocatorDevice()));
        const auto geometry=immediateNativePackScratchGeometry(*fmt,N,K,placement.storagePlacement());
        auto scratch=original_memory
            ? CPUStorageRoot<uint8_t>(geometry.native_bytes,PhysicalMemoryOwner::WeightLoadStaging,
                placement.allocatorDevice(),placement.storagePlacement(),CPUStorageInitialization::Uninitialized).allocate(original_memory)
            : placement.allocate<uint8_t>(geometry.native_bytes);
        initializeNativeWindowOutput(*fmt,N,K,placement,out);
        const int group_blocks=fmt->is_superblock?8:1;
        const size_t groups=(size_t(blocks_per_row)+group_blocks-1)/group_blocks;
        auto bytes=std::span<std::byte>(reinterpret_cast<std::byte*>(scratch.data()),scratch.size());
        if(geometry.maximum_units_per_window>=groups)
        {
            const int rows_per_window=static_cast<int>(std::min(size_t{64},geometry.maximum_units_per_window/groups));
            for(int row=0;row<N;row+=rows_per_window)
                appendNativeWeightGroups(*unpackable,*fmt,K,row_start+row,0,std::min(rows_per_window,N-row),
                    blocks_per_row,row,0,bytes,out);
        }
        else
        {
            // Even one full source row may exceed the cap. Visit native
            // K groups in order, without rebuilding a row-sized shadow.
            const int blocks_per_window=static_cast<int>(geometry.maximum_units_per_window)*group_blocks;
            for(int row=0;row<N;++row)for(int kb=0;kb<blocks_per_row;kb+=blocks_per_window)
                appendNativeWeightGroups(*unpackable,*fmt,K,row_start+row,kb,1,
                    std::min(blocks_per_window,blocks_per_row-kb),row,kb,bytes,out);
        }
        out.sealPublication();
        return true;
    }
}
