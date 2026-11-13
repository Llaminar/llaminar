/**
 * @file MoECPUExpertWorkspaceSource.h
 * @brief Original source-aware format geometry for CPU expert graph workspaces.
 *
 * Shared predictor tensors remain in their own immutable GGUF directory. The
 * loader resolves that original source without copying metadata into the main
 * directory, reading payloads or inferring formats from prepared fragments.
 */
#pragma once
#include "loaders/ModelLoader.h"
#include <array>
#include <stdexcept>
#include <string>
#include <vector>

namespace llaminar2 {
/**
 * @brief Resolve exact native Gate/Up/Down formats from the retained source loader.
 * @param loader Original model loader, including any admitted shared predictor.
 * @param layer Original global model/predictor source layer coordinate.
 * @param columns Complete original model width, before CPU shard slicing.
 * @param intermediate Complete original expert F width, before CPU shard slicing.
 * @param experts Complete original expert axis, independently of selected IDs.
 * @return Canonical native tensor types for the existing workspace quote.
 * @throws std::logic_error For missing or changed original source geometry.
 * @note This passive metadata projection allocates no weight/temporary storage
 * and confers no preparation, PMA or execution readiness.
 */
inline std::array<TensorType,3> originalCPUExpertWorkspaceSourceFormats(const ModelLoader &loader,
    int layer,int columns,int intermediate,int experts)
{
    if(layer<0 || columns<=0 || intermediate<=0 || experts<=0)
        throw std::logic_error("CPU expert workspace requires original positive source geometry");
    const std::array<const char*,3> roles{"ffn_gate_exps.weight","ffn_up_exps.weight","ffn_down_exps.weight"};
    std::array<TensorType,3> result;
    for(size_t role=0;role<roles.size();++role) {
        const auto name="blk."+std::to_string(layer)+"."+roles[role];
        const auto *tensor=loader.tensorInfo(name);
        const bool down=role==2;
        const std::vector<uint64_t> expected{uint64_t(down?intermediate:columns),
            uint64_t(down?columns:intermediate),uint64_t(experts)};
        if(!tensor || tensor->dimensions!=expected)
            throw std::logic_error("CPU expert workspace lost original native source geometry: "+name+
                " expected=["+std::to_string(expected[0])+","+std::to_string(expected[1])+","+
                std::to_string(expected[2])+"] source="+(tensor?"shape_mismatch":"missing"));
        result[role]=ggufToTensorType(tensor->type);
    }
    return result;
}
}
