/**
 * @file CPUFactoryTestAdmission.h
 * @brief Exact finalized-schema admission for model-light factory fixtures.
 *
 * Factory tests intentionally supply only their global native source tensors.
 * This helper records that real inventory and delegates pricing to the same
 * graph-root declarations and canonical owner estimators as other fixtures.
 * It supplies no anonymous reserve and cannot replace a concrete model issuer.
 */
#pragma once
#include "CPUOrchestratorTestAdmission.h"
#include "execution/factory/TestableInferenceRunnerOverrides.h"
namespace llaminar2::test
{
/**
 * @brief Retain explicit fixture admission across factory TP/PP resolution.
 * @param model Mock metadata/source owner; queried only for authored globals.
 * @param batch Actual fixture request capacity.
 * @param context Actual fixture KV context capacity.
 * @return Test-only override invoked with the finalized participant graph.
 */
inline TestableInferenceRunnerOverrides cpuFactoryTestAdmission(
    const std::shared_ptr<IModelContext>& model,int batch,int context)
{
    std::vector<TensorSizeInfo> sources;
    for(const auto *name:{"token_embd.weight","output_norm.weight","output.weight"})
    {
        const auto tensor=model->getWeightForDevice(name,DeviceId::cpu());
        if(!tensor) continue;
        const auto &shape=tensor->shape();
        const auto format=tensor->native_type()==TensorType::FP32?"F32":
            tensor->native_type()==TensorType::FP16?"F16":tensorTypeName(tensor->native_type());
        sources.push_back({.name=name,.native_bytes=tensor->size_bytes(),.quant_type=format,
            .elements=tensor->numel(),.K=shape.size()<2?0:shape.back(),.layer_index=-1});
    }
    return {.test_graph_allocation_admission=
        [sources=std::move(sources),batch,context](const std::shared_ptr<IGraphBuilder>& graph)
        {
            const int rows=graph->config().max_activation_rows;
            if(rows<=0 || !graph->config().default_device.is_cpu())
                throw std::invalid_argument("CPU factory fixture requires its finalized CPU row envelope");
            return cpuInferenceFixtureDependencies(graph,{},batch,context,rows,sources).
                physical_memory_authority;
        }};
}
}
