/**
 * @file Qwen4ExpBranchConcatStage.cpp
 * @brief Prepared-arena Dense0 layout stage with canonical capture identity.
 *
 * The native operation sees original token geometry. Constructor validation
 * rejects repeated descriptors and output aliases before graph preparation;
 * execution only resolves resident storage and submits the immutable operation.
 */
#include "Qwen4ExpBranchConcatStage.h"
#include "../../../../tensors/Tensors.h"
#include <array>
#include <limits>
#include <stdexcept>
namespace llaminar2 {
namespace {
/** @brief Reject malformed arena geometry before capture or weight preparation. */
void require(bool condition){if(!condition)throw std::invalid_argument("Invalid Qwen4ExpBranchConcat arena/row contract");}
/** @brief Mix one canonical identity scalar in fixed field order. */
void mix(std::uint64_t &hash,std::uint64_t value){hash^=value;hash*=1099511628211ull;}
}
Qwen4ExpBranchConcatStage::Qwen4ExpBranchConcatStage(Params params)
    :IComputeStage(params.device_id),params_(std::move(params)),operation_(qwen4exp::createBranchConcatOperation(device())) {
    const auto rows=params_.rows.captureIdentity();
    require(rows.repeats_per_row==1&&params_.hidden_dim>0&&params_.hidden_dim<=std::numeric_limits<int>::max()/8&&
        rows.physical_rows>0&&rows.physical_rows<=std::numeric_limits<int>::max()/4&&
        std::size_t(rows.physical_rows)<=std::numeric_limits<std::size_t>::max()/52/std::size_t(params_.hidden_dim));
    require(!rows.count||reinterpret_cast<std::uintptr_t>(rows.count)%4==0);
    const auto unit=std::size_t(rows.physical_rows)*std::size_t(params_.hidden_dim)*4;
    const std::array bindings{params_.hidden,params_.embedding,params_.output};
    const std::array bytes{unit*4,unit,unit*8};
    for(std::size_t i=0;i<bindings.size();++i){const auto &b=bindings[i];
        require(b.tensor&&b.buffer<BufferId::_COUNT&&b.tensor->native_type()==TensorType::FP32&&b.byteOffset%4==0&&
            b.byteOffset<=b.tensor->size_bytes()&&bytes[i]<=b.tensor->size_bytes()-b.byteOffset);
        for(std::size_t j=0;j<i;++j){const auto &p=bindings[j];
            require(b.buffer!=p.buffer||b.tensor==p.tensor);
            if(i==2&&(b.tensor==p.tensor||b.buffer==p.buffer))
                require(b.byteOffset+bytes[i]<=p.byteOffset||p.byteOffset+bytes[j]<=b.byteOffset);
        }
    }
    std::array<TensorBase*,3>roots{params_.hidden.tensor,params_.embedding.tensor,params_.output.tensor};storage_.emplace(device(),roots);
    // CPU-backed setup addresses provide an additional physical-alias proof.
    // GPU prepared addresses are checked by the same native validator at submit.
    if(!device().is_gpu())qwen4exp::validateBranchConcat({params_.rows,params_.hidden_dim,
        pointer(params_.hidden),pointer(params_.embedding),pointer(params_.output)});
}
float *Qwen4ExpBranchConcatStage::pointer(Qwen4ExpFloatBinding binding)const {
    auto *base=device().is_gpu()?binding.tensor->gpu_data_ptr():binding.tensor->raw_mutable_data();
    if(!base)throw std::logic_error("BranchConcat operand is not already prepared");
    return reinterpret_cast<float*>(static_cast<std::byte*>(base)+binding.byteOffset);
}
bool Qwen4ExpBranchConcatStage::prepareGraphLaunch(IDeviceContext *context,void *stream) {
    require(context&&context->deviceId()==device()&&(!device().is_gpu()||stream));storage_->seal();return true;
}
bool Qwen4ExpBranchConcatStage::execute(IDeviceContext *context) {
    require(context&&context->deviceId()==device());
    if(!device().is_gpu())storage_->seal();storage_->validate();
    void *stream=nullptr;
    if(device().is_gpu()){
        auto execution=gpuExecution();stream=gpuStream();
        execution.requirePreparedInput(params_.hidden.tensor);
        execution.requirePreparedInput(params_.embedding.tensor);
        execution.requirePreparedOutput(params_.output.tensor);
    }
    operation_({params_.rows,params_.hidden_dim,pointer(params_.hidden),pointer(params_.embedding),pointer(params_.output)},stream);
    return true;
}
ComputeStageType Qwen4ExpBranchConcatStage::type()const{return ComputeStageType::QWEN4EXP_BRANCH_CONCAT;}
std::size_t Qwen4ExpBranchConcatStage::estimatedMemoryBytes()const{
    return std::size_t(params_.rows.physicalRows())*std::size_t(params_.hidden_dim)*4*13;
}
bool Qwen4ExpBranchConcatStage::supportsBackend(ComputeBackendType backend)const {
    return (device().is_cpu()&&backend==ComputeBackendType::CPU)||
        (device().is_cuda()&&backend==ComputeBackendType::GPU_CUDA)||
        (device().is_rocm()&&backend==ComputeBackendType::GPU_ROCM);
}
StageBufferContract Qwen4ExpBranchConcatStage::bufferContract()const {
    StageBufferContract c;
    c.addInput(params_.hidden.buffer,"FP32");
    if(params_.embedding.buffer!=params_.hidden.buffer)c.addInput(params_.embedding.buffer,"FP32");
    if(params_.output.buffer==params_.hidden.buffer||params_.output.buffer==params_.embedding.buffer){
        c=StageBufferContract{};
        if(params_.hidden.buffer!=params_.output.buffer)c.addInput(params_.hidden.buffer,"FP32");
        if(params_.embedding.buffer!=params_.output.buffer&&params_.embedding.buffer!=params_.hidden.buffer)c.addInput(params_.embedding.buffer,"FP32");
        c.addPreallocatedInOut(params_.output.buffer,"FP32");
    }else {
        // This write-only arena result must participate in cold executor
        // allocation before prepareGraphLaunch seals its ordinary owner.
        c.addOutput(params_.output.buffer,"FP32");
    }
    return c;
}
StageDumpInfo Qwen4ExpBranchConcatStage::buildDumpInfoImpl()const {
    StageDumpInfo d;const auto m=std::size_t(params_.rows.physicalRows()),h=std::size_t(params_.hidden_dim);
    const auto raw=[](Qwen4ExpFloatBinding b)->const void*{auto*p=static_cast<const std::byte*>(b.tensor->raw_data());return p?p+b.byteOffset:nullptr;};
    d.inputs.push_back({"hidden",raw(params_.hidden),m,h*4,"FP32",4,m*h*16,params_.hidden.tensor,params_.hidden.byteOffset});
    d.inputs.push_back({"embedding",raw(params_.embedding),m,h,"FP32",4,m*h*4,params_.embedding.tensor,params_.embedding.byteOffset});
    d.outputs.push_back({"branch_concat",raw(params_.output),m,h*8,"FP32",4,m*h*32,params_.output.tensor,{},params_.output.byteOffset});
    const auto id=params_.rows.captureIdentity();
    d.addScalarInt("row_count_pointer",static_cast<std::int64_t>(reinterpret_cast<std::uintptr_t>(id.count)));
    d.addScalarInt("count_capacity",id.count_capacity);d.addScalarInt("first_physical_row",id.first_physical_row);
    d.addScalarInt("physical_rows",id.physical_rows);d.addScalarInt("repeats_per_row",id.repeats_per_row);
    d.addScalarInt("row_schema_version",id.schema_version);d.addScalarInt("hidden_dim",params_.hidden_dim);
    return d;
}
std::uint64_t Qwen4ExpBranchConcatStage::graphCaptureVariantSignature()const {
    storage_->validateIfSealed();
    std::uint64_t hash=1469598103934665603ull;mix(hash,1);mix(hash,static_cast<std::uint64_t>(type()));
    mix(hash,static_cast<std::uint64_t>(device().type));mix(hash,device().ordinal);
    const auto id=params_.rows.captureIdentity();mix(hash,reinterpret_cast<std::uintptr_t>(id.count));
    mix(hash,id.count_capacity);mix(hash,id.first_physical_row);mix(hash,id.physical_rows);mix(hash,id.repeats_per_row);mix(hash,id.schema_version);mix(hash,params_.hidden_dim);
    for(auto b:{params_.hidden,params_.embedding,params_.output}){
        mix(hash,reinterpret_cast<std::uintptr_t>(b.tensor));mix(hash,static_cast<std::uint64_t>(b.buffer));mix(hash,b.byteOffset);

    }
    mix(hash,reinterpret_cast<std::uintptr_t>(operation_));return hash;
}
}
