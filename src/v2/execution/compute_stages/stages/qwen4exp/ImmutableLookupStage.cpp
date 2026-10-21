/**
 * @file ImmutableLookupStage.cpp
 * @brief Execute exact native row I/O and decoding through explicit CPU/GPU ownership.
 *
 * CPU calls use one live span and GPU calls consume the same device-owned row
 * count used by the hash producer. Resident lookup reads the original native
 * table directly after acquiring its setup event; it needs no host channel or
 * gathered-row scratch. Hosted responses are acknowledged only after their
 * native bytes reach private scratch, preventing reuse during GPU decoding.
 */
#include "ImmutableLookupStage.h"
#include "../../../../kernels/qwen4exp/ImmutableLookupDecode.h"
#include "../../../../tensors/Tensors.h"
#include "../../../../transfer/TransferEngine.h"
#include "../../../local_execution/graph/RetainedStoragePerfStats.h"
#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>
#include <utility>

namespace llaminar2
{
namespace
{
/** @brief Reject incomplete setup before capture begins. */
void require(bool condition,const char *message)
{
    if(!condition) throw std::invalid_argument(message);
}
/** @brief Prove one typed byte range without materializing it. */
template<TensorType Type>
void check(Qwen4ExpArenaBinding<Type> binding,std::size_t bytes,std::size_t alignment=4)
{
    require(binding.tensor && binding.tensor->native_type()==Type && binding.buffer<BufferId::_COUNT &&
        binding.byteOffset%alignment==0 && binding.byteOffset<=binding.tensor->size_bytes() &&
        bytes<=binding.tensor->size_bytes()-binding.byteOffset,
        "Immutable lookup stage requires aligned native arena views with sufficient admitted capacity");
}
}
ImmutableLookupStage::ImmutableLookupStage(Params params)
    :IComputeStage(params.device_id),params_(std::move(params))
{
    require(device().is_valid() && params_.rows.capacity>0,"Immutable lookup stage requires an explicit device and positive row capacity");
    qwen4exp::lookup_detail::validateGeometry(params_.format,params_.rowElements,params_.rowBytes);
    const auto count=static_cast<std::size_t>(params_.rows.capacity)*qwen4exp::kNgramHeads;
    require(count<=std::numeric_limits<std::size_t>::max()/params_.rowBytes &&
        count<=std::numeric_limits<std::size_t>::max()/params_.rowElements/sizeof(float),
        "Immutable lookup stage row geometry overflows physical bytes");
    check(params_.ids,count*sizeof(std::int64_t),alignof(std::int64_t));
    check(params_.output,count*params_.rowElements*sizeof(float));
    require(params_.ids.tensor!=params_.output.tensor && params_.ids.buffer!=params_.output.buffer,
        "Immutable lookup IDs and decoded output must have independent storage");
    std::vector<TensorBase *> roots{params_.ids.tensor, params_.output.tensor};
    if (const auto scratch = nativeRows())
    {
        check(*scratch, count*params_.rowBytes);
        require(scratch->tensor!=params_.ids.tensor && scratch->buffer!=params_.ids.buffer &&
            scratch->tensor!=params_.output.tensor && scratch->buffer!=params_.output.buffer,
            "Hosted immutable lookup native scratch must have independent storage");
        roots.push_back(scratch->tensor);
    }
    if(params_.rows.liveRows)
    {
        const auto &value=params_.rows.liveRows->value;
        if(auto *arena=std::get_if<Qwen4ExpIntBinding>(&value)) {
            check(*arena,sizeof(std::int32_t)); roots.push_back(arena->tensor);
        }
        else {
            const auto &native=std::get<Qwen4ExpNativeIntInput>(value);
            require(native.owner==device() && native.data && native.capacity>=1 &&
                reinterpret_cast<std::uintptr_t>(native.data)%alignof(std::int32_t)==0,
                "Immutable lookup live rows require the exact execution owner");
        }
    }
    if(auto *cpu=std::get_if<CPUStorage>(&params_.storage))
    {
        require(device().is_cpu() && cpu->table,"Direct immutable table I/O requires CPU execution and the model-owned table");
        if(const auto bound=cpu->table->maximumLiveRows();bound && count>*bound)
            throw std::invalid_argument("CPU lookup stage exceeds the provider's admitted original ID envelope");
        const auto &source=cpu->table->source();
        require(ggufToTensorType(source.format)==params_.format && source.row_elements==static_cast<std::size_t>(params_.rowElements) &&
            source.rowBytes()==params_.rowBytes,"Immutable lookup decode geometry disagrees with its exact native table");
    }
    else if (auto *binding=std::get_if<GPUStorage>(&params_.storage))
    {
        const auto &gpu=*binding;
        require(device().is_gpu() && gpu.boundary && gpu.boundary->device()==device(),
            "GPU immutable lookup requires its exact retained participant storage boundary");
        require(gpu.boundary->channel().maximumIds>=count && gpu.boundary->channel().rowBytes==params_.rowBytes,
            "GPU immutable lookup channel is smaller than its admitted transaction or uses another native row width");
        check(gpu.state,sizeof(qwen4exp::ImmutableLookupDeviceState),alignof(qwen4exp::ImmutableLookupDeviceState));
        require(gpu.state.tensor!=params_.ids.tensor && gpu.state.tensor!=gpu.nativeRows.tensor &&
            gpu.state.tensor!=params_.output.tensor,"GPU lookup protocol state must retain independent ownership");
        roots.push_back(gpu.state.tensor);
    }
    else if (const auto *binding = std::get_if<ResidentGPUStorage>(&params_.storage))
    {
        const auto &resident = *binding;
        require(device().is_gpu() && resident.table && resident.table->device() == device(),
                "Resident lookup requires the exact admitted GPU table owner");
        const auto view = resident.table->view();
        require(view.format() == params_.format && view.rowElements() == params_.rowElements &&
                    view.rowBytes() == params_.rowBytes,
                "Resident lookup geometry differs from its original model source");
        resident.control.requireFor(device());
#ifdef HAVE_CUDA
        if(device().is_cuda()) resident_kernels_=qwen4exp::createCUDAImmutableResidentLookupKernels(view);
#endif
#ifdef HAVE_ROCM
        if(device().is_rocm()) resident_kernels_=qwen4exp::createROCmImmutableResidentLookupKernels(view);
#endif
        require(bool(resident_kernels_) && resident_kernels_->table()==view,
            "Resident lookup backend did not bind the exact immutable source");
    }
    else
    {
        const auto &adaptive = std::get<AdaptiveGPUStorage>(params_.storage);
        require(device().is_gpu() && adaptive.boundary && adaptive.boundary->device() == device() &&
                    adaptive.boundary->isAdaptive(),
                "Adaptive lookup requires its exact protocol2 source boundary");
        const auto &plan = adaptive.boundary->adaptiveKernels().plan().parameters();
        require(plan.format == params_.format && plan.row_elements == params_.rowElements &&
                    plan.row_bytes == params_.rowBytes && count <= plan.maximum_ids,
                "Adaptive lookup differs from its source-native geometry or "
                "admitted frame capacity");
        check(adaptive.frame, plan.frame.bytes);
        check(adaptive.state, sizeof(qwen4exp::ImmutableLookupDeviceState),
              alignof(qwen4exp::ImmutableLookupDeviceState));
        require(adaptive.frame.byteOffset == 0 && adaptive.state.byteOffset == 0,
                "Adaptive lookup must bind complete original frame and protocol "
                "roots");
        if (adaptive.cache)
        {
            check(*adaptive.cache, plan.cache.bytes);
            require(adaptive.cache->byteOffset == 0, "Adaptive cache requires its complete original root");
        }
        adaptive.boundary->requireAdaptiveRoots(adaptive.cache ? adaptive.cache->tensor : nullptr,
                                                adaptive.frame.tensor, adaptive.state.tensor);
        adaptive.control.requireFor(device());
        if (params_.rows.liveRows)
            if (const auto *native = std::get_if<Qwen4ExpNativeIntInput>(&params_.rows.liveRows->value))
                require(native->capacity == 1, "Adaptive queries require their exact "
                                               "single original count word");
        const auto original = params_.rows.liveRows
                                  ? DeviceRowRange::deviceCounted(params_.rows.capacity, liveCount())
                                  : DeviceRowRange::fullyActive(params_.rows.capacity);
        require(adaptive.originalRows.captureIdentity() == original.captureIdentity(),
                "Adaptive lookup must retain the complete original query descriptor");
        require(static_cast<bool>(adaptive.count) == (adaptive.originalRows.countOwner() != nullptr),
                "Adaptive counted queries must declare their original arena "
                "count owner");
        if (adaptive.count)
        {
            check(*adaptive.count, sizeof(std::int32_t));
            require(pointer<const std::int32_t>(*adaptive.count) == adaptive.originalRows.countOwner(),
                    "Adaptive query count differs from its exact declared original "
                    "arena subview");
            require(adaptive.count->tensor != params_.ids.tensor &&
                        adaptive.count->tensor != params_.output.tensor &&
                        adaptive.count->buffer != params_.ids.buffer &&
                        adaptive.count->buffer != params_.output.buffer &&
                        adaptive.count->tensor != adaptive.frame.tensor &&
                        adaptive.count->tensor != adaptive.state.tensor &&
                        (!adaptive.cache || adaptive.count->tensor != adaptive.cache->tensor),
                    "Adaptive original count cannot alias mutable "
                    "cache/frame/protocol storage");
        }
        // Each mutable physical owner has one canonical arena role. Merely
        // retaining pointer lifetimes would hide ordering from the executor.
        const std::array<TensorBase *, 4> mutableRoots{adaptive.cache ? adaptive.cache->tensor : nullptr,
                                                       adaptive.frame.tensor, adaptive.state.tensor,
                                                       adaptive.control.owner()};
        const std::array<BufferId, 4> mutableRoles{adaptive.cache ? adaptive.cache->buffer : BufferId::_COUNT,
                                                   adaptive.frame.buffer, adaptive.state.buffer,
                                                   BufferId::MTP_GENERATION_CONTROL};
        for (std::size_t i = 0; i < mutableRoots.size(); ++i)
        {
            if (!mutableRoots[i])
                continue;
            require(mutableRoots[i] != params_.ids.tensor && mutableRoots[i] != params_.output.tensor &&
                        mutableRoles[i] != params_.ids.buffer && mutableRoles[i] != params_.output.buffer,
                    "Adaptive mutable roots must not alias original IDs or decoded "
                    "output");
            for (std::size_t j = 0; j < i; ++j)
                if (mutableRoots[j])
                    require(mutableRoots[j] != mutableRoots[i] && mutableRoles[j] != mutableRoles[i],
                            "Adaptive cache, frame, protocol and controller must have "
                            "distinct original roles");
            if (adaptive.count)
                require((adaptive.count->tensor == mutableRoots[i]) ==
                            (adaptive.count->buffer == mutableRoles[i]),
                        "Adaptive count role must identify its same original tensor "
                        "owner");
            roots.push_back(mutableRoots[i]);
        }
        if (adaptive.count && std::find(roots.begin(), roots.end(), adaptive.count->tensor) == roots.end())
            roots.push_back(adaptive.count->tensor);
    }
    prepared_.emplace(device(),roots);
}

std::optional<Qwen4ExpIntBinding> ImmutableLookupStage::nativeRows() const
{
    if (const auto *cpu=std::get_if<CPUStorage>(&params_.storage)) return cpu->nativeRows;
    if (const auto *gpu=std::get_if<GPUStorage>(&params_.storage)) return gpu->nativeRows;
    return {};
}

bool ImmutableLookupStage::prepareGraphLaunch(IDeviceContext *context,void *stream)
{
    require(context && context->deviceId()==device() && (!device().is_gpu() || stream),
        "Immutable lookup preparation requires its exact device and explicit GPU stream");
    if (stream) setGPUStream(stream);
    prepared_->seal();
    if (auto *resident=std::get_if<ResidentGPUStorage>(&params_.storage))
    {
        resident->control.requireFor(device());
        require(resident_kernels_->table()==resident->table->view(),"Resident lookup table changed before capture");
        auto &transfers=TransferEngine::instance();
        if (!resident_read_) resident_read_.emplace(resident->table->acquire(stream,transfers));
        transfers.requireImmutableNativeDeviceInput(*resident_read_,device(),stream);
    }
    if (const auto *adaptive = std::get_if<AdaptiveGPUStorage>(&params_.storage))
    {
        adaptive->control.requireFor(device());
        adaptive->boundary->bindAdaptiveGenerationControl(adaptive->control);
        // The executor owns the typed frontier AFTER launch preparation.
        // Setup records stable original addresses before admission produces
        // protocol bytes; ordinary capture/replay still requires ready bytes.
        // Acquiring these roots here would bypass that exact authority.
    }
    return true;
}
template<typename T,TensorType Type>
T *ImmutableLookupStage::pointer(Qwen4ExpArenaBinding<Type> binding) const
{
    void *base=device().is_gpu()?binding.tensor->gpu_data_ptr():binding.tensor->raw_mutable_data();
    require(base!=nullptr,"Immutable lookup operand storage was not prepared");
    auto *result=static_cast<std::byte *>(base)+binding.byteOffset;
    require(reinterpret_cast<std::uintptr_t>(result)%alignof(T)==0,"Immutable lookup native pointer is misaligned");
    return reinterpret_cast<T *>(result);
}
const std::int32_t *ImmutableLookupStage::liveCount() const
{
    if(!params_.rows.liveRows) return nullptr;
    const auto &value=params_.rows.liveRows->value;
    if(auto *arena=std::get_if<Qwen4ExpIntBinding>(&value)) return pointer<const std::int32_t>(*arena);
    return std::get<Qwen4ExpNativeIntInput>(value).data;
}
bool ImmutableLookupStage::execute(IDeviceContext *context)
{
    require(context && context->deviceId()==device(),"Immutable lookup requires the exact participant execution context");
    if (device().is_cpu()) prepared_->seal();
    prepared_->validate();
    if(auto *cpu=std::get_if<CPUStorage>(&params_.storage))
    {
        const int rows=liveCount()?*liveCount():params_.rows.capacity;
        require(rows>=0 && rows<=params_.rows.capacity,"Immutable lookup live rows exceed admitted capacity");
        const auto count=static_cast<std::size_t>(rows)*qwen4exp::kNgramHeads;
        auto *native=pointer<std::byte>(cpu->nativeRows);
        std::optional<HostImmutableRowGatherObservation> observation;
        try
        {
            observation=cpu->table->gather({pointer<const std::int64_t>(params_.ids),count},{native,count*params_.rowBytes});
            qwen4exp::decodeCPUImmutableLookupRows(params_.format,params_.rowElements,params_.rowBytes,
                native,count,pointer<float>(params_.output));
        }
        catch(...)
        {
            recordCPUImmutableLookup(cpu->table->mode(),device(),0,0,RetainedStorageCPUOutcome::Failed);
            throw;
        }
        recordCPUImmutableLookup(cpu->table->mode(),device(),count,count*params_.rowBytes,
            RetainedStorageCPUOutcome::Completed,observation);
        return true;
    }
    const auto execution=gpuExecution();
    execution.requirePreparedInput(params_.ids.tensor);
    execution.requirePreparedOutput(params_.output.tensor);
    if(params_.rows.liveRows)
        if(auto *arena=std::get_if<Qwen4ExpIntBinding>(&params_.rows.liveRows->value))
            execution.requirePreparedInput(arena->tensor);
    if (auto *resident=std::get_if<ResidentGPUStorage>(&params_.storage))
    {
        resident->control.requireFor(device());
        require(resident_read_.has_value(),"Resident lookup has no capture-stream upload acquire");
        TransferEngine::instance().requireImmutableNativeDeviceInput(*resident_read_,device(),gpuStream());
        const auto count=static_cast<std::size_t>(params_.rows.capacity)*qwen4exp::kNgramHeads;
        resident_kernels_->gather({{params_.rows.capacity,liveCount()},pointer<const std::int64_t>(params_.ids),count,
            pointer<float>(params_.output),count*params_.rowElements,resident->control.healthy(),resident->control.error()},gpuStream());
        return true;
    }
    if (const auto *adaptive = std::get_if<AdaptiveGPUStorage>(&params_.storage))
    {
        adaptive->control.requireFor(device());
        for (auto *root : {adaptive->cache ? adaptive->cache->tensor : nullptr, adaptive->frame.tensor,
                           adaptive->state.tensor, adaptive->control.owner()})
            if (root)
            {
                execution.requirePreparedInput(root);
                execution.requirePreparedOutput(root);
            }
        if (adaptive->count)
            execution.requirePreparedInput(adaptive->count->tensor);
        const auto count = static_cast<std::size_t>(params_.rows.capacity) * qwen4exp::kNgramHeads;
        adaptive->boundary->adaptiveKernels().gather(
            *adaptive->boundary->adaptiveStorage(),
            {adaptive->originalRows, pointer<const std::int64_t>(params_.ids), count,
             pointer<float>(params_.output), count * params_.rowElements},
            adaptive->control, gpuStream());
        for (auto *root : {adaptive->cache ? adaptive->cache->tensor : nullptr, adaptive->frame.tensor,
                           adaptive->state.tensor, adaptive->control.owner()})
            if (root)
                execution.publish(root);
        execution.publish(params_.output.tensor);
        return true;
    }
    const auto &gpu=std::get<GPUStorage>(params_.storage);
    execution.requirePreparedInput(gpu.state.tensor);
    execution.requirePreparedOutput(gpu.nativeRows.tensor);
    require(pointer<qwen4exp::ImmutableLookupDeviceState>(gpu.state)==gpu.boundary->state(),
        "Immutable lookup protocol arena disagrees with the retained boundary");
    const auto control=gpu.boundary->generationControl();
    gpu.boundary->kernels().publish({{params_.rows.capacity,liveCount()},pointer<const std::int64_t>(params_.ids),
        gpu.boundary->generation(),gpu.boundary->state(),gpu.boundary->channel()},gpuStream());
    gpu.boundary->kernels().consume({gpu.boundary->state(),gpu.boundary->channel(),pointer<std::byte>(gpu.nativeRows),
        pointer<float>(params_.output),control.healthy,control.error},gpuStream());
    return true;
}
std::size_t ImmutableLookupStage::estimatedFlops() const
{
    return static_cast<std::size_t>(params_.rows.capacity)*qwen4exp::kNgramHeads*params_.rowElements*2;
}
std::size_t ImmutableLookupStage::estimatedMemoryBytes() const
{
    return static_cast<std::size_t>(params_.rows.capacity)*qwen4exp::kNgramHeads*
        (sizeof(std::int64_t)+params_.rowBytes+params_.rowElements*sizeof(float));
}
bool ImmutableLookupStage::supportsBackend(ComputeBackendType backend) const
{
    return (device().is_cpu() && backend==ComputeBackendType::CPU) ||
        (device().is_cuda() && backend==ComputeBackendType::GPU_CUDA) ||
        (device().is_rocm() && backend==ComputeBackendType::GPU_ROCM);
}
StageBufferContract ImmutableLookupStage::bufferContract() const
{
    auto contract=StageBufferContract::build().addInput(params_.ids.buffer,"INT32")
        .addOutput(params_.output.buffer,"FP32");
    if (const auto scratch=nativeRows()) contract.addOutput(scratch->buffer,"INT32");
    if(params_.rows.liveRows)
        if(auto *arena=std::get_if<Qwen4ExpIntBinding>(&params_.rows.liveRows->value))
            contract.addInput(arena->buffer,"INT32");
    if(auto *gpu=std::get_if<GPUStorage>(&params_.storage)) contract.addInOut(gpu->state.buffer,"INT32");
    if (const auto *adaptive = std::get_if<AdaptiveGPUStorage>(&params_.storage))
    {
        if (adaptive->count && adaptive->count->buffer != BufferId::MTP_GENERATION_CONTROL)
            contract.addInput(adaptive->count->buffer, "INT32");
        if (adaptive->cache)
            contract.addPreallocatedInOut(adaptive->cache->buffer, "INT8");
        contract.addPreallocatedInOut(adaptive->frame.buffer, "INT8")
            .addPreallocatedInOut(adaptive->state.buffer, "INT32")
            .addPreallocatedInOut(BufferId::MTP_GENERATION_CONTROL, "INT32");
    }
    return contract;
}
StageDumpInfo ImmutableLookupStage::buildDumpInfoImpl() const
{
    StageDumpInfo dump;
    const auto *base=static_cast<const std::byte *>(params_.output.tensor->raw_data());
    const auto cols=static_cast<std::size_t>(qwen4exp::kNgramHeads)*params_.rowElements;
    dump.outputs.push_back({"gathered_rows",base?base+params_.output.byteOffset:nullptr,
        static_cast<std::size_t>(params_.rows.capacity),cols,"FP32",sizeof(float),
        params_.rows.capacity*cols*sizeof(float),params_.output.tensor,{},params_.output.byteOffset});
    dump.addScalarInt("native_row_bytes",params_.rowBytes);
    return dump;
}

std::uint64_t ImmutableLookupStage::graphCaptureVariantSignature() const
{
    prepared_->validateIfSealed();
    std::uint64_t value=1469598103934665603ULL;
    const auto mix=[&](std::uint64_t word) { value^=word; value*=1099511628211ULL; };
    const auto binding=[&](const auto &view) {
        mix(reinterpret_cast<std::uintptr_t>(view.tensor)); mix(static_cast<std::uint64_t>(view.buffer));
        mix(view.byteOffset);
    };
    mix(params_.storage.index()); mix(static_cast<std::uint64_t>(device().type)); mix(device().ordinal);
    mix(static_cast<std::uint64_t>(params_.format)); mix(params_.rowElements); mix(params_.rowBytes); mix(params_.rows.capacity);
    binding(params_.ids); binding(params_.output);
    if (const auto scratch=nativeRows()) binding(*scratch);
    if (params_.rows.liveRows)
    {
        if (auto *arena=std::get_if<Qwen4ExpIntBinding>(&params_.rows.liveRows->value)) binding(*arena);
        else {
            const auto &native=std::get<Qwen4ExpNativeIntInput>(params_.rows.liveRows->value);
            mix(reinterpret_cast<std::uintptr_t>(native.data)); mix(native.capacity);
        }
    }
    if (auto *cpu=std::get_if<CPUStorage>(&params_.storage)) {
        mix(reinterpret_cast<std::uintptr_t>(cpu->table.get()));mix(static_cast<std::uint64_t>(cpu->table->mode()));
    }
    else if (auto *gpu=std::get_if<GPUStorage>(&params_.storage)) {
        mix(gpu->boundary->captureIdentity()); binding(gpu->state);
    }
    else if (const auto *binding = std::get_if<ResidentGPUStorage>(&params_.storage))
    {
        const auto &resident = *binding;
        resident.control.requireFor(device());
        const auto view = resident.table->view();
        require(view == resident_kernels_->table(), "Resident lookup source identity changed after binding");
        mix(reinterpret_cast<std::uintptr_t>(resident.table.get()));
        mix(reinterpret_cast<std::uintptr_t>(view.nativeData()));
        mix(view.rowCount());
        mix(view.nativeBytes());
        mix(view.storageGeneration());
        mix(resident.control.captureIdentity());
    }
    if (const auto *adaptive = std::get_if<AdaptiveGPUStorage>(&params_.storage))
    {
        adaptive->control.requireFor(device());
        adaptive->boundary->requireAdaptiveRoots(adaptive->cache ? adaptive->cache->tensor : nullptr,
                                                 adaptive->frame.tensor, adaptive->state.tensor);
        mix(adaptive->boundary->captureIdentity());
        mix(adaptive->control.captureIdentity());
        const auto rows = adaptive->originalRows.captureIdentity();
        mix(reinterpret_cast<std::uintptr_t>(rows.count));
        mix(rows.count_capacity);
        mix(rows.first_physical_row);
        mix(rows.physical_rows);
        mix(rows.repeats_per_row);
        mix(rows.schema_version);
        if (adaptive->cache)
            binding(*adaptive->cache);
        binding(adaptive->frame);
        binding(adaptive->state);
        if (adaptive->count)
            binding(*adaptive->count);
    }
    return value ? value : 1;
}
} // namespace llaminar2
