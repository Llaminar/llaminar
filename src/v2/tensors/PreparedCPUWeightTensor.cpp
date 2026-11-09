/**
 * @file PreparedCPUWeightTensor.cpp
 * @brief Preserve committed NativeQuantised engine lifetime without recreating raw tensor storage.
 *
 * The Store-only factory validates the engine's existing CPU/storage/dimension
 * contracts. Every operation requiring source bytes fails before mutation;
 * metadata retirement never releases or rebinds the engine's final weight root.
 */
#include "tensors/PreparedCPUWeightTensor.h"
#include <limits>
#include <stdexcept>
#include <utility>

namespace llaminar2 {
namespace {
/** @return The engine's actual independently retained CPU NativeQuantised storage state. */
bool committedCPUStorage(const std::shared_ptr<ITensorGemm>& engine)
{
    return engine && engine->supports_device(-1) && engine->hasWeights() &&
        engine->canReleaseSourceWeightTensor() && engine->packedWeightBytes()!=0 &&
        engine->exportCPUNativeQuantisedPackedWeights()!=nullptr;
}
}

std::shared_ptr<PreparedCPUWeightTensor> PreparedCPUWeightTensor::fromCommitted(
    std::shared_ptr<ITensorGemm> engine,std::vector<size_t> shape,TensorType execution_type,DeviceId allocator)
{
    if(!allocator.is_valid() || !allocator.is_cpu() || shape.size()!=2 || !shape[0] || !shape[1] ||
       shape[0]>size_t(std::numeric_limits<int>::max()) || shape[1]>size_t(std::numeric_limits<int>::max()) ||
       shape[0]>std::numeric_limits<size_t>::max()/shape[1] || !committedCPUStorage(engine) ||
       engine->get_n()!=static_cast<int>(shape[0]) || engine->get_k()!=static_cast<int>(shape[1]))
        throw std::invalid_argument("Prepared CPU metadata requires its actual committed CPU matrix engine/root");
    const auto* expected=native_quantised_formats::forQuantType(tensorTypeName(execution_type));
    if(expected) {
        NativeQuantisedSourceIdentity actual;
        if(!engine->exportNativeQuantisedSourceIdentity(actual) || !actual.present ||
           actual.codebook_id!=expected->codebook_id || actual.is_superblock!=expected->is_superblock)
            throw std::invalid_argument("Prepared CPU metadata changed its published execution-native codebook");
    } else if(execution_type!=TensorType::FP32 && execution_type!=TensorType::FP16 && execution_type!=TensorType::BF16) {
        throw std::invalid_argument("Prepared CPU metadata has an unsupported published execution-native type");
    }
    return std::shared_ptr<PreparedCPUWeightTensor>(new PreparedCPUWeightTensor(
        std::move(engine),std::move(shape),execution_type,allocator));
}

PreparedCPUWeightTensor::PreparedCPUWeightTensor(std::shared_ptr<ITensorGemm> engine,
    std::vector<size_t> shape,TensorType execution_type,DeviceId allocator)
    : engine_(std::move(engine)),shape_(std::move(shape)),allocator_(allocator),execution_type_(execution_type) {}
PreparedCPUWeightTensor::~PreparedCPUWeightTensor()=default;
bool PreparedCPUWeightTensor::hasPreparedDeviceState() const { return committedCPUStorage(engine_); }
[[noreturn]] void PreparedCPUWeightTensor::rejectRawOperation()
{
    throw std::logic_error("Prepared-only CPU matrix has no raw source storage; borrow its committed PreparedWeightStore engine");
}
void PreparedCPUWeightTensor::setLayout(TensorLayout layout) { if(layout!=TensorLayout::UNKNOWN)rejectRawOperation(); }
const float* PreparedCPUWeightTensor::data() const { rejectRawOperation(); }
float* PreparedCPUWeightTensor::mutable_data() { rejectRawOperation(); }
bool PreparedCPUWeightTensor::copyFrom(const TensorBase*) { rejectRawOperation(); }
std::unique_ptr<ITensorGemm> PreparedCPUWeightTensor::createGemm() { rejectRawOperation(); }
std::shared_ptr<TensorBase> PreparedCPUWeightTensor::create_view(const std::vector<size_t>&,size_t) { rejectRawOperation(); }
TensorBase* PreparedCPUWeightTensor::transferStorageOwner() { rejectRawOperation(); }
const TensorBase* PreparedCPUWeightTensor::transferStorageOwner() const { rejectRawOperation(); }
bool PreparedCPUWeightTensor::ensureOnDevice(DeviceId,void*) { rejectRawOperation(); }
bool PreparedCPUWeightTensor::ensureOnHost(void*) { rejectRawOperation(); }
void PreparedCPUWeightTensor::to_fp32(float*) const { rejectRawOperation(); }
void PreparedCPUWeightTensor::to_bf16(uint16_t*) const { rejectRawOperation(); }
void PreparedCPUWeightTensor::to_fp16(uint16_t*) const { rejectRawOperation(); }
void PreparedCPUWeightTensor::to_int8_blocked(int8_t*,float*,size_t) const { rejectRawOperation(); }
bool PreparedCPUWeightTensor::to_int8_perchannel(int8_t*,float*,float*) const { rejectRawOperation(); }
void PreparedCPUWeightTensor::to_fp32_row(size_t,float*) const { rejectRawOperation(); }
void PreparedCPUWeightTensor::to_fp32_span(size_t,size_t,float*) const { rejectRawOperation(); }
} // namespace llaminar2
