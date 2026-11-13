/**
 * @file PooledImmutableLocalHotOperands.cpp
 * @brief Original producer declarations and cold physical seals for local native cache phases.
 *
 * The seal borrows actual allocator records once before capture. Reuse validates
 * addresses, native runtime and controller ownership without inspecting a live
 * hash, count, request, controller word or response payload.
 */
#include "PooledImmutableLocalHotOperands.h"
#include "transfer/TransferEngine.h"
#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>

namespace llaminar2::qwen4exp {
namespace {
/** @brief Closed schemas prevent an unused input from becoming an implicit authority. */
enum class Schema { Frame, Query, Adoption, Read, Result, RequestReset };
/** @return Checked immutable extent; never a live communication count. */
std::size_t multiply(std::size_t a, std::size_t b)
{
    if (a && b > std::numeric_limits<std::size_t>::max() / a)
        throw std::overflow_error("Local hot producer extent overflows");
    return a * b;
}
/** @brief Validate the actual root role and exact bounded span before physical preparation. */
void declaration(const PooledNativeTensorSpan& span, std::size_t bytes,
                 std::size_t alignment, TensorType type)
{
    if (!span.root || span.root->is_view() || span.root->transferStorageOwner() != span.root.get() ||
        span.buffer >= BufferId::_COUNT || span.root->native_type() != type || span.bytes != bytes ||
        !bytes || span.offset % alignment || span.offset > span.root->size_bytes() ||
        bytes > span.root->size_bytes() - span.offset)
        throw std::invalid_argument("Local hot phase requires its original exact type/role/extent");
}
/** @return Only the already-prepared original native address. */
std::byte* address(const PooledNativeTensorSpan& span)
{
    return static_cast<std::byte*>(span.root->gpu_data_ptr()) + span.offset;
}
/** @brief Mix named immutable fields; object padding and mutable native values are excluded. */
void mix(std::uint64_t& value, std::uint64_t word)
{
    value = (value ^ word) * 1099511628211ULL;
}
}

/** @brief Strong original producers outlive their allocator-record borrowers by declaration order. */
struct PooledImmutableLocalHotOperands::Data {
    const PooledImmutableLocalHotPreparedStorage* storage;
    local_hot_transport::ConsumerGeometry consumer;
    DeviceGenerationControlBinding controller;
    Schema schema;
    std::vector<PooledNativeTensorSpan> spans;
    std::optional<DeviceRowRange> rows;
    std::optional<Qwen4ExpPrefillNextWindowBinding> next_window;
    PooledImmutableLocalHotAccessIntent intent = PooledImmutableLocalHotAccessIntent::CurrentInvocation;
    std::optional<PooledNativeTransferSpan> own_request;
    void* sealed_request = nullptr;
    std::optional<Qwen4ExpPreparedStorage> prepared;
    std::vector<std::shared_ptr<const CapturedDeviceStorageLease>> captured;

    /** @brief Bind one original storage/controller pair; no source or live execution state is created. */
    Data(const PooledImmutableLocalHotPreparedStorage& owner,
         local_hot_transport::ConsumerGeometry originalConsumer,
         DeviceGenerationControlBinding originalController, Schema inputSchema)
        : storage(&owner), consumer(originalConsumer), controller(originalController), schema(inputSchema)
    {
        const auto& g = owner.geometry();
        if (!local_hot_transport::valid(consumer) || consumer.maximum_original != g.maximumOriginalIds() ||
            consumer.maximum_misses != g.maximumCompactMissRows() || consumer.source_rows != g.extent().rows() ||
            consumer.native_row_bytes != g.rowBytes() || consumer.native_format != std::uint32_t(g.format()) ||
            consumer.consumer_storage_identity != owner.consumerStorageIdentity())
            throw std::invalid_argument("Local hot producer changed its original consumer geometry/root identity");
        controller.requireFor(g.allocator().device);
    }
};

PooledImmutableLocalHotOperands PooledImmutableLocalHotOperands::query(
    const PooledImmutableLocalHotPreparedStorage& storage, local_hot_transport::ConsumerGeometry consumer,
    DeviceRowRange rows, PooledNativeTensorSpan ids, std::optional<PooledNativeTensorSpan> count,
    DeviceGenerationControlBinding controller, PooledImmutableLocalHotAccessIntent intent)
{
    const auto id = rows.captureIdentity();
    if (id.first_physical_row || id.repeats_per_row != 1 || id.physical_rows != id.count_capacity ||
        rows.capacity() > int(storage.geometry().maximumOriginalIds() / 16) ||
        bool(rows.countOwner()) != bool(count) ||
        (intent != PooledImmutableLocalHotAccessIntent::CurrentInvocation &&
         intent != PooledImmutableLocalHotAccessIntent::FutureReadAhead))
        throw std::invalid_argument("Local hot query needs its complete original token-row descriptor");
    auto data = std::make_shared<Data>(storage, consumer, controller, Schema::Query);
    data->rows = rows.repeatEachRow(16);
    data->intent = intent;
    declaration(ids, multiply(data->rows->physicalRows(), sizeof(std::uint64_t)), 8, TensorType::INT32);
    data->spans.push_back(std::move(ids));
    if (count) {
        declaration(*count, 4, 4, TensorType::INT32);
        data->spans.push_back(std::move(*count));
    }
    return PooledImmutableLocalHotOperands(std::move(data));
}

PooledImmutableLocalHotOperands PooledImmutableLocalHotOperands::nextWindowQuery(
    const PooledImmutableLocalHotPreparedStorage& storage,local_hot_transport::ConsumerGeometry consumer,
    Qwen4ExpPrefillNextWindowBinding window,PooledNativeTensorSpan ids,DeviceGenerationControlBinding controller)
{
    if(window.device()!=storage.geometry().allocator().device || !window.native().declared() ||
        window.native().next_bucket>int(storage.geometry().maximumOriginalIds()/16))
        throw std::invalid_argument("Local next-window query changed its original participant or bounded capacity");
    auto result=query(storage,consumer,DeviceRowRange::fullyActive(window.native().next_bucket),std::move(ids),{},
        controller,PooledImmutableLocalHotAccessIntent::FutureReadAhead);
    result.data_->next_window=std::move(window);return result;
}
bool PooledImmutableLocalHotOperands::hasNextWindow()const noexcept{return data_->next_window.has_value();}
std::uint64_t PooledImmutableLocalHotOperands::nextWindowIdentity()const noexcept
{return data_->next_window?data_->next_window->captureIdentity():0;}

PooledImmutableLocalHotOperands PooledImmutableLocalHotOperands::adoption(
    const PooledImmutableLocalHotPreparedStorage& storage, local_hot_transport::ConsumerGeometry consumer,
    DeviceRowRange rows, PooledNativeTensorSpan ids, std::optional<PooledNativeTensorSpan> count,
    PooledNativeTransferSpan admission, DeviceGenerationControlBinding controller)
{
    auto result = query(storage, consumer, rows, std::move(ids), std::move(count), controller,
                        PooledImmutableLocalHotAccessIntent::CurrentInvocation);
    admission.require(storage.geometry().allocator().device, alignof(local_hot_transport::SourceAdmissionHeader));
    if (admission.bytes() != sizeof(local_hot_transport::SourceAdmissionHeader))
        throw std::invalid_argument("Local adoption requires its actual original request publication");
    result.data_->schema = Schema::Adoption;
    result.data_->own_request.emplace(std::move(admission));
    return result;
}

PooledImmutableLocalHotOperands PooledImmutableLocalHotOperands::read(
    const PooledImmutableLocalHotPreparedStorage& storage, local_hot_transport::ConsumerGeometry consumer,
    PooledNativeTensorSpan output, DeviceGenerationControlBinding controller)
{
    const auto row = multiply(std::size_t(storage.geometry().rowElements()), sizeof(float));
    if (!output.bytes || output.bytes % multiply(row, 16) || output.bytes / row > storage.geometry().maximumOriginalIds())
        throw std::invalid_argument("Local hot output must name complete actual sixteen-head token rows");
    declaration(output, output.bytes, alignof(float), TensorType::FP32);
    auto data = std::make_shared<Data>(storage, consumer, controller, Schema::Read);
    data->spans.push_back(std::move(output));
    return PooledImmutableLocalHotOperands(std::move(data));
}

PooledImmutableLocalHotOperands PooledImmutableLocalHotOperands::result(
    const PooledImmutableLocalHotPreparedStorage& storage, local_hot_transport::ConsumerGeometry consumer,
    PooledNativeTransferSpan request, DeviceGenerationControlBinding controller)
{
    request.require(storage.geometry().allocator().device, alignof(local_hot_transport::RequestHeader));
    if (request.bytes() != sizeof(local_hot_transport::RequestHeader))
        throw std::invalid_argument("Local hot response acquisition needs its exact original request header");
    auto data = std::make_shared<Data>(storage, consumer, controller, Schema::Result);
    data->own_request.emplace(std::move(request));
    return PooledImmutableLocalHotOperands(std::move(data));
}

PooledImmutableLocalHotOperands PooledImmutableLocalHotOperands::requestReset(
    const PooledImmutableLocalHotPreparedStorage& storage, local_hot_transport::ConsumerGeometry consumer,
    PooledNativeTransferSpan admission, DeviceGenerationControlBinding controller)
{
    admission.require(storage.geometry().allocator().device, alignof(local_hot_transport::SourceAdmissionHeader));
    if (admission.bytes() != sizeof(local_hot_transport::SourceAdmissionHeader))
        throw std::invalid_argument("Local hot request reset requires its actual original admission publication");
    auto data = std::make_shared<Data>(storage, consumer, controller, Schema::RequestReset);
    data->own_request.emplace(std::move(admission));
    return PooledImmutableLocalHotOperands(std::move(data));
}

PooledImmutableLocalHotOperands PooledImmutableLocalHotOperands::frame(
    const PooledImmutableLocalHotPreparedStorage& storage, local_hot_transport::ConsumerGeometry consumer,
    DeviceGenerationControlBinding controller)
{
    return PooledImmutableLocalHotOperands(std::make_shared<Data>(storage, consumer, controller, Schema::Frame));
}

void PooledImmutableLocalHotOperands::seal()
{
    const auto device = data_->storage->geometry().allocator().device;
    data_->controller.requireFor(device);
    if (!data_->prepared) {
        std::vector<TensorBase*> roots{data_->controller.owner()};
        for (const auto& span : data_->spans)
            if (std::find(roots.begin(), roots.end(), span.root.get()) == roots.end()) roots.push_back(span.root.get());
        data_->prepared.emplace(device, roots);
    }
    data_->prepared->seal();
    if(data_->next_window){
        const auto& w=data_->next_window->native();
        const auto begin=reinterpret_cast<std::uintptr_t>(address(data_->spans.at(0)));
        const auto end=begin+data_->spans.at(0).bytes;
        const auto disjoint=[&](const std::int32_t* p,std::size_t count){
            const auto input=reinterpret_cast<std::uintptr_t>(p);
            return input<=std::numeric_limits<std::uintptr_t>::max()-count*4 &&
                (input+count*4<=begin || input>=end);
        };
        if(!disjoint(w.request_tokens,w.request_capacity)||!disjoint(w.request_positions,1)||
            !disjoint(w.request_total,1)||!disjoint(w.current_positions,1)||!disjoint(w.current_rows,1))
            throw std::logic_error("Local next-window native input aliases its hash output");
    }
    if (data_->own_request) {
        auto* current = data_->own_request->pointer(device, alignof(local_hot_transport::RequestHeader));
        if (data_->sealed_request && data_->sealed_request != current)
            throw std::logic_error("Local hot own-request native address changed after sealing");
        data_->sealed_request = current;
    }
    if (data_->captured.empty()) {
        if (isGraphCaptureActive()) throw std::logic_error("Local hot producers must be borrowed before capture");
        data_->captured.reserve(data_->spans.size());
        for (const auto& span : data_->spans)
            data_->captured.push_back(TransferEngine::instance().retainCapturedDeviceStorage(
                *span.root, device, span.root->size_bytes()));
    }
    if (data_->rows && data_->rows->countOwner() &&
        data_->rows->countOwner() != reinterpret_cast<const std::int32_t*>(address(data_->spans.at(1))))
        throw std::logic_error("Local hot count descriptor differs from its actual original producer span");
}

void PooledImmutableLocalHotOperands::require(PooledImmutableLocalHotAction action,
    const PooledImmutableLocalHotPreparedStorage& storage) const
{
    const Schema wanted = action == PooledImmutableLocalHotAction::Probe ? Schema::Query :
        action == PooledImmutableLocalHotAction::Adopt ? Schema::Adoption :
        action == PooledImmutableLocalHotAction::LocalRead ? Schema::Read :
        action == PooledImmutableLocalHotAction::AcceptResult ? Schema::Result :
        action == PooledImmutableLocalHotAction::ResetRequest ? Schema::RequestReset : Schema::Frame;
    if (data_->storage != &storage || wanted != data_->schema ||
        std::uint32_t(action) > std::uint32_t(PooledImmutableLocalHotAction::ResetRequest) || !data_->prepared)
        throw std::logic_error("Local hot launch changed its exact storage/action or lacks a cold seal");
    data_->controller.requireFor(storage.geometry().allocator().device);
    data_->prepared->validate();
    if (data_->own_request && (!data_->sealed_request ||
        data_->own_request->pointer(storage.geometry().allocator().device, alignof(local_hot_transport::RequestHeader)) != data_->sealed_request))
        throw std::logic_error("Local hot own-request producer changed after physical sealing");
    for (std::size_t i = 0; i < data_->spans.size(); ++i) {
        const auto& span = data_->spans[i];
        TransferEngine::instance().requireCapturedDeviceStorage(*data_->captured.at(i), *span.root,
            storage.geometry().allocator().device, span.root->size_bytes());
        if (!span.root->gpu_data_ptr() || reinterpret_cast<std::uintptr_t>(address(span)) %
            ((data_->schema == Schema::Query || data_->schema == Schema::Adoption) && i == 0 ? 8 : 4))
            throw std::logic_error("Local hot native producer lost its exact aligned prepared address");
    }
    if (data_->rows && data_->rows->countOwner() &&
        data_->rows->countOwner() != reinterpret_cast<const std::int32_t*>(address(data_->spans.at(1))))
        throw std::logic_error("Local hot original counted producer changed after capture");
}

const std::vector<PooledNativeTensorSpan>& PooledImmutableLocalHotOperands::spans() const noexcept
{return data_->spans;}
PooledImmutableLocalHotAccessIntent PooledImmutableLocalHotOperands::queryIntent() const
{
    if (!data_ || data_->schema != Schema::Query)
        throw std::logic_error("Query intent requires original Probe operands");
    return data_->intent;
}
std::uint64_t PooledImmutableLocalHotOperands::controllerIdentity() const noexcept
{return data_->controller.captureIdentity();}
TensorBase* PooledImmutableLocalHotOperands::controllerOwner() const noexcept
{return data_->controller.owner();}
std::uint64_t PooledImmutableLocalHotOperands::declarationIdentity() const
{
    std::uint64_t value = 1469598103934665603ULL;
    mix(value, local_hot::NativeVersion); mix(value, reinterpret_cast<std::uintptr_t>(data_->storage));
    mix(value, std::uint32_t(data_->schema)); mix(value, data_->controller.captureIdentity());
    mix(value, data_->consumer.coordinate); mix(value, data_->consumer.consumers);
    mix(value, data_->consumer.maximum_original); mix(value, data_->consumer.maximum_misses);
    mix(value, data_->consumer.source_rows); mix(value, data_->consumer.native_row_bytes);
    mix(value, data_->consumer.native_format); mix(value, data_->consumer.consumer_storage_identity);
    mix(value, std::uint32_t(data_->intent));
    if (data_->own_request) mix(value, data_->own_request->identity());
    mix(value,data_->next_window.has_value());
    if(data_->next_window)mix(value,data_->next_window->captureIdentity());
    for (const auto& span : data_->spans) {
        mix(value, reinterpret_cast<std::uintptr_t>(span.root.get())); mix(value, std::uint64_t(span.buffer));
        mix(value, span.offset); mix(value, span.bytes); mix(value, std::uint64_t(span.root->native_type()));
    }
    if (data_->rows) {
        const auto r = data_->rows->captureIdentity();
        mix(value, reinterpret_cast<std::uintptr_t>(r.count)); mix(value, r.count_capacity);
        mix(value, r.first_physical_row); mix(value, r.physical_rows); mix(value, r.repeats_per_row); mix(value, r.schema_version);
    }
    return value ? value : 1;
}
std::uint64_t PooledImmutableLocalHotOperands::captureIdentity() const
{
    if (!data_->prepared) throw std::logic_error("Local hot capture identity requires its actual prepared seal");
    auto value = declarationIdentity(); mix(value, data_->prepared->preparedIdentity());
    mix(value, reinterpret_cast<std::uintptr_t>(data_->sealed_request));
    return value ? value : 1;
}

local_hot::NativeFrame PooledImmutableLocalHotOperands::nativeFrame(const PooledImmutableLocalHotPreparedStorage& storage) const
{
    local_hot::NativeFrame frame;
    static_cast<local_hot::NativeRoots&>(frame) = storage.nativeRoots();
    frame.healthy = data_->controller.healthy(); frame.error = data_->controller.error(); frame.consumer = data_->consumer;
    return frame;
}
local_hot::NativeQuery PooledImmutableLocalHotOperands::nativeQuery(const PooledImmutableLocalHotPreparedStorage& storage) const
{
    return {nativeFrame(storage), *data_->rows, address(data_->spans.at(0)),
            pooled_native::OriginalIdWidth::UInt64, data_->intent};
}
local_hot::NativeAdoption PooledImmutableLocalHotOperands::nativeAdoption(const PooledImmutableLocalHotPreparedStorage& storage) const
{
    return {nativeQuery(storage), static_cast<const local_hot_transport::SourceAdmissionHeader*>(data_->sealed_request)};
}
local_hot::NativeNextWindowQuery PooledImmutableLocalHotOperands::nativeNextWindowQuery(
    const PooledImmutableLocalHotPreparedStorage& storage)const
{
    if(!data_->next_window || data_->schema!=Schema::Query)
        throw std::logic_error("Next-window native query requires its original Probe declaration");
    return {nativeQuery(storage),data_->next_window->native()};
}
local_hot::NativeRead PooledImmutableLocalHotOperands::nativeRead(const PooledImmutableLocalHotPreparedStorage& storage) const
{
    return {nativeFrame(storage), reinterpret_cast<float*>(address(data_->spans.at(0))),
        std::uint32_t(data_->spans.at(0).bytes / (sizeof(float) * storage.geometry().rowElements()))};
}
local_hot::NativeRequestReset PooledImmutableLocalHotOperands::nativeRequestReset(const PooledImmutableLocalHotPreparedStorage& storage) const
{
    return {nativeFrame(storage), static_cast<const local_hot_transport::SourceAdmissionHeader*>(data_->sealed_request)};
}
local_hot::NativeResult PooledImmutableLocalHotOperands::nativeResult(const PooledImmutableLocalHotPreparedStorage& storage) const
{
    return {nativeFrame(storage), static_cast<const local_hot_transport::RequestHeader*>(data_->sealed_request)};
}
} // namespace llaminar2::qwen4exp
