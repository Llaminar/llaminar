/**
 * @file CPUTensorStorageDeclaration.cpp
 * @brief One dtype-to-root mapping for CPU planning and native materialization.
 *
 * Every native byte extent comes from the production tensor extent authority.
 * Factory NUMA binding precedes the common root's zero initialization, and all
 * tensors adopt that exact backing without per-caller lease/deleter wrappers.
 */
#include "CPUTensorStorageDeclaration.h"
#include "tensors/TensorClasses.h"
#include "tensors/TensorFactory.h"
#include "tensors/NativeTensorExtent.h"
#include <string>
#include <limits>
#include <type_traits>
namespace llaminar2
{
CPUTensorStorageDeclaration::CPUTensorStorageDeclaration(size_t rows, size_t columns,
    BufferTensorType dtype, PhysicalMemoryOwner owner, DeviceId allocator, CPUStoragePlacement placement)
    : shape_{rows,columns}, dtype_(dtype), root_(declareRoot(rows,columns,dtype,owner,allocator,placement)) {}
CPUTensorStorageDeclaration CPUTensorStorageDeclaration::fromDescriptor(
    const BufferDescriptor &descriptor, PhysicalMemoryOwner owner)
{
    if(descriptor.shape.empty())
        throw std::invalid_argument("CPU tensor storage: graph buffer '"+descriptor.name+"' must declare at least one shape dimension");
    const size_t rows=descriptor.shape.front();
    if(rows==0)
        throw std::invalid_argument("CPU tensor storage: graph buffer '"+descriptor.name+"' has a zero-sized row dimension");
    // Rank one is [rows,1]; flatten every trailing axis exactly once. This
    // geometry is shared by schema BOM projection and arena registration.
    size_t columns=1;
    for(size_t axis=1;axis<descriptor.shape.size();++axis)
    {
        const size_t dimension=descriptor.shape[axis];
        if(dimension==0)
            throw std::invalid_argument("CPU tensor storage: graph buffer '"+descriptor.name+"' has a zero-sized dimension at axis "+std::to_string(axis));
        if(columns>std::numeric_limits<size_t>::max()/dimension)
            throw std::overflow_error("CPU tensor storage: trailing shape product overflows for graph buffer '"+descriptor.name+"'");
        columns*=dimension;
    }
    if(rows>std::numeric_limits<size_t>::max()/columns)
        throw std::overflow_error("CPU tensor storage: element count overflows for graph buffer '"+descriptor.name+"'");
    return CPUTensorStorageDeclaration(rows,columns,descriptor.tensor_type,owner,descriptor.device);
}
CPUTensorStorageDeclaration::Root CPUTensorStorageDeclaration::declareRoot(size_t rows, size_t columns,
    BufferTensorType dtype, PhysicalMemoryOwner owner, DeviceId allocator, CPUStoragePlacement placement)
{
    switch(dtype)
    {
    case BufferTensorType::FP32:
        return CPUStorageRoot<float>(nativeMatrixExtent(rows,columns,1,1),owner,allocator,placement);
    case BufferTensorType::FP16: case BufferTensorType::BF16:
        return CPUStorageRoot<uint16_t>(nativeMatrixExtent(rows,columns,1,1),owner,allocator,placement);
    case BufferTensorType::INT32:
        return CPUStorageRoot<int32_t>(nativeMatrixExtent(rows,columns,1,1),owner,allocator,placement);
    case BufferTensorType::Q8_1:
        return CPUStorageRoot<uint8_t>(nativeMatrixExtent<Q8_1Block>(rows,columns),owner,allocator,placement);
    case BufferTensorType::Q16_1:
        return CPUStorageRoot<uint8_t>(nativeMatrixExtent<Q16_1Block>(rows,columns),owner,allocator,placement);
    default: throw std::invalid_argument("CPU tensor storage declaration requires a supported native arena dtype");
    }
}
size_t CPUTensorStorageDeclaration::physicalBytes() const
{
    return std::visit([](const auto &root){return root.physicalBytes();},root_);
}
std::unique_ptr<TensorBase> CPUTensorStorageDeclaration::materialize(
    std::shared_ptr<PhysicalMemoryAuthority> authority, TensorFactory *factory) const
{
    // Binding is the factory's existing admitted rank CPU policy. It must
    // precede the first writes from CPUStorageRoot's Zero initialization.
    if(factory) factory->ensureNumaBinding();
    const std::vector<size_t> shape{rows(),columns()};
    return std::visit([&](const auto &root)->std::unique_ptr<TensorBase>{
        using R=std::remove_cvref_t<decltype(root)>;
        auto storage=root.allocate(authority);
        if constexpr(std::is_same_v<R,CPUStorageRoot<float>>)
            return std::make_unique<FP32Tensor>(shape,std::move(storage),root.allocator());
        else if constexpr(std::is_same_v<R,CPUStorageRoot<uint16_t>>)
        {
            if(dtype_==BufferTensorType::FP16)
                return std::make_unique<FP16Tensor>(shape,std::move(storage),root.allocator());
            return std::make_unique<BF16Tensor>(shape,std::move(storage),root.allocator());
        }
        else if constexpr(std::is_same_v<R,CPUStorageRoot<int32_t>>)
            return std::make_unique<INT32Tensor>(shape,std::move(storage),root.allocator());
        else
        {
            if(dtype_==BufferTensorType::Q8_1)
                return std::make_unique<Q8_1Tensor>(shape,std::move(storage),root.allocator());
            return std::make_unique<Q16_1Tensor>(shape,std::move(storage),root.allocator());
        }
    },root_);
}
BufferTensorType CPUTensorStorageDeclaration::requireDtype(std::string_view dtype)
{
    if(dtype=="FP32") return BufferTensorType::FP32;
    if(dtype=="FP16") return BufferTensorType::FP16;
    if(dtype=="BF16") return BufferTensorType::BF16;
    if(dtype=="INT32") return BufferTensorType::INT32;
    if(dtype=="Q8_1") return BufferTensorType::Q8_1;
    if(dtype=="Q16_1") return BufferTensorType::Q16_1;
    throw std::invalid_argument("CPU tensor storage has unsupported dtype '"+std::string(dtype)+"'");
}
}
