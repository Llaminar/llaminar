#pragma once

#include "tensor_base.h"
#include <vector>
#include <algorithm>
#include <stdexcept>

namespace llaminar {

/**
 * Simple in-memory tensor implementation for non-distributed operations.
 * Compatible with our existing Tensor struct but wrapped in the TensorBase interface.
 */
class SimpleTensor : public TensorBase {
private:
    std::vector<float> data_;
    std::vector<int> shape_;

public:
    // Constructors
    SimpleTensor() = default;
    
    explicit SimpleTensor(const std::vector<int>& dims) : shape_(dims) {
        int total_size = 1;
        for (int dim : dims) {
            if (dim <= 0) {
                throw std::invalid_argument("Tensor dimensions must be positive");
            }
            total_size *= dim;
        }
        data_.resize(total_size);
    }
    
    SimpleTensor(const std::vector<int>& dims, const std::vector<float>& values) 
        : shape_(dims), data_(values) {
        if (total_elements() != static_cast<int>(values.size())) {
            throw std::invalid_argument("Data size does not match tensor shape");
        }
    }
    
    // Constructor for compatibility with double data (from existing code)
    SimpleTensor(const std::vector<int>& dims, const std::vector<double>& values) : shape_(dims) {
        data_.reserve(values.size());
        for (double val : values) {
            data_.push_back(static_cast<float>(val));
        }
    }

    // TensorBase interface implementation
    const std::vector<int>& shape() const override { return shape_; }
    
    int size() const override { return static_cast<int>(data_.size()); }
    
    int ndim() const override { return static_cast<int>(shape_.size()); }
    
    float* data() override { return data_.data(); }
    
    const float* data() const override { return data_.data(); }
    
    std::string type_name() const override { return "SimpleTensor"; }
    
    bool is_distributed() const override { return false; }
    
    void zero() override {
        std::fill(data_.begin(), data_.end(), 0.0f);
    }
    
    void fill(float value) override {
        std::fill(data_.begin(), data_.end(), value);
    }
    
    std::shared_ptr<TensorBase> copy() const override {
        return std::make_shared<SimpleTensor>(shape_, data_);
    }
    
    void copy_from(const TensorBase& other) override {
        if (!is_compatible_shape(other)) {
            throw std::invalid_argument("Incompatible tensor shapes for copy");
        }
        
        // If other is also a SimpleTensor, we can do efficient copy
        if (auto simple_other = dynamic_cast<const SimpleTensor*>(&other)) {
            data_ = simple_other->data_;
        } else {
            // Generic copy using data pointer
            const float* other_data = other.data();
            std::copy(other_data, other_data + size(), data_.begin());
        }
    }

    // Direct access to underlying data for compatibility
    std::vector<float>& get_data() { return data_; }
    const std::vector<float>& get_data() const { return data_; }
    
    std::vector<int>& get_shape() { return shape_; }
    const std::vector<int>& get_shape() const { return shape_; }

    // Utility methods
    void reshape(const std::vector<int>& new_shape) {
        int new_size = 1;
        for (int dim : new_shape) {
            new_size *= dim;
        }
        
        if (new_size != size()) {
            throw std::invalid_argument("Cannot reshape tensor: size mismatch");
        }
        
        shape_ = new_shape;
    }
    
    void resize(const std::vector<int>& new_shape) {
        shape_ = new_shape;
        int new_size = 1;
        for (int dim : new_shape) {
            new_size *= dim;
        }
        data_.resize(new_size);
    }

    // Compatibility with legacy Tensor struct
    struct LegacyView {
        std::vector<float>& data;
        std::vector<int>& shape;
        
        LegacyView(SimpleTensor& tensor) : data(tensor.data_), shape(tensor.shape_) {}
    };
    
    LegacyView legacy_view() { return LegacyView(*this); }
};

} // namespace llaminar