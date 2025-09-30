#include "model_loader.h"
#include "logger.h"
#include <iostream>
#include <limits>
#include <cmath>

int main(int argc, char **argv)
{
    initializeLogging();
    if (argc < 3)
    {
        std::cerr << "usage: inspect_embedding <gguf> <tensor_name>" << std::endl;
        return 1;
    }
    ModelLoader loader;
    if (!loader.loadModel(argv[1]))
    {
        std::cerr << "loadModel failed" << std::endl;
        return 1;
    }
    auto tensor = loader.loadTensor(argv[2]);
    if (!tensor)
    {
        std::cerr << "loadTensor failed" << std::endl;
        return 1;
    }
    auto shape = tensor->shape();
    std::cout << "shape=";
    for (size_t i = 0; i < shape.size(); ++i)
    {
        if (i)
            std::cout << 'x';
        std::cout << shape[i];
    }
    std::cout << std::endl;
    const float *data = tensor->data();
    size_t total = 1;
    for (int dim : shape)
    {
        total *= static_cast<size_t>(dim);
    }
    double min_v = std::numeric_limits<double>::infinity();
    double max_v = -std::numeric_limits<double>::infinity();
    size_t nan_count = 0;
    size_t inf_count = 0;
    size_t zero_count = 0;
    for (size_t i = 0; i < total; ++i)
    {
        float v = data[i];
        if (std::isnan(v))
        {
            ++nan_count;
            continue;
        }
        if (std::isinf(v))
        {
            ++inf_count;
            continue;
        }
        if (v == 0.0f)
        {
            ++zero_count;
        }
        if (v < min_v)
            min_v = v;
        if (v > max_v)
            max_v = v;
    }
    std::cout << "min=" << min_v << " max=" << max_v << std::endl;
    std::cout << "zero_count=" << zero_count << " nan_count=" << nan_count << " inf_count=" << inf_count << std::endl;
    std::cout << "first10=";
    for (int i = 0; i < 10 && i < static_cast<int>(total); ++i)
    {
        if (i)
            std::cout << ',';
        std::cout << data[i];
    }
    std::cout << std::endl;
    return 0;
}
