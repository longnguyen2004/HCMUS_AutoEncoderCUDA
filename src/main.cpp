#include "core/cifar_reader.h"
#include <fstream>
#include <iostream>
#include <layer.h>
#include <memory>
#include <string>
#include <vector>
#include <random>

int main(int argc, char const *argv[])
{
    std::vector<Image> images;
    std::vector<int> labels;
    images.reserve(50000);
    labels.reserve(50000);

    for (int i = 1; i <= 5; ++i) {
        std::string path = "cifar-10-batches-bin/data_batch_" + std::to_string(i) + ".bin";
        std::ifstream file(path, std::ios_base::binary);
        if (!file) {
            std::cerr << "Error opening file: " << path << std::endl;
            continue;
        }
        auto batch = read_cifar10(file);
        std::cout << "Batch " << i << " size: " << batch.size() << '\n';
        images.insert(images.end(), batch.begin(), batch.end());
    }
    std::cout << "Total images: " << images.size() << std::endl;

    std::vector<std::shared_ptr<Layer>> layers;
    auto input = std::make_shared<InputGPU>();
    layers.push_back(input);

    // Encoder
    layers.push_back(std::make_shared<Conv2DGPU>(*layers.rbegin(), 3, 256));
    layers.push_back(std::make_shared<ReluGPU>(*layers.rbegin()));
    layers.push_back(std::make_shared<MaxPool2DGPU>(*layers.rbegin()));
    layers.push_back(std::make_shared<Conv2DGPU>(*layers.rbegin(), 3, 128));
    layers.push_back(std::make_shared<ReluGPU>(*layers.rbegin()));
    layers.push_back(std::make_shared<MaxPool2DGPU>(*layers.rbegin()));
    auto [x, y, c] = (*layers.rbegin())->dimension();
    std::cout << "Encoder output dimension: " << x << ' ' << y << ' ' << c << std::endl;

    // Decoder
    layers.push_back(std::make_shared<Conv2DGPU>(*layers.rbegin(), 3, 128));
    layers.push_back(std::make_shared<ReluGPU>(*layers.rbegin()));
    layers.push_back(std::make_shared<UpSample2DGPU>(*layers.rbegin()));
    layers.push_back(std::make_shared<Conv2DGPU>(*layers.rbegin(), 3, 256));
    layers.push_back(std::make_shared<ReluGPU>(*layers.rbegin()));
    layers.push_back(std::make_shared<UpSample2DGPU>(*layers.rbegin()));
    layers.push_back(std::make_shared<Conv2DGPU>(*layers.rbegin(), 3, 3));
    auto [x2, y2, c2] = (*layers.rbegin())->dimension();
    std::cout << "Decoder output dimension: " << x2 << ' ' << y2 << ' ' << c2 << std::endl;

    // Setting up params
    std::cout << "Initializing parameters...\n";
    float *params;
    size_t paramsCount = 0;
    for (const auto& layer: layers)
        paramsCount += layer->paramCount();
    std::vector<float> paramsVec(paramsCount);
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution dist(-1.0f, 1.0f);
    for (auto& param: paramsVec)
        param = dist(mt);
    CHECK(cudaMalloc(reinterpret_cast<void**>(&params), paramsCount * sizeof(float)));
    CHECK(cudaMemcpy(params, paramsVec.data(), paramsCount * sizeof(float), cudaMemcpyHostToDevice));

    size_t idx = 0;
    for (const auto& layer: layers)
    {
        layer->setParams(params + idx);
        idx += layer->paramCount();
    }

    // Here we go
    for (const auto& image: images)
    {
        input->setImage(image.data);
        (*layers.rbegin())->forward();
    }
    return 0;
}