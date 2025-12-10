#include "core/cifar_reader.h"
#include <fstream>
#include <iostream>
#include <layer.h>
#include <memory>
#include <string>
#include <vector>
#include <random>

using namespace std::literals;

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

    auto output = std::make_shared<OutputGPU>(*layers.rbegin());
    layers.push_back(output);
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
    std::normal_distribution<float> dist(0.0f, 0.02f);  // He initialization approximation
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
    int epochs = 20;
    float learning_rate = 0.001f;
    std::vector<const Image*> image_refs;
    for (const auto &image: images)
        image_refs.push_back(&image);
    
    for (int i = 0; i < epochs; ++i)
    {
        std::shuffle(image_refs.begin(), image_refs.end(), mt);
        int img_count = 0;
        float loss_sum = 0.0f;
        
        for (const auto& image: images)
        {
            input->setImage(image.data);
            output->setReferenceImage(image.data);
            (*layers.rbegin())->forward();
            loss_sum += output->loss();
            img_count++;

            if (img_count % 100 == 0) {
                std::cout << "Epoch " << i << " Image " << img_count << " Avg Loss: " << (loss_sum / 100.0f) << std::endl;
                loss_sum = 0.0f;
            }
            
            (*layers.rbegin())->backward(learning_rate, nullptr);
        }

        std::ofstream paramsOut("params_epoch_"s + std::to_string(i) + ".bin"s);
        cudaMemcpy(paramsVec.data(), params, paramsVec.size() * sizeof(float), cudaMemcpyDeviceToHost);
        paramsOut.write(reinterpret_cast<char*>(paramsVec.data()), paramsVec.size() * sizeof(float));
    }
    return 0;
}