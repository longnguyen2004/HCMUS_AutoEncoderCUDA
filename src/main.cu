#include "core/cifar_reader.h"
#include "core/network.h"
#include <fstream>
#include <iostream>
#include <cmath>
#include <layer/layer.h>
#include <memory>
#include <string>
#include <vector>
#include <random>
#include <algorithm>
#include <nvtx3/nvToolsExt.h>
#include <algorithm>
#include <helper/gpu_helper.h>

using namespace std::literals;

int main(int argc, char const *argv[])
{
    std::vector<Image> images;
    std::vector<int> labels;
    images.reserve(50000);
    labels.reserve(50000);

    for (int i = 1; i <= 5; ++i) {
        std::string path = "../dataset/cifar-10-batches-bin/data_batch_" + std::to_string(i) + ".bin";
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

    auto input = std::make_shared<InputGPU>();

    // Encoder
    auto encoder = make_encoder_gpu(input);
    auto [x, y, c] = (*encoder.rbegin())->dimension();
    std::cout << "Encoder output dimension: " << x << ' ' << y << ' ' << c << std::endl;

    // Decoder
    auto decoder = make_decoder_gpu(*encoder.rbegin());
    auto [x2, y2, c2] = (*decoder.rbegin())->dimension();
    std::cout << "Decoder output dimension: " << x2 << ' ' << y2 << ' ' << c2 << std::endl;

    auto output = std::make_shared<OutputGPU>(*decoder.rbegin());

    std::vector<std::shared_ptr<LayerGPU>> layers;
    layers.push_back(input);
    layers.insert(layers.end(), encoder.begin(), encoder.end());
    layers.insert(layers.end(), decoder.begin(), decoder.end());
    layers.push_back(output);

    // Setting up params
    std::cout << "Initializing parameters...\n";
    float *params;
    size_t paramsCount = 0;
    for (const auto& layer: layers)
        paramsCount += layer->paramCount();
    std::vector<float> paramsVec(paramsCount);
    std::random_device rd;
    std::mt19937 mt(rd());

    size_t params_idx = 0;
    for (size_t i = 0; i < layers.size(); ++i)
    {
        auto& layer = layers[i];
        if (auto conv2d = std::dynamic_pointer_cast<Conv2DGPU>(layer); conv2d != nullptr)
        {
            auto [prev_x, prev_y, prev_z] = layers[i - 1]->dimension();
            std::normal_distribution<float> dist(0.0f, std::sqrt(2.0f / (3 * 3 * prev_z)));
            for (size_t j = 0; j < conv2d->weightCount(); ++j)
                paramsVec[params_idx + j] = dist(mt);
            for (size_t j = conv2d->weightCount(); j < conv2d->paramCount(); ++j)
                paramsVec[params_idx + j] = 0.0f;
        }
        params_idx += layer->paramCount();
    }
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
    
    GPUTimer timer;
    
    for (int i = 0; i < epochs; ++i)
    {
        std::cout << "=== Epoch " << i << " ===" << std::endl;
        
        std::string msg = "Epoch " + std::to_string(i);
        nvtxRangePushA(msg.c_str());
        
        std::shuffle(image_refs.begin(), image_refs.end(), mt);
        int img_count = 0;
        float loss_sum = 0.0f;
        timer.Start();
        
        for (const auto& image: image_refs)
        {
            input->setImage(image->data);
            output->setReferenceImage(image->data);
            output->forward();
            
            float current_loss = output->loss();
            if (std::isnan(current_loss) || std::isinf(current_loss)) {
                std::cerr << "NaN/Inf detected at epoch " << i << " image " << img_count << std::endl;
                std::cerr << "Loss was: " << loss_sum / (std::max)(1, img_count % 100) << std::endl;
                return 1;
            }
            
            loss_sum += current_loss;
            img_count++;

            if (img_count % 100 == 0) {
                timer.Stop();
                float elapsed = timer.Elapsed() / 1000.0f; // Convert ms to seconds
                float avg_loss = loss_sum / 100.0f;
                std::cout << "Epoch " << i << " Image " << img_count << " Avg Loss: " << avg_loss << " Time: " << elapsed << "s" << std::endl;
                loss_sum = 0.0f;
                timer.Start();
            }
            
            output->backward(learning_rate, nullptr);
        }

        std::ofstream paramsOut("params_epoch_"s + std::to_string(i) + ".bin"s, std::ios::binary);
        cudaMemcpy(paramsVec.data(), params, paramsVec.size() * sizeof(float), cudaMemcpyDeviceToHost);
        paramsOut.write(reinterpret_cast<char*>(paramsVec.data()), paramsVec.size() * sizeof(float));
        paramsOut.close();
        
        std::cout << "Epoch " << i << " completed. Parameters saved." << std::endl;
        
        nvtxRangePop();
    }
    return 0;
}