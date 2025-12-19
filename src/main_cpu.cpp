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

using namespace std::literals;
#include <chrono>

int main(int argc, char const *argv[])
{
    // Parse command line arguments
    std::string dataset_dir = "../dataset/cifar-10-batches-bin";
    if (argc > 1) {
        dataset_dir = argv[1];
    }

    std::cout << "Using dataset directory: " << dataset_dir << std::endl;

    std::vector<Image> images;
    std::vector<int> labels;
    images.reserve(50000);
    labels.reserve(50000);

    for (int i = 1; i <= 5; ++i) {
        std::string path = dataset_dir + "/data_batch_" + std::to_string(i) + ".bin";
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

    auto input = std::make_shared<InputCPU>();

    // Encoder
    auto encoder = make_encoder_cpu(input);
    auto [x, y, c] = (*encoder.rbegin())->dimension();
    std::cout << "Encoder output dimension: " << x << ' ' << y << ' ' << c << std::endl;

    // Decoder
    auto decoder = make_decoder_cpu(*encoder.rbegin());
    auto [x2, y2, c2] = (*decoder.rbegin())->dimension();
    std::cout << "Decoder output dimension: " << x2 << ' ' << y2 << ' ' << c2 << std::endl;

    auto output = std::make_shared<OutputCPU>(*decoder.rbegin());

    std::vector<std::shared_ptr<LayerCPU>> layers;
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
        if (auto conv2d = std::dynamic_pointer_cast<Conv2DCPU>(layer); conv2d != nullptr)
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
    
    // Allocate parameters in CPU memory
    params = new float[paramsCount];
    std::copy(paramsVec.begin(), paramsVec.end(), params);

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
    
    constexpr float TRAIN_DATA_PERCENTAGE = 0.02f;
    size_t train_size = static_cast<size_t>(image_refs.size() * TRAIN_DATA_PERCENTAGE);
    if (train_size < image_refs.size()) {
        image_refs.resize(train_size);
        std::cout << "Training on " << train_size << " images (" << (TRAIN_DATA_PERCENTAGE * 100) << "%)" << std::endl;
    }

    for (int i = 0; i < epochs; ++i)
    {
        std::cout << "=== Epoch " << i << " ===" << std::endl;
        
        std::shuffle(image_refs.begin(), image_refs.end(), mt);
        int img_count = 0;
        float loss_sum = 0.0f;
        auto start_time = std::chrono::high_resolution_clock::now();
        
        for (const auto& image: image_refs)
        {
            input->setImage(image->data);
            output->setReferenceImage(image->data);
            output->forward();
            
            float current_loss = output->loss();
            if (std::isnan(current_loss) || std::isinf(current_loss)) {
                std::cerr << "NaN/Inf detected at epoch " << i << " image " << img_count << std::endl;
                std::cerr << "Loss was: " << loss_sum / (std::max)(1, img_count % 100) << std::endl;
                delete[] params;
                return 1;
            }
            
            loss_sum += current_loss;
            img_count++;

            if (img_count % 100 == 0) {
                auto current_time = std::chrono::high_resolution_clock::now();
                std::chrono::duration<float> elapsed = current_time - start_time;
                float avg_loss = loss_sum / 100.0f;
                std::cout << "Epoch " << i << " Image " << img_count << " Avg Loss: " << avg_loss << " Time: " << elapsed.count() << "s" << std::endl;
                start_time = current_time;
                loss_sum = 0.0f;
            }
            
            output->backward(learning_rate, nullptr);
        }

        std::ofstream paramsOut("params_epoch_"s + std::to_string(i) + ".bin"s, std::ios::binary);
        std::copy(params, params + paramsCount, paramsVec.begin());
        paramsOut.write(reinterpret_cast<char*>(paramsVec.data()), paramsVec.size() * sizeof(float));
        paramsOut.close();
        
        std::cout << "Epoch " << i << " completed. Parameters saved." << std::endl;
    }
    
    delete[] params;
    
    return 0;
}
