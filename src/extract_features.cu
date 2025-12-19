#include "core/cifar_reader.h"
#include "core/network.h"
#include <fstream>
#include <iostream>
#include <cmath>
#include <filesystem>
#include <layer/layer.h>
#include <memory>
#include <string>
#include <vector>
#include <algorithm>
#include <helper/gpu_helper.h>

#include "stb_image_write.h"
#include "planar/planar.h"

using namespace std::literals;

int main(int argc, char const *argv[])
{
    // Parse command line arguments
    std::string dataset_dir = "../dataset/cifar-10-batches-bin";
    if (argc > 1) {
        dataset_dir = argv[1];
    }

    std::cout << "Using dataset directory: " << dataset_dir << std::endl;
    std::cout << "Loading CIFAR-10 training set..." << std::endl;
    std::vector<Image> train_images;
    
    for (int batch = 1; batch <= 5; batch++) {
        std::string train_path = dataset_dir + "/data_batch_" + std::to_string(batch) + ".bin";
        std::ifstream train_file(train_path, std::ios_base::binary);
        if (!train_file) {
            std::cerr << "Error opening train file: " << train_path << std::endl;
            return 1;
        }
        auto batch_images = read_cifar10(train_file);
        train_images.insert(train_images.end(), batch_images.begin(), batch_images.end());
        std::cout << "Loaded batch " << batch << ": " << batch_images.size() << " images" << std::endl;
    }
    std::cout << "Training set size: " << train_images.size() << std::endl;

    std::cout << "Loading CIFAR-10 test set..." << std::endl;
    std::vector<Image> test_images;
    std::string test_path = dataset_dir + "/test_batch.bin";
    std::ifstream test_file(test_path, std::ios_base::binary);
    if (!test_file) {
        std::cerr << "Error opening test file: " << test_path << std::endl;
        return 1;
    }
    test_images = read_cifar10(test_file);
    std::cout << "Test set size: " << test_images.size() << std::endl;

    std::vector<std::shared_ptr<Layer>> layers;
    auto input = std::make_shared<InputGPU>();
    layers.push_back(input);

    auto encoder = make_encoder_gpu(input);
    layers.insert(layers.end(), encoder.begin(), encoder.end());
    
    auto [x, y, c] = (*layers.rbegin())->dimension();
    std::cout << "Encoder output dimension (Latent Space): " << x << ' ' << y << ' ' << c << std::endl;

    auto decoder = make_decoder_gpu(*encoder.rbegin());
    layers.insert(layers.end(), decoder.begin(), decoder.end());

    auto output = std::make_shared<OutputGPU>(*layers.rbegin());
    layers.push_back(output);
    auto [x2, y2, c2] = (*layers.rbegin())->dimension();
    std::cout << "Decoder output dimension: " << x2 << ' ' << y2 << ' ' << c2 << std::endl;

    std::cout << "Loading parameters from params_epoch_19.bin..." << std::endl;
    float *params;
    size_t paramsCount = 0;
    for (const auto& layer: layers)
        paramsCount += layer->paramCount();
    
    std::cout << "Total parameters: " << paramsCount << std::endl;
    
    std::vector<float> paramsVec(paramsCount);
    std::ifstream paramsIn("params_epoch_19.bin", std::ios::binary);
    if (!paramsIn) {
        std::cerr << "Error opening params_epoch_19.bin" << std::endl;
        return 1;
    }
    
    paramsIn.read(reinterpret_cast<char*>(paramsVec.data()), paramsCount * sizeof(float));
    if (!paramsIn) {
        std::cerr << "Error reading parameters. Expected " << paramsCount << " floats." << std::endl;
        return 1;
    }
    paramsIn.close();
    std::cout << "Parameters loaded successfully." << std::endl;

    CHECK(cudaMalloc(reinterpret_cast<void**>(&params), paramsCount * sizeof(float)));
    CHECK(cudaMemcpy(params, paramsVec.data(), paramsCount * sizeof(float), cudaMemcpyHostToDevice));

    size_t idx = 0;
    for (const auto& layer: layers)
    {
        layer->setParams(params + idx);
        idx += layer->paramCount();
    }

    int enc_x = x, enc_y = y, enc_c = c;
    const size_t feature_size = static_cast<size_t>(enc_x) * static_cast<size_t>(enc_y) * static_cast<size_t>(enc_c);
    std::cout << "Feature size for export: " << feature_size << " (should be 8192 for 8x8x128 latent space)" << std::endl;
    
    std::cout << "\nExtracting features from training set..." << std::endl;
    std::ofstream train_feat_out("train_features.csv", std::ios::binary);
    std::ofstream train_label_out("train_labels.csv", std::ios::binary);
    if (!train_feat_out || !train_label_out) {
        std::cerr << "Failed to open output files train_features.csv or train_labels.csv" << std::endl;
        return 1;
    }
    
    float train_total_loss = 0.0f;
    int train_img_count = 0;
    
    for (const auto& image: train_images)
    {
        input->setImage(image.data);
        output->setReferenceImage(image.data);
        (*layers.rbegin())->forward();

        const auto &encoder_layer = *encoder.rbegin();
        const float *enc_dev = encoder_layer->output();
        std::vector<float> enc_host(feature_size);
        if (encoder_layer->deviceType() == DeviceType::GPU) {
            CHECK(cudaMemcpy(enc_host.data(), enc_dev, feature_size * sizeof(float), cudaMemcpyDeviceToHost));
        } else {
            std::copy(enc_dev, enc_dev + feature_size, enc_host.begin());
        }
        for (size_t i = 0; i < feature_size; ++i) {
            train_feat_out << enc_host[i];
            if (i + 1 < feature_size) train_feat_out << ',';
        }
        train_feat_out << '\n';
        train_label_out << static_cast<int>(image.label) << '\n';
        
        float current_loss = output->loss();
        train_total_loss += current_loss;
        train_img_count++;
        
        if (train_img_count % 1000 == 0) {
            float avg_loss = train_total_loss / train_img_count;
            std::cout << "Processed " << train_img_count << " images, Average Loss: " << avg_loss << std::endl;
        }
    }
    
    train_feat_out.close();
    train_label_out.close();
    
    float train_final_avg_loss = train_total_loss / train_img_count;
    std::cout << "\n========================================" << std::endl;
    std::cout << "Training Feature Extraction Complete!" << std::endl;
    std::cout << "Total training images: " << train_img_count << std::endl;
    std::cout << "Average loss: " << train_final_avg_loss << std::endl;
    std::cout << "Features saved to train_features.csv and labels to train_labels.csv" << std::endl;
    std::cout << "========================================" << std::endl;
    
    std::cout << "\nExtracting features from test set..." << std::endl;
    std::ofstream test_feat_out("test_features.csv", std::ios::binary);
    std::ofstream test_label_out("test_labels.csv", std::ios::binary);
    if (!test_feat_out || !test_label_out) {
        std::cerr << "Failed to open output files test_features.csv or test_labels.csv" << std::endl;
        return 1;
    }
    
    float test_total_loss = 0.0f;
    int test_img_count = 0;
    
    for (const auto& image: test_images)
    {
        input->setImage(image.data);
        output->setReferenceImage(image.data);
        (*layers.rbegin())->forward();

        const auto &encoder_layer = *encoder.rbegin();
        const float *enc_dev = encoder_layer->output();
        std::vector<float> enc_host(feature_size);
        if (encoder_layer->deviceType() == DeviceType::GPU) {
            CHECK(cudaMemcpy(enc_host.data(), enc_dev, feature_size * sizeof(float), cudaMemcpyDeviceToHost));
        } else {
            std::copy(enc_dev, enc_dev + feature_size, enc_host.begin());
        }
        for (size_t i = 0; i < feature_size; ++i) {
            test_feat_out << enc_host[i];
            if (i + 1 < feature_size) test_feat_out << ',';
        }
        test_feat_out << '\n';
        test_label_out << static_cast<int>(image.label) << '\n';
        
        float current_loss = output->loss();
        test_total_loss += current_loss;
        test_img_count++;
        
        if (test_img_count % 100 == 0) {
            float avg_loss = test_total_loss / test_img_count;
            std::cout << "Processed " << test_img_count << " images, Average Loss: " << avg_loss << std::endl;
        }
    }
    
    test_feat_out.close();
    test_label_out.close();
    
    float test_final_avg_loss = test_total_loss / test_img_count;
    std::cout << "\n========================================" << std::endl;
    std::cout << "Test Feature Extraction Complete!" << std::endl;
    std::cout << "Total test images: " << test_img_count << std::endl;
    std::cout << "Average loss: " << test_final_avg_loss << std::endl;
    std::cout << "Features saved to test_features.csv and labels to test_labels.csv" << std::endl;
    std::cout << "========================================" << std::endl;
    
    cudaFree(params);
    
    return 0;
}

