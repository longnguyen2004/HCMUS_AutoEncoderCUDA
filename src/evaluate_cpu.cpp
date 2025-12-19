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

    // Load test set
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

    // Build the same model architecture (CPU version)
    std::vector<std::shared_ptr<Layer>> layers;
    auto input = std::make_shared<InputCPU>();
    layers.push_back(input);

    // Encoder
    auto encoder = make_encoder_cpu(input);
    auto [x, y, c] = (*layers.rbegin())->dimension();
    std::cout << "Encoder output dimension: " << x << ' ' << y << ' ' << c << std::endl;

    // Decoder
    auto decoder = make_decoder_cpu(*encoder.rbegin());
    layers.insert(layers.end(), encoder.begin(), encoder.end());
    layers.insert(layers.end(), decoder.begin(), decoder.end());

    auto output = std::make_shared<OutputCPU>(*layers.rbegin());
    layers.push_back(output);
    auto [x2, y2, c2] = (*layers.rbegin())->dimension();
    std::cout << "Decoder output dimension: " << x2 << ' ' << y2 << ' ' << c2 << std::endl;

    std::cout << "Loading parameters from params_epoch_1.bin..." << std::endl;
    float *params;
    size_t paramsCount = 0;
    for (const auto& layer: layers)
        paramsCount += layer->paramCount();
    
    std::cout << "Total parameters: " << paramsCount << std::endl;
    
    std::vector<float> paramsVec(paramsCount);
    std::ifstream paramsIn("params_epoch_19.bin", std::ios::binary);
    if (!paramsIn) {
        std::cerr << "Error opening params_epoch_1.bin" << std::endl;
        return 1;
    }
    
    paramsIn.read(reinterpret_cast<char*>(paramsVec.data()), paramsCount * sizeof(float));
    if (!paramsIn) {
        std::cerr << "Error reading parameters. Expected " << paramsCount << " floats." << std::endl;
        return 1;
    }
    paramsIn.close();
    std::cout << "Parameters loaded successfully." << std::endl;

    // Allocate parameters in CPU memory
    params = new float[paramsCount];
    std::copy(paramsVec.begin(), paramsVec.end(), params);

    size_t idx = 0;
    for (const auto& layer: layers)
    {
        layer->setParams(params + idx);
        idx += layer->paramCount();
    }

    std::cout << "\nEvaluating on test set..." << std::endl;
    float total_loss = 0.0f;
    int img_count = 0;
    // cuML features export
    int enc_x = x, enc_y = y, enc_c = c;
    const size_t feature_size = static_cast<size_t>(enc_x) * static_cast<size_t>(enc_y) * static_cast<size_t>(enc_c);
    std::ofstream feat_out("features.csv", std::ios::binary);
    std::ofstream label_out("labels.csv", std::ios::binary);
    if (!feat_out || !label_out) {
        std::cerr << "Failed to open output files features.csv or labels.csv" << std::endl;
        return 1;
    }
    
    std::filesystem::create_directory("test_outputs");
    
    for (const auto& image: test_images)
    {
        input->setImage(image.data);
        output->setReferenceImage(image.data);
        (*layers.rbegin())->forward();

        // Grab encoder output features
        const auto &encoder_layer = *encoder.rbegin();
        const float *enc_dev = encoder_layer->output();
        std::vector<float> enc_host(feature_size);
        if (encoder_layer->deviceType() == DeviceType::GPU) {
            // Should not happen in CPU version, but keeping for safety
            std::cerr << "Error: GPU layer detected in CPU evaluation!" << std::endl;
            return 1;
        } else {
            std::copy(enc_dev, enc_dev + feature_size, enc_host.begin());
        }
        // Write row to CSV (comma-separated), and label to labels.csv
        for (size_t i = 0; i < feature_size; ++i) {
            feat_out << enc_host[i];
            if (i + 1 < feature_size) feat_out << ',';
        }
        feat_out << '\n';
        label_out << static_cast<int>(image.label) << '\n';
        
        float current_loss = output->loss();
        total_loss += current_loss;
        img_count++;
        
        if (img_count <= 20) {
            auto output_data = output->output();
            std::vector<float> reconstructed(3072);
            // CPU version - direct copy
            std::copy(output_data, output_data + 3072, reconstructed.begin());
            
            std::vector<uint8_t> img_u8(3072);
            for (int i = 0; i < 3072; i++) {
                float val = reconstructed[i] * 255.0f;
                val = std::max(0.0f, std::min(255.0f, val));
                img_u8[i] = static_cast<uint8_t>(val);
            }
            
            std::vector<uint8_t> img_hwc(3072);
            planar_to_packed(img_hwc.data(), img_u8.data(), img_u8.data() + 1024, img_u8.data() + 2048, 1024);
            
            std::vector<uint8_t> orig_u8(3072);
            for (int i = 0; i < 3072; i++) {
                float val = image.data[i] * 255.0f;
                orig_u8[i] = static_cast<uint8_t>(val);
            }
            
            std::vector<uint8_t> orig_hwc(3072);
            planar_to_packed(orig_hwc.data(), orig_u8.data(), orig_u8.data() + 1024, orig_u8.data() + 2048, 1024);
            
            std::string orig_filename = "test_outputs/original_" + std::to_string(img_count) + ".png";
            std::string recon_filename = "test_outputs/reconstructed_" + std::to_string(img_count) + ".png";
            stbi_write_png(orig_filename.c_str(), 32, 32, 3, orig_hwc.data(), 32 * 3);
            stbi_write_png(recon_filename.c_str(), 32, 32, 3, img_hwc.data(), 32 * 3);
        }
        
        if (img_count % 100 == 0) {
            float avg_loss = total_loss / img_count;
            std::cout << "Processed " << img_count << " images, Average Loss: " << avg_loss << std::endl;
        }
    }
    
    float final_avg_loss = total_loss / img_count;
    std::cout << "\n========================================" << std::endl;
    std::cout << "Evaluation Complete!" << std::endl;
    std::cout << "Total test images: " << img_count << std::endl;
    std::cout << "Average test loss: " << final_avg_loss << std::endl;
    std::cout << "First 20 reconstructed images saved to test_outputs/" << std::endl;
    std::cout << "Features saved to features.csv and labels to labels.csv" << std::endl;
    std::cout << "========================================" << std::endl;
    
    delete[] params;
    
    return 0;
}
