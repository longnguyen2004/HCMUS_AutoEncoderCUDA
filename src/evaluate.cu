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

#define WEIGHTS_FILE "params_epoch_19.bin"

int main(int argc, char const *argv[])
{
    // Load test set
    std::cout << "Loading CIFAR-10 test set..." << std::endl;
    std::vector<Image> test_images;
    std::string test_path = "../dataset/cifar-10-batches-bin/test_batch.bin";
    std::ifstream test_file(test_path, std::ios_base::binary);
    if (!test_file) {
        std::cerr << "Error opening test file: " << test_path << std::endl;
        return 1;
    }
    test_images = read_cifar10(test_file);
    std::cout << "Test set size: " << test_images.size() << std::endl;

    // Build the same model architecture
    std::vector<std::shared_ptr<Layer>> layers;
    auto input = std::make_shared<InputGPU>();
    layers.push_back(input);

    // Encoder
    auto encoder = make_encoder_gpu(input);
    // Add encoder to layers BEFORE capturing dimensions
    layers.insert(layers.end(), encoder.begin(), encoder.end());
    
    // Now get the correct latent space dimensions (8x8x128)
    auto [x, y, c] = (*layers.rbegin())->dimension();
    std::cout << "Encoder output dimension (Latent Space): " << x << ' ' << y << ' ' << c << std::endl;

    // Decoder
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

    // Copy parameters to GPU
    CHECK(cudaMalloc(reinterpret_cast<void**>(&params), paramsCount * sizeof(float)));
    CHECK(cudaMemcpy(params, paramsVec.data(), paramsCount * sizeof(float), cudaMemcpyHostToDevice));

    size_t idx = 0;
    for (const auto& layer: layers)
    {
        layer->setParams(params + idx);
        idx += layer->paramCount();
    }

    std::cout << "\nEvaluating on test set..." << std::endl;
    float total_loss = 0.0f;
    int img_count = 0;
    
    std::filesystem::create_directory("test_outputs");
    
    for (const auto& image: test_images)
    {
        input->setImage(image.data);
        output->setReferenceImage(image.data);
        (*layers.rbegin())->forward();
        
        float current_loss = output->loss();
        total_loss += current_loss;
        img_count++;
        
        if (img_count <= 20) {
            auto output_data = output->output();
            std::vector<float> reconstructed(3072);
            CHECK(cudaMemcpy(reconstructed.data(), output_data, 3072 * sizeof(float), cudaMemcpyDeviceToHost));
            
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
    std::cout << "========================================" << std::endl;
    
    cudaFree(params);
    
    return 0;
}
