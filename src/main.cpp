#include <fstream>
#include "core/cifar_reader.h"
#include <iostream>
#include <layer.h>
#include <memory>
#include <string>

int main(int argc, char const *argv[])
{
    std::vector<float> images;
    std::vector<int> labels;
    images.reserve(50000 * 3072);
    labels.reserve(50000);

    for (int i = 1; i <= 5; ++i) {
        std::string path = "/home/minhnhat/Documents/HCMUS_AutoEncoderCUDA/dataset/cifar-10-batches-bin/data_batch_" + std::to_string(i) + ".bin";
        std::ifstream file(path, std::ios_base::binary);
        if (!file) {
            std::cerr << "Error opening file: " << path << std::endl;
            continue;
        }
        auto batch = read_cifar10(file);
        std::cout << "Batch " << i << " size: " << batch.size() << '\n';
        for (const auto& image : batch) {
            images.insert(images.end(), image.data.begin(), image.data.end());
            labels.push_back(image.label);
        }
    }
    std::cout << "Total images: " << labels.size() << std::endl;
    std::shared_ptr<Layer> layer;
    layer = std::make_shared<InputGPU>(images);
    layer = std::make_shared<ReluGPU>(layer);
    layer = std::make_shared<MaxPool2DGPU>(layer);
    layer->forward();
    auto [x, y, c] = layer->dimension();
    std::cout << x << ' ' << y << ' ' << c << std::endl;
    int a;
    std::cin >> a;
}