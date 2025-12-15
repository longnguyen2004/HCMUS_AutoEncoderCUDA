#include "network.h"
#include <layer/layer.h>
#include <memory>

std::vector<std::shared_ptr<LayerGPU>> make_encoder_gpu(
    std::shared_ptr<LayerGPU> input
) {
    std::vector<std::shared_ptr<LayerGPU>> layers;
    layers.push_back(std::make_shared<Conv2DGPU>(input, 3, 256));
    layers.push_back(std::make_shared<ReluGPU>(*layers.rbegin()));
    layers.push_back(std::make_shared<MaxPool2DGPU>(*layers.rbegin()));
    layers.push_back(std::make_shared<Conv2DGPU>(*layers.rbegin(), 3, 128));
    layers.push_back(std::make_shared<ReluGPU>(*layers.rbegin()));
    layers.push_back(std::make_shared<MaxPool2DGPU>(*layers.rbegin()));
    
    return layers;
}

std::vector<std::shared_ptr<LayerGPU>> make_decoder_gpu(
    std::shared_ptr<LayerGPU> input
) {
    std::vector<std::shared_ptr<LayerGPU>> layers;
    layers.push_back(std::make_shared<Conv2DGPU>(input, 3, 128));
    layers.push_back(std::make_shared<ReluGPU>(*layers.rbegin()));
    layers.push_back(std::make_shared<UpSample2DGPU>(*layers.rbegin()));
    layers.push_back(std::make_shared<Conv2DGPU>(*layers.rbegin(), 3, 256));
    layers.push_back(std::make_shared<ReluGPU>(*layers.rbegin()));
    layers.push_back(std::make_shared<UpSample2DGPU>(*layers.rbegin()));
    layers.push_back(std::make_shared<Conv2DGPU>(*layers.rbegin(), 3, 3));
    
    return layers;
}
