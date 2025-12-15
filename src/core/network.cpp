#include "network.h"
#include <layer/layer.h>
#include <memory>

std::vector<std::shared_ptr<LayerCPU>> make_encoder_cpu(
    std::shared_ptr<LayerCPU> input
) {
    std::vector<std::shared_ptr<LayerCPU>> layers;
    layers.push_back(std::make_shared<Conv2DCPU>(input, 3, 256));
    layers.push_back(std::make_shared<ReluCPU>(*layers.rbegin()));
    layers.push_back(std::make_shared<MaxPool2DCPU>(*layers.rbegin()));
    layers.push_back(std::make_shared<Conv2DCPU>(*layers.rbegin(), 3, 128));
    layers.push_back(std::make_shared<ReluCPU>(*layers.rbegin()));
    layers.push_back(std::make_shared<MaxPool2DCPU>(*layers.rbegin()));
    
    return layers;
}

std::vector<std::shared_ptr<LayerCPU>> make_decoder_cpu(
    std::shared_ptr<LayerCPU> input
) {
    std::vector<std::shared_ptr<LayerCPU>> layers;
    layers.push_back(std::make_shared<Conv2DCPU>(input, 3, 128));
    layers.push_back(std::make_shared<ReluCPU>(*layers.rbegin()));
    layers.push_back(std::make_shared<UpSample2DCPU>(*layers.rbegin()));
    layers.push_back(std::make_shared<Conv2DCPU>(*layers.rbegin(), 3, 256));
    layers.push_back(std::make_shared<ReluCPU>(*layers.rbegin()));
    layers.push_back(std::make_shared<UpSample2DCPU>(*layers.rbegin()));
    layers.push_back(std::make_shared<Conv2DCPU>(*layers.rbegin(), 3, 3));
    
    return layers;
}
