#pragma once

#include <memory>
#include <vector>
#include <layer/base/layer_base.h>

std::vector<std::shared_ptr<LayerCPU>> make_encoder_cpu(std::shared_ptr<LayerCPU> input);
std::vector<std::shared_ptr<LayerGPU>> make_encoder_gpu(std::shared_ptr<LayerGPU> input);
std::vector<std::shared_ptr<LayerCPU>> make_decoder_cpu(std::shared_ptr<LayerCPU> input);
std::vector<std::shared_ptr<LayerGPU>> make_decoder_gpu(std::shared_ptr<LayerGPU> input);
