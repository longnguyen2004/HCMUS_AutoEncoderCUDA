#include "./layer_base.h"
#include <helper/gpu_helper.h>

LayerGPU::~LayerGPU()
{
  CHECK(cudaFree(m_output));
  CHECK(cudaFree(grad_input));
}
