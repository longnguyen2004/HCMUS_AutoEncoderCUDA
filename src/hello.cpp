#include "convolution/convolution.h"
#include "planar/planar.h"
#include <stb_image.h>
#include <stb_image_write.h>
#include <algorithm>
#include <memory>
#include <vector>

int main(int argc, char const *argv[])
{
    int x, y, channels;
    auto img = stbi_load("in.png", &x, &y, &channels, 0);
    channels = 3;
    std::vector<float> in(x * y * channels);
    std::vector<float> out(x * y * channels);
    std::vector<float> in_r(x * y), in_g(x * y), in_b(x * y);
    std::vector<float> out_r(x * y), out_g(x * y), out_b(x * y);
    std::copy(img, img + (x * y * channels), in.begin());
    packed_to_planar(in.data(), in_r.data(), in_g.data(), in_b.data(), x * y);
    float n = 1.0f/9;
    std::vector<float> kernel = {n,n,n,n,n,n,n,n,n};
    std::unique_ptr<Convolution> convolver(new ConvolutionCpu);
    convolver->convolve(
        out_r.data(), out_g.data(), out_b.data(),
        in_r.data(), in_g.data(), in_b.data(),
        kernel.data(),
        y, x, 3
    );
    planar_to_packed(out.data(), out_r.data(), out_g.data(), out_b.data(), x * y);
    std::vector<std::uint8_t> out_u8(out.begin(), out.end());
    stbi_write_png("out.png", x, y, channels, out_u8.data(), x * channels);
    return 0;
}
