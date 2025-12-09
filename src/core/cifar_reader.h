#pragma once
#include <istream>
#include <cstdlib>
#include <vector>
#include <cstdint>

struct Image
{
  uint8_t label;
  std::vector<float> data;
};

std::vector<Image> read_cifar10(std::istream& in)
{
  std::vector<Image> images;
  uint8_t image_u8[3072];
  while (true)
  {
    Image image;
    if (!in.read(reinterpret_cast<char*>(&image.label), 1))
      break;
    image.data.resize(3072);
    if (!in.read(reinterpret_cast<char*>(image_u8), 3072))
      break;
    for (int i = 0; i < 3072; i++)
      image.data[i] = image_u8[i] / 255.0f;
    images.push_back(std::move(image));
  }
  return images;
}
