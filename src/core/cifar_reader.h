#pragma once
#include <istream>
#include <cstdlib>
#include <vector>

struct Image
{
  uint8_t label;
  std::vector<uint8_t> data;
};

std::vector<Image> read_cifar10(std::istream& in)
{
  std::vector<Image> images;
  while (true)
  {
    Image image;
    if (!in.read(reinterpret_cast<char*>(&image.label), 1))
      break;
    image.data.resize(3072);
    if (!in.read(reinterpret_cast<char*>(image.data.data()), 3072))
      break;
    images.push_back(std::move(image));
  }
  return images;
}
