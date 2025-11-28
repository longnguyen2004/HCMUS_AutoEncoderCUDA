#include <stb_image.h>

int main(int argc, char const *argv[])
{
    int x, y, channels;
    auto img = stbi_load("image.png", &x, &y, &channels, 0);
    return 0;
}
