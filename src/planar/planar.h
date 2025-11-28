template <typename T>
void packed_to_planar(T* src, T* r, T* g, T* b, int n)
{
  for (int i = 0; i < n; i++)
  {
    r[i] = src[3 * i];
    g[i] = src[3 * i + 1];
    b[i] = src[3 * i + 2];
  }
}

template <typename T>
void planar_to_packed(T* dst, T* r, T* g, T* b, int n)
{
  for (int i = 0; i < n; i++)
  {
    dst[3 * i] = r[i];
    dst[3 * i + 1] = g[i];
    dst[3 * i + 2] = b[i];
  }
}
