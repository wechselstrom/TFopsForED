#ifndef KERNEL_SPARSECONV_H_
#define KERNEL_SPARSECONV_H_

template <typename Device, typename T>
struct SparseConvFunctor {
  void operator()(const Device& d, int num_pairs, int num_k_1, int num_k_2, int num_k_3,
                  int num_channels, int num_channels_new,
                  const T* idcs, const T* vals, const T* ps, const T* ws, const T* ovals);
};

#endif KERNEL_SPARSECONV_H_
