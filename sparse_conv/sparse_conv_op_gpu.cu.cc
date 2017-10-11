
#ifdef GOOGLE_CUDA

#define EIGEN_USE_GPU
#define EIGEN_USE_THREADS

#include "sparse_conv_op.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"

using namespace tensorflow;

#define EIGEN_USE_GPU

// Define the CUDA kernel.
template <typename T>
__global__ void SparseConvCudaKernel(const int size, const T* in, T* out) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size;
       i += blockDim.x * gridDim.x) {
    out[i] = 2 * ldg(in + i);
  }
}

// Define the GPU implementation that launches the CUDA kernel.
template <typename T>
struct SparseConvFunctor<GPUDevice, T> {
  void operator()(OpKernelContext* context, int num_pairs, int num_k_1, int num_k_2, int num_k_3,
                  int num_channels, int num_channels_new,
                  typename TTypes<const int64>::ConstMatrix &idcs,
                  typename TTypes<T>::ConstMatrix &vals,
                  typename TTypes<const int64>::ConstMatrix &ps,
                  typename TTypes<T,5>::ConstTensor &ws,
                  typename TTypes<T>::Matrix &ovals) {
    // Launch the cuda kernel.
    //
    // See core/util/cuda_kernel_helper.h for example of computing
    // block count and thread_per_block count.
    int block_count = 1024;
    int thread_per_block = 20;
    SparseConvCudaKernel<T>
        <<<block_count, thread_per_block, 0, d.stream()>>>(size, in, out);
  }
};

// Instantiate functors for the types of OpKernels registered.
typedef Eigen::GpuDevice GPUDevice;
template struct SparseConvFunctor<GPUDevice, float>;
template struct SparseConvFunctor<GPUDevice, int32>;

#endif  // GOOGLE_CUDA
