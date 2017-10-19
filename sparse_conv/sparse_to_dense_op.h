#ifndef KERNEL_SPARSETODENSE_H_
#define KERNEL_SPARSETODENSE_H_

template <typename Device, typename T, typename TIDX>
struct SparseToDenseFunctor {
  void operator()(const Device& d, int num_active, int num_channels,
                  const T* idcs, const T* vals, const T* out);
};

template <typename Device, typename T, typename TIDX>
struct SparseToDenseGradFunctor {
  void operator()(const Device& d, int num_active, int num_channels
                  //const T* idcs, const T* dense_grad, const T* vals_grad_out
                  );
};

#endif KERNEL_SPARSETODENSE_H_
