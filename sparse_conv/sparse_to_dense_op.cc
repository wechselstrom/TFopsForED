#define EIGEN_USE_THREADS
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/util/work_sharder.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "sparse_to_dense_op.h"


using namespace tensorflow;



using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

REGISTER_OP("MixedSparseToDense")
    .Attr("T: {float, double, int32, int64}")
    .Attr("TIDX: {int32, int64}")
    .Input("indices_in: TIDX")
    .Input("values_in: T")
    .Input("shape: TIDX")
    .Output("dense: T")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {

      tensorflow::shape_inference::ShapeHandle out;
      TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(2, &out));
      c->set_output(0, out);
      return Status::OK();
    })
    .Doc(R"doc(
         Mixed Sparse To Dense operation.)doc");

REGISTER_OP("MixedSparseToDenseGrad")
    .Attr("T: {float, double}")
    .Attr("TIDX: {int32, int64}")
    .Input("indices_in: TIDX")
    .Input("values_in: T")
    .Input("dense_grad: T")
    .Output("values_out_grad: T")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {

       c->set_output(0, c->input(1));        
      return Status::OK();
    })
    .Doc(R"doc(
         Mixed Sparse To Dense operation gradient.)doc");


template <typename T, typename TIDX>
struct SparseToDenseFunctor<CPUDevice, T, TIDX> {

  void operator()(OpKernelContext* context, int num_active, int num_channels,
                  typename TTypes<const TIDX>::ConstMatrix &idcs,
                  typename TTypes<const T>::ConstMatrix &vals,
                  typename TTypes<T,5>::Tensor &out) {
      //num_pairs
       auto work = [&](int start, int limit) {
           for (int i=start; i<limit; i++) {
               for (int j=0; j<num_channels; j++) {
                       out(idcs(i,0), idcs(i,1), idcs(i,2), idcs(i,3), j) = vals(i,j);
               }
           }
       };
       
       //auto threadpool = tensorflow::thread::ThreadPool();
       const int default_cost = 100;
       const DeviceBase::CpuWorkerThreads& worker_threads =
            *(context->device()->tensorflow_cpu_worker_threads());
       Shard(worker_threads.num_threads, worker_threads.workers,
               num_active, num_channels*default_cost, work);
  }
};

template <typename T, typename TIDX>
struct SparseToDenseGradFunctor<CPUDevice, T, TIDX> {

  void operator()(OpKernelContext* context, int num_active, int num_channels,
                  typename TTypes<const TIDX>::ConstMatrix &idcs,
                  typename TTypes<const T,5>::ConstTensor &dense_grad,
                  typename TTypes<T>::Matrix &vals_grad_out
                  ) {
      //num_pairs
       auto work = [&](int start, int limit) {
           for (int i=start; i<limit; i++) {
               for (int j=0; j<num_channels; j++) {
                       vals_grad_out(i,j) = dense_grad(idcs(i,0), idcs(i,1), idcs(i,2), idcs(i,3), j);
               }
           }
       };
       
       //auto threadpool = tensorflow::thread::ThreadPool();
       const int default_cost = 100;
       const DeviceBase::CpuWorkerThreads& worker_threads =
            *(context->device()->tensorflow_cpu_worker_threads());
       Shard(worker_threads.num_threads, worker_threads.workers,
               num_active, num_channels*default_cost, work);
  }
};




template<typename Device, typename T, typename TIDX>
class MixedSparseToDenseOp : public OpKernel {
private:

public:
    explicit MixedSparseToDenseOp(OpKernelConstruction* context)
        : OpKernel(context)
    {
    }

    void Compute(OpKernelContext* ctx) override
    {
        // Create an output tensor

        const Tensor *inp_indices, *inp_values, *shape;

        OP_REQUIRES_OK(ctx, ctx->input("indices_in", &inp_indices));
        OP_REQUIRES_OK(ctx, ctx->input("values_in", &inp_values));
        OP_REQUIRES_OK(ctx, ctx->input("shape", &shape));

        
        OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(inp_indices->shape()),
            errors::InvalidArgument(
                        "Input indices should be a matrix but received shape: ",
                        inp_indices->shape().DebugString()));

        const int64 num_active = inp_indices->dim_size(0);
        const int64 num_dimensions = inp_indices->dim_size(1);

        OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(inp_values->shape()),
            errors::InvalidArgument(
                        "This operation is special in that the value tensor is NOT supposed to be a vector,\n\
                    but a matrix (num_indices x num_channels).\n\
                    Input values should be a matrix but received shape: ",
                        inp_values->shape().DebugString()));

        const int64 num_channels = inp_values->dim_size(1);
        

        OP_REQUIRES(ctx, shape->dim_size(0) == inp_indices->dim_size(1) + inp_values->dims()-1,
            errors::InvalidArgument("Expected that inp_shape.shape[0] = inp_indices.shape[1] + len(inp_values.shape)-1\n",
                "however the received values were: ", shape->dim_size(0), " != ", inp_indices->dim_size(1), " + ", inp_values->dims(), "-1",
                shape->shape().DebugString()));
        
        auto output_shape_vec = shape->flat<int64>();
        TensorShape output_tensor_shape;

        OP_REQUIRES_OK(ctx, TensorShapeUtils::MakeShape(output_shape_vec.data(),
                                                      output_shape_vec.size(),
                                                      &output_tensor_shape));
        Tensor* dense = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_tensor_shape, &dense));


       
       auto idcs = inp_indices->matrix<TIDX>();
       auto vals = inp_values->matrix<T>();
       auto shape_vec = shape->vec<TIDX>();
       auto dsh = {(int64) shape_vec(0), (int64) shape_vec(1), (int64)
                   (int64) shape_vec(2), (int64) shape_vec(3), (int64) shape_vec(4)};
       auto out = dense->shaped<T, 5>(dsh);

       SparseToDenseFunctor<Device, T, TIDX>()(ctx, num_active, num_channels,
                  idcs, vals, out
               );

       ctx->set_output(0, *dense);

    }
};


template<typename Device, typename T, typename TIDX>
class MixedSparseToDenseGradOp : public OpKernel {
private:

public:
    explicit MixedSparseToDenseGradOp(OpKernelConstruction* context)
        : OpKernel(context)
    {
    }

    void Compute(OpKernelContext* ctx) override
    {
        // Create an output tensor

        const Tensor *inp_indices, *inp_values, *dense_grad;

        OP_REQUIRES_OK(ctx, ctx->input("indices_in", &inp_indices));
        OP_REQUIRES_OK(ctx, ctx->input("values_in", &inp_values));
        OP_REQUIRES_OK(ctx, ctx->input("dense_grad", &dense_grad));

        
        OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(inp_indices->shape()),
            errors::InvalidArgument(
                        "Input indices should be a matrix but received shape: ",
                        inp_indices->shape().DebugString()));

        const int64 num_active = inp_indices->dim_size(0);
        const int64 num_dimensions = inp_indices->dim_size(1);

        OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(inp_values->shape()),
            errors::InvalidArgument(
                        "This operation is special in that the value tensor is NOT supposed to be a vector,\n\
                    but a matrix (num_indices x num_channels).\n\
                    Input values should be a matrix but received shape: ",
                        inp_values->shape().DebugString()));

        const int64 num_channels = inp_values->dim_size(1);
        

        
        Tensor* values_grad_out = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, inp_values->shape(), &values_grad_out));


       
       auto idcs = inp_indices->matrix<TIDX>();
       auto sh = dense_grad->shape().dim_sizes();
       auto d_grad = dense_grad->shaped<T, 5>(sh);
       auto vals_grad_out = values_grad_out->matrix<T>();

       SparseToDenseGradFunctor<Device, T, TIDX>()(ctx, num_active, num_channels,
                  idcs, d_grad, vals_grad_out
               );

       ctx->set_output(0, *values_grad_out);

    }
};

//REGISTER_KERNEL_BUILDER(Name("MixedSparseToDense").Device(DEVICE_CPU), SparseConvOp);
#define REGISTER_CPU(T,TIDX)                                          \
      REGISTER_KERNEL_BUILDER(                                       \
                    Name("MixedSparseToDense").Device(DEVICE_CPU).TypeConstraint<T>("T").TypeConstraint<TIDX>("TIDX"), \
                    MixedSparseToDenseOp<CPUDevice, T, TIDX>);
REGISTER_CPU(float, int32);
REGISTER_CPU(double, int32);
REGISTER_CPU(float, int64);
REGISTER_CPU(double, int64);
REGISTER_CPU(int32, int32);
REGISTER_CPU(int64, int32);
REGISTER_CPU(int32, int64);
REGISTER_CPU(int64, int64);


#define REGISTER_CPU_GRAD(T,TIDX)                                          \
      REGISTER_KERNEL_BUILDER(                                       \
                    Name("MixedSparseToDenseGrad").Device(DEVICE_CPU).TypeConstraint<T>("T").TypeConstraint<TIDX>("TIDX"), \
                    MixedSparseToDenseGradOp<CPUDevice, T, TIDX>);
REGISTER_CPU_GRAD(float, int32);
REGISTER_CPU_GRAD(double, int32);
REGISTER_CPU_GRAD(float, int64);
REGISTER_CPU_GRAD(double, int64);
