#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#define NUM_COLUMNS 5

using namespace tensorflow;


using CPUDevice = Eigen::ThreadPoolDevice;


REGISTER_OP("SparseConv")
    .Input("indices_in: int64")
    .Input("values_in: float32")
    .Input("pairs: int64")
    .Input("kernel: float32")
    .Output("values_out: float32")
    //.Attr("threshold: int64")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {

        //new values.shape = [#values.shape[0] x #new_channels]
       c->set_output(0, c->Matrix(c->Dim(c->input(1),0), c->Dim(c->input(3),4)));        


        return Status::OK();
    })
    .Doc(R"doc(
         Sparse convolution operation.)doc");

REGISTER_OP("SparseConvGrad")
    .Input("indices_in: int64")
    .Input("values_in: float32")
    .Input("pairs: int64")
    .Input("kernel: float32")
    .Input("values_grad: float32")
    .Output("values_out_grad: float32")
    .Output("kernel_out_grad: float32")
    //.Attr("threshold: int64")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {

        //new values.shape = [#values.shape[0] x #new_channels]
       c->set_output(0, c->input(1));        
       c->set_output(1, c->input(3));        


        return Status::OK();
    })
    .Doc(R"doc(
         Sparse convolution operation.)doc");

// end of registering, start of classes
    
class SparseConvOp : public OpKernel {
private:
    bool strict_;

public:
    explicit SparseConvOp(OpKernelConstruction* context)
        : OpKernel(context)
    {
    }

    void Compute(OpKernelContext* ctx) override
    {
        // Create an output tensor

        const Tensor *inp_indices, *inp_values, *pairs, *kernel;

        OP_REQUIRES_OK(ctx, ctx->input("indices_in", &inp_indices));
        OP_REQUIRES_OK(ctx, ctx->input("values_in", &inp_values));
        OP_REQUIRES_OK(ctx, ctx->input("pairs", &pairs));
        OP_REQUIRES_OK(ctx, ctx->input("kernel", &kernel));

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
        const int64 num_channels_new = kernel->dim_size(4);
        const int64 num_k_1 = kernel->dim_size(0);
        const int64 num_k_2 = kernel->dim_size(1);
        const int64 num_k_3 = kernel->dim_size(2);
        
        const int64 num_pairs = pairs->dim_size(0);

        OP_REQUIRES(
            ctx, inp_values->dim_size(0) == num_active,
            errors::InvalidArgument("Expected ", num_active,
                " non-empty input values, got ",
               inp_values->dim_size(0)));
        
       OP_REQUIRES_OK(ctx, ctx->input("kernel", &kernel));
       kernel->dims();

       OP_REQUIRES(
            ctx, kernel->dims() == 5,
            errors::InvalidArgument("Expected rank of ", 5,
                " for the kernel, instead got rank of ",
               kernel->dims()));
       
       Tensor *out_values;

       
       OP_REQUIRES_OK(ctx,
           ctx->allocate_output(0, TensorShape({ num_active, num_channels_new }),
               &out_values));
       auto ps = pairs->matrix<int64>();
       auto idcs = inp_indices->matrix<int64>();
       auto ws = kernel->tensor<float, 5>();
       auto vals = inp_values->matrix<float>();
       auto ovals = out_values->matrix<float>();

       for (int i=0; i<num_pairs; i++) {
           //idcs(ps(i,*), 0) is batch, thus we ignore it.
           
           int x1 = ps(i,0);
           int x2 = ps(i,1);
           #define COMPUTE_STEP(a, b) { \
               long d1 = idcs(b, 1) + (int) num_k_1/2 - idcs(a, 1) ;\
               long d2 = idcs(b, 2) + (int) num_k_2/2 - idcs(a, 2) ;\
               long d3 = idcs(b, 3) + (int) num_k_3/2 - idcs(a, 3) ;\
               for (int j=0; j<num_channels_new; j++) {\
                   for (int k=0; k<num_channels; k++) {\
                       ovals(a, j) += vals(b, k)*ws(d1, d2, d3, k, j);\
                   }\
               }\
           }
           COMPUTE_STEP(x1, x2);
           if (x1!=x2) {
               COMPUTE_STEP(x2, x1);
           }
       }
       
       
       ctx->set_output(0, *out_values);

    }
};

REGISTER_KERNEL_BUILDER(Name("SparseConv").Device(DEVICE_CPU), SparseConvOp);


class SparseConvGradOp : public OpKernel {
private:
    bool strict_;

public:
    explicit SparseConvGradOp(OpKernelConstruction* context)
        : OpKernel(context)
    {
    }

    void Compute(OpKernelContext* ctx) override
    {
        // Create an output tensor

        const Tensor *inp_indices, *inp_values, *pairs, *kernel, *values_grad;

        OP_REQUIRES_OK(ctx, ctx->input("indices_in", &inp_indices));
        OP_REQUIRES_OK(ctx, ctx->input("values_in", &inp_values));
        OP_REQUIRES_OK(ctx, ctx->input("pairs", &pairs));
        OP_REQUIRES_OK(ctx, ctx->input("kernel", &kernel));
        OP_REQUIRES_OK(ctx, ctx->input("values_grad", &values_grad));

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
        const int64 num_channels_new = kernel->dim_size(4);
        const int64 num_k_1 = kernel->dim_size(0);
        const int64 num_k_2 = kernel->dim_size(1);
        const int64 num_k_3 = kernel->dim_size(2);
        
        const int64 num_pairs = pairs->dim_size(0);

        OP_REQUIRES(
            ctx, inp_values->dim_size(0) == num_active,
            errors::InvalidArgument("Expected ", num_active,
                " non-empty input values, got ",
               inp_values->dim_size(0)));
        
       OP_REQUIRES_OK(ctx, ctx->input("kernel", &kernel));
       kernel->dims();

       OP_REQUIRES(
            ctx, kernel->dims() == 5,
            errors::InvalidArgument("Expected rank of ", 5,
                " for the kernel, instead got rank of ",
               kernel->dims()));
       
       Tensor *values_grad_out, *kernel_grad_out;

       
       OP_REQUIRES_OK(ctx,
           ctx->allocate_output(0, inp_values->shape(),
               &values_grad_out));
       OP_REQUIRES_OK(ctx,
           ctx->allocate_output(1, kernel->shape(),
               &kernel_grad_out));
       auto ps = pairs->matrix<int64>();
       auto idcs = inp_indices->matrix<int64>();
       auto ws = kernel->tensor<float, 5>();
       auto vals = inp_values->matrix<float>();
       auto grad = values_grad->matrix<float>();
       auto vals_grad_out = values_grad_out->matrix<float>();
       auto ws_grad_out = kernel_grad_out->tensor<float, 5>();

       for (int i=0; i<num_pairs; i++) {
           //idcs(ps(i,*), 0) is batch, thus we ignore it.
           
           int x1 = ps(i,0);
           int x2 = ps(i,1);
           #define GRAD_COMPUTE_STEP(a, b) { \
               long d1 = idcs(b, 1) + (int) num_k_1/2 - idcs(a, 1) ;\
               long d2 = idcs(b, 2) + (int) num_k_2/2 - idcs(a, 2) ;\
               long d3 = idcs(b, 3) + (int) num_k_3/2 - idcs(a, 3) ;\
               for (int j=0; j<num_channels_new; j++) {\
                   for (int k=0; k<num_channels; k++) {\
                       ws_grad_out(d1, d2, d3, k, j) += vals(b, k)*grad(a, j);\
                       vals_grad_out(b, k) += ws(d1, d2, d3, k, j)*grad(a, j);\
                   }\
               }\
           }
           GRAD_COMPUTE_STEP(x1, x2);
           if (x1!=x2) {
               GRAD_COMPUTE_STEP(x2, x1);
           }
       }
       
       
       ctx->set_output(0, *values_grad_out);
       ctx->set_output(1, *kernel_grad_out);

    }
};

REGISTER_KERNEL_BUILDER(Name("SparseConvGrad").Device(DEVICE_CPU), SparseConvGradOp);
