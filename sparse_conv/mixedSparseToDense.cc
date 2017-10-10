#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"


using namespace tensorflow;


using CPUDevice = Eigen::ThreadPoolDevice;


REGISTER_OP("MixedSparseToDense")
    .Input("indices_in: int64")
    .Input("values_in: float32")
    .Input("shape: int64")
    .Output("dense: float32")
    //.Attr("threshold: int64")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {

      tensorflow::shape_inference::ShapeHandle out;
      TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(2, &out));
      c->set_output(0, out);
      return Status::OK();
    })
    .Doc(R"doc(
         Mixed Sparse To Dense operation.)doc");

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


       
       auto idcs = inp_indices->matrix<int64>();
       auto vals = inp_values->matrix<float>();
       auto shape_vec = shape->vec<int64>();
       auto dsh = {shape_vec(0), shape_vec(1), shape_vec(2), shape_vec(3), shape_vec(4)};
       auto out = dense->shaped<float, 5>(dsh);

       for (int i=0; i<num_active; i++) {
           for (int j=0; j<num_channels; j++) {
                   out(idcs(i,0), idcs(i,1), idcs(i,2), idcs(i,3), j) = vals(i,j);
           }
       }

       ctx->set_output(0, *dense);

    }
};

REGISTER_KERNEL_BUILDER(Name("MixedSparseToDense").Device(DEVICE_CPU), SparseConvOp);
