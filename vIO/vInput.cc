#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "yarp/os/all.h"
#include "iCub/eventdriven/all.h"


#define NUM_COLUMNS 5

using namespace tensorflow;
using namespace ev;



REGISTER_OP("VInput")
    .Output("events: int32")
    .Attr("portname: string = '/tensorflow:i'")
    .Attr("strict: bool = True")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->Matrix(c->UnknownDim(), NUM_COLUMNS));
      c->set_output(1, c->Scalar());
      return Status::OK();
    })
    .Doc(R"doc(
Reads vBottles from yarp and provides them as a tensorflow tensor.
)doc");


class VInputOp : public OpKernel {
 private:
    yarp::os::Network yarp;
    yarp::os::BufferedPort<ev::vBottle> p;
    string portname_;
    bool strict_;
 public:
  explicit VInputOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("portname", &portname_));
    OP_REQUIRES_OK(context, context->GetAttr("strict", &strict_));

    if (!yarp.initialized()) {
      yarp.init();
    }
    p.setStrict(strict_);
    p.open(portname_);
  }

  void Compute(OpKernelContext* context) override {
    // Create an output tensor

    ev::vBottle *bot = p.read();
    yarp::os::Stamp s;
    p.getEnvelope(s);

    vQueue vq = bot->get<AE>();
    if(!vq.size()) {
        return;
    }
    
    Tensor* events = NULL;
    Tensor* timestamp = NULL;
    tensorflow::TensorShape sh = tensorflow::TensorShape({(long long) vq.size(),5});
    OP_REQUIRES_OK(context, context->allocate_output(0, sh,
                                                     &events));
    OP_REQUIRES_OK(context, context->allocate_output(1, tensorflow::TensorShape(),
                                                     &timestamp));
    auto m = events->matrix<int32>();

    for(int i=0; i<vq.size(); i++)// ev::vQueue::iterator qi = vq.begin(); qi != vq.end(); qi++)
    {
    
        auto v = ev::is_event<AddressEvent>(vq[i]);
        m(i,0) = v->channel;
        m(i,1) = v->stamp;
        m(i,2) = v->x;
        m(i,3) = v->y;
        m(i,4) = v->polarity;
    }

    auto unpacked_stamp = timestamp->scalar<double>();
    unpacked_stamp.setConstant(s.getTime());
    

  }
};

REGISTER_KERNEL_BUILDER(Name("VInput").Device(DEVICE_CPU), VInputOp);

