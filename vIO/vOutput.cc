#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "yarp/os/all.h"
#include "iCub/eventdriven/all.h"


#define NUM_COLUMNS 5

	using namespace tensorflow;
	using namespace ev;


	REGISTER_OP("VOutput")
	    .Input("in: int32")
	    .Attr("portname: string = '/tensorflow:o'")
	    .Attr("strict: bool = True")
	    .SetShapeFn(::tensorflow::shape_inference::NoOutputs)
	    .Doc(R"doc(
	Writes tensorflow event tensors to yarp as vBottles.
	)doc");


	class VOutputOp : public OpKernel {
	 private:
	    yarp::os::Network yarp;
	    yarp::os::BufferedPort<ev::vBottle> p;
	    string portname_;
	    bool strict_;
	 public:
	  explicit VOutputOp(OpKernelConstruction* context) : OpKernel(context) {
	    OP_REQUIRES_OK(context, context->GetAttr("portname", &portname_));
	    OP_REQUIRES_OK(context, context->GetAttr("strict", &strict_));
	    if (!yarp.initialized()) {
	      yarp.init();
	    }
	    p.open(portname_);
	  }

	  void Compute(OpKernelContext* context) override {
	    // Create an output tensor

	    ev::vBottle &bot = p.prepare();

	    bot.clear();

	    
	    const Tensor& input_tensor = context->input(0);
	    auto m = input_tensor.matrix<int32>();
	    auto sh = input_tensor.shape();
	    int max = sh.dim_size(0);
	  

	    for(int i=0; i<max; i++)
	    {
	    
		//ev::AddressEvent ev = ev::AddressEvent();
		//auto v = ev::is_event<AE>(ev);
		AddressEvent *ae = new ev::AddressEvent();
		event<AE> v =  event<ev::AE>(ae);
		//auto v = std::shared_ptr<ev::AddressEvent>(&ev);
		v->channel  = m(i,0);
		v->stamp    = m(i,1);
		v->x        = m(i,2);
		v->y        = m(i,3);
		v->polarity = m(i,4);
		bot.addEvent(v);
	    }
	    
	    if(bot.size()) {
		p.write(strict_);
    }
    else
        p.unprepare();

  }
};

REGISTER_KERNEL_BUILDER(Name("VOutput").Device(DEVICE_CPU), VOutputOp);

