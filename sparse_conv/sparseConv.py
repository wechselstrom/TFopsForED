
import tensorflow as tf
from tensorflow.python.framework import ops

sparseConv_module = tf.load_op_library('build/libsparseConv.so')
sparse_conv = sparseConv_module.sparse_conv
sparse_conv_grad = sparseConv_module.sparse_conv_grad

@ops.RegisterGradient("SparseConv")
def _sparse_conv_grad(op, grad):
    indices = op.inputs[0]; values = op.inputs[1]
    pairs = op.inputs[2]; kernel = op.inputs[3]
    values_grad, kernel_grad = sparse_conv_grad(
            indices, values, pairs, kernel, grad, name=None)
    return [None, values_grad, None, kernel_grad]


