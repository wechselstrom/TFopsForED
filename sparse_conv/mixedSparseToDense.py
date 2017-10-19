
import tensorflow as tf
from tensorflow.python.framework import ops

sparseToDense_module = tf.load_op_library('build/libmixedSparseToDense.so')
mixed_sparse_to_dense = sparseToDense_module.mixed_sparse_to_dense


@ops.RegisterGradient("MixedSparseToDense")
def _mixed_sparse_to_dense_grad(op, grad):
    indices = op.inputs[0]; values = op.inputs[1]
    dense_grad = grad
    values_grad = sparseToDense_module.mixed_sparse_to_dense_grad(indices,
            values, dense_grad)
    return [None, values_grad, None]
