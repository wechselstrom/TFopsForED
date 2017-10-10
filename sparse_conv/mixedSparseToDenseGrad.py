
import tensorflow as tf
from tensorflow.python.framework import ops


@ops.RegisterGradient("MixedSparseToDense")
def _mixed_sparse_to_dense_grad(op, grad):
    val_grad = tf.gather_nd(op.outputs[0], op.inputs[0])
    return [None, val_grad, None]


