import tensorflow as tf
import numpy as np
import scipy.spatial
import mixedSparseToDense
import sparseConv

f = sparseConv.sparse_conv
g = mixedSparseToDense.mixed_sparse_to_dense

class SparseConvTest(tf.test.TestCase):
  def setup(self):

    def sparseConv(indices, values, pairs, kernel, shape):
        convedVals = f(indices, values, pairs, kernel)
        sh = np.concatenate([shape[:-1], [convedVals.shape[1].value]])
        return g(indices, convedVals, sh)

    def sparseConvByDense(indices, values, pairs, kernel, shape):
        data = g(indices, values, shape)
        mask =  tf.reduce_all(tf.not_equal(data, 0,), axis=-1, keep_dims=True)
        conved =  tf.nn.conv3d(data, kernel, [1,1,1,1,1], 'SAME')
        return conved*tf.cast(mask,tf.float32)

    sh = np.int64([10, 8, 47, 48, 20])
    indices = np.int64(np.random.random((10000, len(sh)-1))*sh[:-1])
    indices = indices[np.lexsort(indices.T[::-1])]
    indices = np.unique(list(map(tuple, indices)),axis=0)
    values = tf.constant(np.float32(np.random.random((indices.shape[0],
        sh[-1]))))
    tree = scipy.spatial.cKDTree(indices*(100,1,1,1))
    pairs = np.array(list(tree.query_pairs(1.0,p=np.inf)), dtype=np.int64)
    pairs = np.concatenate([pairs, np.array(list(zip(*[np.arange(len(indices))]*2)))],
            axis=0)

    W = tf.constant(np.float32(np.random.random([3, 3, 3, sh[-1], 25])))
    self.result = sparseConv(indices, values, pairs, W, sh)
    self.target = sparseConvByDense(indices, values, pairs, W, sh)
    self.oldvalues = values
    self.kernel=W

  def testSparseConv(self):
    self.setup()
    with tf.device('/cpu:0'):
      with self.test_session():
        r1 = self.result.eval()
        r2 = self.target.eval()
        print('Values', np.abs(r1-r2).max())
        self.assertAllClose(r1, r2, rtol=1e-03, atol=1e-03)
  
  def testSparseConvGradValues(self):
    self.setup()
    grad_sparse = tf.gradients(self.result, self.oldvalues)
    grad_dense = tf.gradients(self.target, self.oldvalues)
    with tf.device('/cpu:0'):
      with self.test_session():
        r1 = self.result[0].eval()
        r2 = self.target[0].eval()
        print('ValuesGrad:',np.abs(r1-r2).max())
        self.assertAllClose(r1, r2, rtol=1e-03, atol=1e-03)
  
  def testSparseConvGradKernel(self):
    self.setup()
    grad_sparse = tf.gradients(self.result, self.kernel)
    grad_dense = tf.gradients(self.target, self.kernel)
    with tf.device('/cpu:0'):
      with self.test_session():
        r1 = self.result[0].eval()
        r2 = self.target[0].eval()
        print('KernelGrad:', np.abs(r1-r2).max())
        self.assertAllClose(r1, r2, rtol=1e-03, atol=1e-03)

if __name__ == "__main__":
  tf.test.main()

