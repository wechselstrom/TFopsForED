import tensorflow as tf
import numpy as np

class VInputTest(tf.test.TestCase):
  def testVInput(self):
    vInput_module = tf.load_op_library('build/libvInput.so')
    with self.test_session():
      result = vInput_module.v_input()
      while True:
        print(str(result.eval()))
      #self.assertAllEqual(result.eval(), np.zeros((10,5),dtype = np.uint32)+25)

if __name__ == "__main__":
  tf.test.main()

