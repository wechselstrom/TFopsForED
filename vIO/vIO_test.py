import tensorflow as tf
import numpy as np
import threading

class VIOTest(tf.test.TestCase):
  testEvents = np.array(\
                [[0, 1992282, 288,  38, 1],\
                 [0, 1992284,  80,  38, 1],\
                 [0, 1992291, 273,  53, 1],\
                 [0, 2005996, 121, 107, 1],\
                 [0, 2006115, 103,   4, 1],\
                 [1, 2006134, 180, 124, 1]], dtype=np.int32)
  def testVIO(self):
    vOutput_module = tf.load_op_library('build/libvOutput.so')
    vInput_module = tf.load_op_library('build/libvInput.so')
    mat = tf.constant(self.testEvents)
    send = vOutput_module.v_output(mat)
    receive = vInput_module.v_input()
    def call_op(op, coord, session):
        x = session.run(op)
        print(x)
    with self.test_session() as session:
        coord = tf.train.Coordinator()
        threads = [threading.Thread(target=call_op, args=(send, coord,
                                                          session)) for op in
                   [send, receive]]

        while True:
            val = session.run([send, receive])
            print(val, mat)
        #self.assertAllEqual(result.eval(), np.zeros((10,5),dtype = np.uint32)+25)

if __name__ == "__main__":
  tf.test.main()

