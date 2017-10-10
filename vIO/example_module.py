import tensorflow as tf
import numpy as np

capacity=100

class VInputTest(tf.test.TestCase):
    def testVInput(self):
        vOutput_module = tf.load_op_library('build/libvOutput.so')
        vInput_module = tf.load_op_library('build/libvInput.so')
        inp, ts = vInput_module.v_input()
        q = tf.FIFOQueue(capacity=capacity, dtypes=[tf.int32, tf.float64])
        enqueue_op = q.enqueue([inp, ts])
        to_send, ts_out = q.dequeue()
        to_send = tf.concat([to_send[:,:-1], 1-to_send[:,-1:]], axis=1)
        qr = tf.train.QueueRunner(q, [enqueue_op])
        send = vOutput_module.v_output(to_send)

        with self.test_session() as session:
            coord = tf.train.Coordinator()
            enqueue_threads = qr.create_threads(session, coord=coord, start=True)


            try:
                while True:
                    ts_, val_, _ = session.run([ts_out, to_send, send])
                    print(ts_, val_)
            except KeyboardInterrupt:
                coord.request_stop()
                coord.join(enqueue_threads)


            #self.assertAllEqual(result.eval(), np.zeros((10,5),dtype = np.uint32)+25)

if __name__ == "__main__":
    tf.test.main()

