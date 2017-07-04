import tensorflow as tf
import numpy as np

'''
1. Store the C matrix computation cumulatively
'''

class MPS(object):

    '''
    Class variables:
        input_size: int
        d_matrix: int
        d_feature: int
        d_output: int
        nodes: tf.TensorArray
        feature: tf.Tensor of shape (intput_size, batch_size, d_output)
    '''

    def __init__(self, d_matrix, d_feature, d_output, input_size):

        # structure parameters
        self.input_size = input_size
        self.d_matrix = d_matrix
        self.d_feature = d_feature
        self.d_output = d_output

        # Initialise the nodes
        self._setup_nodes()

    def test(self, test_feature, test_label):
        '''
        feature must be numpy array of dtype float32
        '''
        feature = tf.placeholder(tf.float32, shape=[self.input_size, None, self.d_feature])
        label = tf.placeholder(tf.float32, shape=[None, self.d_output])
        f = self.predict(feature)
        cost = self.cost(f, label)
        accuracy = self.accuracy(f, label)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            test_cost = sess.run(cost, {feature:test_feature, label:test_label})
            test_accuracy = sess.run(accuracy, {feature:test_feature, label:test_label})
            print(test_cost)
            print(test_accuracy)

    def write_graph(self, logdir = "output"):
        writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
        writer.close()

    # ================
    # hidden functions
    # ================

    def cost(self, f, label):
        cost = tf.einsum('tl,tl->', f-label, f-label)
        return 0.5 * cost

    def accuracy(self, f, label):
        prediction = tf.argmax(f, axis=1)
        true_value = tf.argmax(label, axis=1)
        correct_prediction = tf.equal(prediction, true_value)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return accuracy

    def _setup_nodes(self):

        self.nodes = tf.TensorArray(tf.float32, size = 0, dynamic_size= True,
                                    clear_after_read= False, infer_shape= False)
        # First node
        self.nodes = self.nodes.write(0, self._make_random_normal([self.d_feature, self.d_matrix]))
        # The Second node with output leg attached
        self.nodes = self.nodes.write(1, self._make_random_normal([self.d_output, self.d_feature, self.d_matrix, self.d_matrix]))
        # The rest of the matrix nodes
        for i in range(self.input_size - 3):
            self.nodes = self.nodes.write(i+2, self._make_random_normal([self.d_feature, self.d_matrix, self.d_matrix]))
        # Last node
        self.nodes = self.nodes.write(self.input_size-1, self._make_random_normal([self.d_feature, self.d_matrix]))

    def _make_random_normal(self, shape, mean=0, stddev=1):
        return tf.Variable(tf.random_normal(shape, mean=mean, stddev=stddev))

    def predict(self, feature):

        # Read in feature
        self.feature = feature

        # Read in the nodes
        node1 = self.nodes.read(0)
        node1.set_shape([self.d_feature, None])
        node2 = self.nodes.read(1)
        node1 = tf.Print(node1, [node1], message = "mpspredict")
        node2.set_shape([self.d_output, self.d_feature, None, None])
        nodelast = self.nodes.read(self.input_size-1)
        nodelast.set_shape([self.d_feature, None])

        # Calculate C1
        C1 = tf.einsum('ni,tn->ti', node1, feature[0])
        contracted_node2 = tf.einsum('lnij,tn->tlij', node2, feature[1])
        C1 = tf.einsum('ti,tlij->tlj', C1, contracted_node2)

        # Calculate C2
        C2 = tf.einsum('mi,tm->ti', nodelast, feature[self.input_size-1])

        #counter = tf.Variable(2, dtype=tf.int32)
        counter = 2
        cond = lambda c, b: tf.less(c, self.input_size-1)
        _, C1 = tf.while_loop(cond=cond, body=self._chain_multiply, loop_vars=[counter, C1],
                                        shape_invariants=[tf.TensorShape([]), tf.TensorShape([None, self.d_output, None])])
        f = tf.einsum('tli,ti->tl', C1, C2)
        return f

    def _chain_multiply(self, counter, C1):
        node = self.nodes.read(counter)
        node.set_shape([self.d_feature, None, None])
        input_leg = self.feature[counter]
        contracted_node = tf.einsum('mij,tm->tij', node, input_leg)
        C1 = tf.einsum('tli,tij->tlj', C1, contracted_node)
        counter = counter + 1
        return [counter, C1]

if __name__ == '__main__':

    # Model parameters
    input_size = 4
    d_feature = 2
    d_matrix = 5
    d_output = 6
    rate_of_change = 0.2
    batch_size = 1000
    m = 5

    # Make up input and output
    phi = np.random.normal(size=(input_size, batch_size, d_feature)).astype(np.float32)

    delta = np.zeros((batch_size, d_output))
    delta[:500, 1] = 1
    delta[500:, 4] = 1

    # Initialise the model
    network = MPS(d_matrix, d_feature, d_output, input_size)
    network.test(phi, delta)
