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
        self._special_node_loc = int(np.floor(self.input_size / 2))

        # Initialise the nodes
        # self._setup_nodes()

    def test(self, test_feature, test_label):
        '''
        feature must be numpy array of dtype float32
        '''
        feature = tf.placeholder(tf.float32, shape=[self.input_size, None, self.d_feature])
        label = tf.placeholder(tf.float32, shape=[None, self.d_output])
        self._setup_nodes(feature)
        f = self.predict(feature)
        f = tf.Print(f, [f], summarize=190, message="prediction")
        cost = self.cost(f, label)
        accuracy = self.accuracy(f, label)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            test_cost, test_accuracy = sess.run([cost, accuracy], {feature: test_feature, label: test_label})
            print(test_cost)
            print(test_accuracy)

    def load_nodes(self, weights):
        updated_nodes = tf.TensorArray(tf.float32, size=0, dynamic_size=True,
                                       clear_after_read=False, infer_shape=False)
        with tf.name_scope("newMPSNodes"):
            for index, weight in enumerate(weights):
                updated_nodes = updated_nodes.write(index, tf.Variable(weight))
        self.nodes = updated_nodes

    # ================
    # hidden functions
    # ================

    def cost(self, f, label):
        with tf.name_scope("cost"):
            cost = tf.einsum('tl,tl->', f - label, f - label)
        return 0.5 * cost

    def accuracy(self, f, label):
        with tf.name_scope("accuracy"):
            prediction = tf.argmax(f, axis=1)
            true_value = tf.argmax(label, axis=1)
            correct_prediction = tf.equal(prediction, true_value)
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return accuracy

    def _setup_nodes(self, feature):
        self.averaged_feature = tf.reduce_mean(feature, axis=1)
        with tf.name_scope("MPSnodes"):
            self.nodes_list = []
            self.nodes = tf.TensorArray(tf.float32, size=0, dynamic_size=True,
                                        clear_after_read=False, infer_shape=False)
            # First node
            self.nodes_list.append(tf.placeholder_with_default(self._make_end_vector(0), [self.d_feature, None]))
            self.nodes = self.nodes.write(0, self.nodes_list[-1])

            for i in range(1, self.input_size - 1):
                if i == self._special_node_loc:
                    # The Second node with output leg attached
                    self.nodes_list.append(tf.placeholder_with_default(self._make_special_node(),
                                                                       [self.d_output, self.d_feature, None, None]))
                    self.nodes = self.nodes.write(i, self.nodes_list[-1])
                else:
                    self.nodes_list.append(tf.placeholder_with_default(self._make_middle_node(i),
                                                                       [self.d_feature, None, None]))
                    self.nodes = self.nodes.write(i, self.nodes_list[-1])

            # Last node
            self.nodes_list.append(
                tf.placeholder_with_default(self._make_end_vector(self.input_size - 1), [self.d_feature, None]))
            self.nodes = self.nodes.write(self.input_size - 1, self.nodes_list[-1])

    def _make_end_vector(self, index):
        vector = tf.ones(self.d_matrix)
        return tf.stack([vector] * self.d_feature)

    def _make_middle_node(self, index):
        cos = self.averaged_feature[index, 0]
        sin = self.averaged_feature[index, 1]
        identity = tf.eye(self.d_matrix)
        return tf.stack([identity * cos, identity * sin])

    def _make_special_node(self):
        cos = self.averaged_feature[self._special_node_loc, 0]
        sin = self.averaged_feature[self._special_node_loc, 1]
        identity = tf.eye(self.d_matrix)
        stacked = tf.stack([identity * cos, identity * sin])
        stacked_identity = tf.stack([identity] * 2)
        return tf.stack([stacked] + [stacked_identity] * (self.d_output - 1))

    def predict_old(self, feature):
        self.feature = feature

        # Read in the nodes 
        node1 = self.nodes.read(0)
        node1.set_shape([self.d_feature, None])
        node2 = self.nodes.read(1)
        # node1 = tf.Print(node1, [node1], message = "mpspredictnode1")
        node2.set_shape([self.d_output, self.d_feature, None, None])
        nodelast = self.nodes.read(self.input_size - 1)
        nodelast.set_shape([self.d_feature, None])

        # Calculate C1 
        C1 = tf.einsum('ni,tn->ti', node1, feature[0])
        contracted_node2 = tf.einsum('lnij,tn->tlij', node2, feature[1])
        C1 = tf.einsum('ti,tlij->tlj', C1, contracted_node2)

        # Calculate C2
        C2 = tf.einsum('mi,tm->ti', nodelast, feature[self.input_size - 1])

        # counter = tf.Variable(2, dtype=tf.int32)
        counter = 2
        cond = lambda c, b: tf.less(c, self.input_size - 1)
        _, C1 = tf.while_loop(cond=cond, body=self._chain_multiply, loop_vars=[counter, C1],
                              shape_invariants=[tf.TensorShape([]), tf.TensorShape([None, self.d_output, None])])
        f = tf.einsum('tli,ti->tl', C1, C2)
        return f

    def _chain_multiply(self, counter, C1):
        with tf.name_scope("chain_multiply"):
            node = self.nodes.read(counter)
            node.set_shape([self.d_feature, None, None])
            input_leg = self.feature[counter]
            contracted_node = tf.einsum('mij,tm->tij', node, input_leg)
            C1 = tf.einsum('tli,tij->tlj', C1, contracted_node)
            counter = counter + 1
        return [counter, C1]

    def _chain_multiply_l(self, counter, C1):
        with tf.name_scope("chain_multiply_l"):
            node = self.nodes.read(counter)
            node.set_shape([self.d_feature, None, None])
            input_leg = self.feature[counter]
            contracted_node = tf.einsum('mij,tm->tij', node, input_leg)
            C1 = tf.einsum('ti,tij->tj', C1, contracted_node)
            counter = counter + 1
        return [counter, C1]

    def predict(self, feature):
        with tf.name_scope("MPS_predict"):
            # Read in feature 
            self.feature = feature

            # Read in the nodes 
            node1 = self.nodes.read(0)
            node1.set_shape([self.d_feature, None])
            sp_node = self.nodes.read(self._special_node_loc)
            sp_node.set_shape([self.d_output, self.d_feature, None, None])
            nodelast = self.nodes.read(self.input_size - 1)
            nodelast.set_shape([self.d_feature, None])

            # Calculate C1 
            C1 = tf.einsum('ni,tn->ti', node1, feature[0])
            cond = lambda c, b: tf.less(c, self._special_node_loc)

            counter, C1 = tf.while_loop(cond=cond, body=self._chain_multiply_l, loop_vars=[1, C1],
                                        shape_invariants=[tf.TensorShape([]), tf.TensorShape([None, None])],
                                        parallel_iterations=1)
            contracted_node2 = tf.einsum('lnij,tn->tlij', sp_node, feature[self._special_node_loc])

            C1 = tf.einsum('ti,tlij->tlj', C1, contracted_node2)

            cond2 = lambda c, b: tf.less(c, self.input_size - 1)
            _, C1 = tf.while_loop(cond=cond2, body=self._chain_multiply, loop_vars=[counter + 1, C1],
                                  shape_invariants=[tf.TensorShape([]), tf.TensorShape([None, self.d_output, None])],
                                  parallel_iterations=1)

            C2 = tf.einsum('mi,tm->ti', nodelast, feature[self.input_size - 1])
            f = tf.einsum('tli,ti->tl', C1, C2)
        return f

    def create_feed_dict(self, weights):
        feed_dict = {}
        if weights != None:
            for index, element in enumerate(weights):
                feed_dict[self.nodes_list[index]] = element
        return feed_dict


if __name__ == '__main__':
    # Model parameters
    input_size = 196
    d_feature = 2
    d_matrix = 5
    d_output = 6
    rate_of_change = 100
    batch_size = 1000
    m = 5

    # Make up input and output
    phi = np.random.normal(size=(input_size, batch_size, d_feature)).astype(np.float32)

    delta = np.zeros((batch_size, d_output))
    ind = int(batch_size / 2)
    delta[:ind, 1] = 1
    delta[ind:, 4] = 1

    # Initialise the model
    network = MPS(d_matrix, d_feature, d_output, input_size, init_param)
    network.test(phi, delta)
