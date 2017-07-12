import tensorflow as tf
import numpy as np
import preprocessing

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

    def __init__(self, d_feature, d_output, input_size):
        # structure parameters
        self.input_size = input_size
        self.d_matrix = d_output + 1 
        self.d_feature = d_feature
        self.d_output = d_output
        self._special_node_loc = 1 #int(np.floor(self.input_size / 2))

    def prepare(self, feature, label):
        accuracy = self._lin_reg(feature, label)
        self._setup_nodes()

        return accuracy

    def test(self, test_feature, test_label):
        '''
        feature must be numpy array of dtype float32
        '''
        feature = tf.placeholder(tf.float32, shape=[self.input_size, None, self.d_feature])
        label = tf.placeholder(tf.float32, shape=[None, self.d_output])
        accuracy = self.prepare(feature, label)

        f = self.predict(feature)
        f = tf.Print(f, [f], summarize=190, message="prediction")
        cost = self.cost(f, label)
        new_accuracy = self.accuracy(f, label)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            reg_acc, test_cost, test_acc = sess.run([accuracy, cost, new_accuracy], {feature: test_feature, label: test_label})
            print(reg_acc)
            print(test_cost)
            print(test_acc)

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
            cost = 0.5 * tf.reduce_sum(tf.square(f-label))
        return cost

    def accuracy(self, f, label):
        with tf.name_scope("accuracy"):
            prediction = tf.argmax(f, axis=1)
            true_value = tf.argmax(label, axis=1)
            correct_prediction = tf.equal(prediction, true_value)
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return accuracy

    def _lin_reg(self, feature, label):
        self.weight = tf.Variable(tf.zeros([self.input_size, self.d_output]))
        self.bias = tf.Variable(tf.zeros([self.d_output]))

        x = tf.transpose(feature[:, :, 1])

        prediction = tf.matmul(x, self.weight) + self.bias
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=prediction)
        train_step = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy)

        correct_prediction = tf.equal(tf.argmax(label,1), tf.argmax(prediction,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        return accuracy


    def _setup_nodes(self):

        with tf.name_scope("MPSnodes"):
            self.nodes_list = []
            self.nodes = tf.TensorArray(tf.float32, size=0, dynamic_size=True,
                                        clear_after_read=False, infer_shape=False)
            # First node
            self.nodes_list.append(tf.placeholder_with_default(self._make_start_vector(), [self.d_feature, None]))
            self.nodes = self.nodes.write(0, self.nodes_list[-1])

            for i in range(1, self.input_size - 1):
                if i == self._special_node_loc:
                    # The Second node with output leg attached
                    self.nodes_list.append(tf.placeholder_with_default(self._make_special_node(i),
                                                                       [self.d_output, self.d_feature, None, None]))
                    self.nodes = self.nodes.write(i, self.nodes_list[-1])
                else:
                    self.nodes_list.append(tf.placeholder_with_default(self._make_middle_node(i),
                                                                       [self.d_feature, None, None]))
                    self.nodes = self.nodes.write(i, self.nodes_list[-1])

            # Last node
            self.nodes_list.append(
                tf.placeholder_with_default(self._make_end_vector(), [self.d_feature, None]))
            self.nodes = self.nodes.write(self.input_size - 1, self.nodes_list[-1])

    def _make_start_vector(self):
        vector1 = np.ones([self.d_matrix], dtype=np.float32)
        vector1[0] = 0
        vector2 = np.zeros([self.d_matrix], dtype=np.float32)
        v1_t = tf.Variable(vector1)
        v2_t = tf.Variable(vector2)

        return tf.stack([v1_t, v2_t])

    def _make_end_vector(self):
        vector1 = tf.constant(1, dtype=tf.float32)
        b = tf.unstack(self.bias)
        vector1 = tf.stack([vector1] + b)
        vector2 = tf.constant(0, dtype=tf.float32)
        w = tf.unstack(self.weight[-1])
        vector2 = tf.stack([vector2] + w)
        return tf.stack([vector1, vector2])

    def _make_middle_node(self, index):
        identity = tf.eye(self.d_matrix)
        matrix2 = tf.Variable(np.zeros([self.d_matrix, self.d_matrix], dtype=np.float32))
        w = self.weight[index]
        indices = [[i, 0] for i in range(1, self.d_matrix)]
        matrix2 = tf.scatter_nd_add(matrix2, indices , w)
        stacked = [identity, matrix2]
        return tf.stack(stacked)

    def _make_special_node(self, index):
        
        def _make_matrix(i):
            m1 = np.zeros([self.d_matrix, self.d_matrix], dtype=np.float32)
            m1[0, 0] = 1 
            m1[i+1, i+1] = 1 
            m1 = tf.Variable(m1)
            m2 = tf.Variable(np.zeros([self.d_matrix, self.d_matrix], dtype=np.float32))
            m2 = tf.scatter_nd_add(m2, [[i+1, 0]], tf.reshape(self.weight[index, i], [1]))

            return tf.stack([m1, m2])

        stacked = [_make_matrix(i) for i in range(self.d_output)]

        return tf.stack(stacked)

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
    d_output = 10
    batch_size = 1000

    data_source = preprocessing.MNISTData()
    (batch_feature, batch_label) = data_source.next_training_data_batch(batch_size)
    
    # Initialise the model
    network = MPS(d_feature, d_output, input_size)
    network.test(batch_feature, batch_label)
