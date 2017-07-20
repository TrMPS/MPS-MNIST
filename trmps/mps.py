import tensorflow as tf
import numpy as np
import MNISTpreprocessing

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
        self.d_matrix = d_output * 2
        self.d_feature = d_feature
        self.d_output = d_output
        self._special_node_loc = int(np.floor(self.input_size / 2))

    def prepare(self, data_source):
        self._lin_reg(data_source)
        self._setup_nodes()

    def test(self, test_feature, test_label):
        '''
        feature must be numpy array of dtype float32
        '''
        feature = tf.placeholder(tf.float32, shape=[self.input_size, None, self.d_feature])
        label = tf.placeholder(tf.float32, shape=[None, self.d_output])

        f = self.predict(feature)
        f = tf.Print(f, [f], summarize=190, message="prediction")
        cost = self.cost(f, label)
        accuracy = self.accuracy(f, label)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            test_cost, test_acc = sess.run([cost, accuracy], {feature: test_feature, label: test_label})
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
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=f)) # 0.5 * tf.reduce_sum(tf.square(f-label))
        return cost

    def accuracy(self, f, label):
        with tf.name_scope("accuracy"):
            prediction = tf.argmax(f, axis=1)
            true_value = tf.argmax(label, axis=1)
            correct_prediction = tf.equal(prediction, true_value)
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return accuracy

    def _lin_reg(self, data_source):
        weight = tf.Variable(tf.zeros([self.input_size, self.d_output]))
        bias = tf.Variable(tf.zeros([self.d_output]))

        feature = tf.placeholder(tf.float32, shape=[self.input_size, None, self.d_feature])
        label = tf.placeholder(tf.float32, shape=[None, self.d_output])

        x = tf.transpose(feature[:, :, 1])

        prediction = tf.matmul(x, weight) + bias
        #cross_entropy = 0.5 * tf.reduce_sum(tf.square(prediction-label))
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=prediction))
        train_step = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy)
        #train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)

        correct_prediction = tf.equal(tf.argmax(label,1), tf.argmax(prediction,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for _ in range(1000):    
                batch_feature, batch_label = data_source.next_training_data_batch(100)
                sess.run(train_step, feed_dict={feature: batch_feature, label: batch_label})
            batch_feature, batch_label = data_source.next_training_data_batch(10000)
            acc = accuracy.eval(feed_dict={feature: batch_feature, label: batch_label})
            print('Lin regression gives an accuracy of {}'.format(acc))
            self.weight = sess.run(weight)
            self.bias = sess.run(bias)

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
        start_vector = np.zeros([2, self.d_matrix], dtype=np.float32)
        start_vector[0, self.d_output:] = 1
        start_vector[1, 0:self.d_output] = self.weight[0]
        return tf.Variable(start_vector, dtype=tf.float32)

    def _make_end_vector(self):

        end_vector = np.zeros([2, self.d_matrix], dtype=np.float32)
        end_vector[0, 0:self.d_output] = 1 
        end_vector[0, self.d_output:] = self.bias 
        end_vector[1, self.d_output:] = self.weight[-1] 

        return tf.Variable(end_vector, dtype=tf.float32)

    def _make_middle_node(self, index):

        middle_node = np.zeros([2, self.d_matrix, self.d_matrix], dtype=np.float32)
        middle_node[0] = np.identity(self.d_matrix)
        middle_node[1, self.d_output:, 0:self.d_output] = np.diag(self.weight[index])

        return tf.Variable(middle_node, dtype=tf.float32)

    def _make_special_node(self, index):
        
        def _make_matrix(i):
            m = np.zeros([2, self.d_matrix, self.d_matrix], dtype=np.float32)
            m[0, i, i] = 1 
            m[0, i+self.d_output, i+self.d_output] = 1
            m[1, i+self.d_output, i] = self.weight[index, i]

            return tf.Variable(m)

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
    input_size = 784
    shrink = True
    if shrink:
        input_size = 196
    d_feature = 2
    d_output = 10
    batch_size = 1000
    permuted = False

    data_source = MNISTpreprocessing.MNISTDatasource(shrink, permuted = permuted)
    
    # Initialise the model
    network = MPS(d_feature, d_output, input_size)
    network.prepare(data_source)
    feature, label = data_source.next_training_data_batch(1000)
    network.test(feature, label)

