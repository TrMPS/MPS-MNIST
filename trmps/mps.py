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

    def prepare(self, data_source, iterations=1000, learning_rate=0.05):
        self._lin_reg(data_source, iterations, learning_rate)
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
        
    def confusion_matrix(self, f, label):
        predictions = tf.argmax(f, axis=1)
        true_values = tf.argmax(label, axis=1)
        confusion_mat = tf.confusion_matrix(true_values, predictions, num_classes = self.d_output)
        return confusion_mat

    def _lin_reg(self, data_source, iterations, learning_rate):

        x_dim = self.input_size * (self.d_feature - 1)

        with tf.name_scope('model_params'):
            weight = tf.Variable(tf.zeros([x_dim, self.d_output]))
            bias = tf.Variable(tf.zeros([self.d_output]))

        with tf.name_scope('data'):
            feature = tf.placeholder(tf.float32, shape=[self.input_size, None, self.d_feature])
            label = tf.placeholder(tf.float32, shape=[None, self.d_output])

        with tf.name_scope('lin_model'):
            x = tf.transpose(feature[:, :, 1:], perm=[1, 0, 2])
            x = tf.contrib.layers.flatten(x) # flatten out everything except the first dim


            prediction = tf.matmul(x, weight) + bias
            #cross_entropy = 0.5 * tf.reduce_sum(tf.square(prediction-label))
            cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=prediction))
            train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy) 

        correct_prediction = tf.equal(tf.argmax(label,1), tf.argmax(prediction,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        reshaped_weight = tf.transpose(weight)
        reshaped_weight = tf.reshape(reshaped_weight, [self.d_output, self.input_size, self.d_feature-1])
        reshaped_weight = tf.transpose(reshaped_weight, perm=[1, 2, 0])


        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for _ in range(iterations):

                batch_feature, batch_label = data_source.next_training_data_batch(100)
                sess.run(train_step, feed_dict={feature: batch_feature, label: batch_label})
            train_acc = accuracy.eval(feed_dict={feature: batch_feature, label: batch_label})
            print('Lin regression gives a training accuracy of {}'.format(train_acc))
            batch_feature, batch_label = data_source.test_data
            test_acc = accuracy.eval(feed_dict={feature: batch_feature, label: batch_label})
            print('Lin regression gives a test accuracy of {}'.format(test_acc))
            self.weight = sess.run(reshaped_weight)
            self.bias = sess.run(bias)
            del batch_feature, batch_label

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
        start_vector = np.zeros([self.d_feature, self.d_matrix], dtype=np.float32)
        start_vector[0, self.d_output:] = 1
        start_vector[1:, 0:self.d_output] = self.weight[0]
        return tf.Variable(start_vector, dtype=tf.float32)

    def _make_end_vector(self):

        end_vector = np.zeros([self.d_feature, self.d_matrix], dtype=np.float32)
        end_vector[0, 0:self.d_output] = 1 
        end_vector[0, self.d_output:] = self.bias 
        end_vector[1:, self.d_output:] = self.weight[-1] 

        return tf.Variable(end_vector, dtype=tf.float32)

    def _make_middle_node(self, index):

        middle_node = np.zeros([self.d_feature, self.d_matrix, self.d_matrix], dtype=np.float32)
        middle_node[0] = np.identity(self.d_matrix)
        for i in range(1, self.d_feature):
            middle_node[i:, self.d_output:, 0:self.d_output] = np.diag(self.weight[index, i-1])

        return tf.Variable(middle_node, dtype=tf.float32)

    def _make_special_node(self, index):
        
        def _make_matrix(i):
            m = np.zeros([self.d_feature, self.d_matrix, self.d_matrix], dtype=np.float32)
            m[0, i, i] = 1 
            m[0, i+self.d_output, i+self.d_output] = 1
            m[1:, i+self.d_output, i] = self.weight[index, :, i]

            return tf.Variable(m)

        stacked = [_make_matrix(i) for i in range(self.d_output)]

        return tf.stack(stacked)

    def _chain_multiply_r(self, counter, C1):
        with tf.name_scope('chain_multiply_right'):
            node = self.nodes.read(counter)
            node.set_shape([self.d_feature, None, None])
            contracted_node = tf.tensordot(self.feature[counter], node, 
                                            [[1], [0]])
            C1 = tf.einsum('ti,tij->tj', C1, contracted_node)
            counter = counter + 1
        return [counter, C1]

    def _chain_multiply_l(self, counter, C2):
        with tf.name_scope('chain_multiply_left'):
            node = self.nodes.read(counter)
            node.set_shape([self.d_feature, None, None])
            contracted_node = tf.tensordot(self.feature[counter], node, 
                                            [[1], [0]])
            C2 = tf.einsum('tij,tj->ti', contracted_node, C2)
            counter = counter - 1
        
        return [counter, C2]
        

    def predict(self, feature):
        with tf.name_scope("MPS_predict"):
            # Read in feature 
            self.feature = feature

            with tf.name_scope('calculate_C1'):
                node1 = self.nodes.read(0)
                node1.set_shape([self.d_feature, None])
                C1 = tf.matmul(feature[0], node1)
                cond = lambda c, b: tf.less(c, self._special_node_loc)
                counter, C1 = tf.while_loop(cond=cond, body=self._chain_multiply_r, loop_vars=[1, C1],
                                            shape_invariants=[tf.TensorShape([]), tf.TensorShape([None, None])],
                                            parallel_iterations=5)

            with tf.name_scope('calculate_C2'):
                nodelast = self.nodes.read(self.input_size - 1)
                nodelast.set_shape([self.d_feature, None])
                C2 = tf.matmul(feature[self.input_size-1], nodelast)
                cond2 = lambda c, b: tf.greater(c, self._special_node_loc)
                _, C2 = tf.while_loop(cond=cond2, body=self._chain_multiply_l, loop_vars=[self.input_size-2, C2],
                                      shape_invariants=[tf.TensorShape([]), tf.TensorShape([None, None])],
                                      parallel_iterations=5)


            # contract special node with C1 
            sp_node = self.nodes.read(self._special_node_loc)
            sp_node.set_shape([self.d_output, self.d_feature, None, None])
            contracted_sp_node = tf.einsum('lnij,tn->tlij', sp_node, feature[self._special_node_loc])
            C1 = tf.einsum('ti,tlij->tlj', C1, contracted_sp_node)
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

