import tensorflow as tf
import numpy as np
import MNISTpreprocessing


class MPS(object):
    """
    MPS represents a 'Matrix Product State', which can be optimised (using MPSOptimizer)
    to create a model which performs well for certain machine learning tasks, following
    Supervised learning with Quantum-Inspired Tensor Networks
    by E.Miles Stoudenmire and David J.Schwab
    The class can also be used on its own to perform inference.
    Variables:
    input_size: int
        The input size, i.e. the number of matrices composing the matrix product state
    d_matrix: int
        The initial sizes of the matrices
    d_feature: int
        The sizes of the feature vectors. 
        (Referred to as 'Local dimension' in the paper)
        Try to keep this low (if possible), as the optimisation algorithm scales
        as (d_feature)^3.
    d_output: int
        The size of the output. e.g. with 10-class classification,
        expressed as a one-hot vector, this would be 10
    nodes: tf.TensorArray
        a TensorArray containing all of the weights for this model.
    feature: tf.Placeholder of shape (input_size, batch_size, d_feature)
    
    Usage example:
    
    from optimizer import *
    import activitypreprocessing as ap
    
    # Model parameters
    d_feature = 4
    d_output = 7
    batch_size = 2000
    permuted = False
    shuffled = False
    input_size = 100
    lin_reg_iterations = 1000
    
    max_size = 15
    
    rate_of_change = 10**(-7)
    logging_enabled = False
    
    cutoff = 10 # change this next
    n_step = 10
    
    data_source = ap.activityDatasource(shuffled = shuffled)
    batch_size = data_source.num_train_samples
    
    print(data_source.num_train_samples, data_source.num_test_samples)
    
    # Testing
    
    # load weights that we have saved to a file from a previous run.
    with open('weights', 'rb') as fp:
       weights = pickle.load(fp)
       if len(weights) != input_size:
           weights = None
        
    network.prepare(data_source)   
    feed_dict = network.create_feed_dict(weights)
    test_features, test_labels = data_source.test_data
    features = tf.placeholder(tf.float32, shape=[input_size, None, d_feature])
    labels = tf.placeholder(tf.float32, shape=[None, d_output])
    f = network.predict(features)
    confusion_matrix = network.confusion_matrix(f, labels)
    accuracy = network.accuracy(f, labels)
    feed_dict[features] = test_features
    feed_dict[labels] = test_labels
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        conf, acc = sess.run([confusion_matrix, accuracy], feed_dict = feed_dict)
    print("\n\n\n\n Accuracy is:" + str(acc))
    print("\n\n\n\n" + str(conf))
    """

    def __init__(self, d_feature, d_output, input_size):
        """
        Initialises the MPS. Currently, the prepare method must be called immediately
        after this before anything else can be done.
        :param d_feature: integer
            The sizes of the feature vectors. 
            (Referred to as 'Local dimension' in the paper)
            Try to keep this low (if possible), as the optimisation algorithm scales
            as (d_feature)^3.
        :param d_output: integer
            The size of the output. e.g. with 10-class classification,
            expressed as a one-hot vector, this would be 10
        :param input_size: int
            The input size, i.e. the number of matrices composing the matrix product state
        """
        # structure parameters
        self.input_size = input_size
        self.d_matrix = d_output * 2
        self.d_feature = d_feature
        self.d_output = d_output
        self._special_node_loc = int(np.floor(self.input_size / 2))

    def prepare(self, data_source, iterations=1000, learning_rate=0.05):
        """
        Prepares the MPS using linear regression. This can be thought of as pre-training the network,
        and dramatically shortens the required training time. This function must be called immediately
        after the initialiser.
        :param data_source: (some subclass of) MPSDatasource
            The data/labels that the MPS will be trained on.
        :param iterations: integer
            The number of iterations for which the linear regression model is trained.
        :param learning_rate: float
            The learning rate to use when training with linear regression
        :return: nothing
        """
        self._lin_reg(data_source, iterations, learning_rate)
        self._setup_nodes()

    def test(self, test_feature, test_label):
        """
        A function to test the MPS.
        :param test_feature: a numpy array of type float32 of shape (input_size, batch_size, d_feature)
            The features for which the testing is to be done.
        :param test_label: a numpy array of shape (batch_size, d_output)
            The 'correct' labels against which the predictions from the test_feature will be judged.
        :return: nothing
        """
        # feature must be numpy array of dtype float32
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

#     def load_nodes(self, weights):
#         """
#         Deprecated.
#         :param weights:
#         :return:
#         """
#         updated_nodes = tf.TensorArray(tf.float32, size=0, dynamic_size=True,
#                                        clear_after_read=False, infer_shape=False)
#         with tf.name_scope("newMPSNodes"):
#             for index, weight in enumerate(weights):
#                 updated_nodes = updated_nodes.write(index, tf.Variable(weight))
#         self.nodes = updated_nodes

    # ================
    # hidden functions - not so hidden anymore
    # TODO: Move functions around so this actually makes sense
    # ================

    def cost(self, f, labels):
        """
        Returns the cost (softmax cross entropy with logits) given the predictions(f), and the correct labels
        :param f: tensorflow Tensor of shape (batch_size, d_output)
            The predictions that are to be judged.
        :param labels: tensorflow Tensor of shape (batch_size, d_output)
            The correct labels.
        :return: a tensorflow scalar
            The cost of the predictions, as judged by softmax cross entropy with logits
        """
        with tf.name_scope("cost"):
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=f)) # 0.5 * tf.reduce_sum(tf.square(f-label))
        return cost

    def accuracy(self, f, labels):
        """
        Returns the accuracy given the predictions(f), and the correct labels.
        :param f: tensorflow Tensor of shape (batch_size, d_output)
            The predictions that are to be judged.
        :param labels: tensorflow Tensor of shape (batch_size, d_output)
            The correct labels.
        :return: a tensorflow scalar
            The accuracy of the predictions
        """
        with tf.name_scope("accuracy"):
            prediction = tf.argmax(f, axis=1)
            true_value = tf.argmax(labels, axis=1)
            correct_prediction = tf.equal(prediction, true_value)
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return accuracy
        
    def confusion_matrix(self, f, labels):
        """
        Returns the confusion matrix given the predictions(f), and the correct labels
        :param f: tensorflow Tensor of shape (batch_size, d_output)
            The predictions that are to be judged.
        :param labels: tensorflow Tensor of shape (batch_size, d_output)
            The correct labels.
        :return: a tensorflow Tensor of shape (d_output, d_output)
            The confusion matrix
        """
        predictions = tf.argmax(f, axis=1)
        true_values = tf.argmax(labels, axis=1)
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

        accuracy = self.accuracy(prediction, label)
        confusion_matrix = self.confusion_matrix(prediction, label)

        reshaped_weight = tf.transpose(weight)
        reshaped_weight = tf.reshape(reshaped_weight, [self.d_output, self.input_size, self.d_feature-1])
        reshaped_weight = tf.transpose(reshaped_weight, perm=[1, 2, 0])


        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for _ in range(iterations):

                batch_feature, batch_label = data_source.next_training_data_batch(100)
                sess.run(train_step, feed_dict={feature: batch_feature, label: batch_label})

            train_acc, train_conf = sess.run([accuracy, confusion_matrix], feed_dict={feature: batch_feature, label: batch_label})
            print('Lin regression gives a training accuracy of {}'.format(train_acc))
            print(train_conf)

            batch_feature, batch_label = data_source.test_data
            test_acc, test_conf = sess.run([accuracy, confusion_matrix], feed_dict={feature: batch_feature, label: batch_label})
            print('Lin regression gives a test accuracy of {}'.format(test_acc))
            print(test_conf)

            self.weight = sess.run(reshaped_weight)
            self.bias = sess.run(bias)
            del batch_feature, batch_label

    def _setup_nodes(self):
        """

        :return:
        """

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
        """

        :return:
        """
        start_vector = np.zeros([self.d_feature, self.d_matrix], dtype=np.float32)
        start_vector[0, self.d_output:] = 1
        start_vector[1:, 0:self.d_output] = self.weight[0]
        return tf.Variable(start_vector, dtype=tf.float32)

    def _make_end_vector(self):
        """

        :return:
        """

        end_vector = np.zeros([self.d_feature, self.d_matrix], dtype=np.float32)
        end_vector[0, 0:self.d_output] = 1 
        end_vector[0, self.d_output:] = self.bias 
        end_vector[1:, self.d_output:] = self.weight[-1] 

        return tf.Variable(end_vector, dtype=tf.float32)

    def _make_middle_node(self, index):
        """

        :param index:
        :return:
        """

        middle_node = np.zeros([self.d_feature, self.d_matrix, self.d_matrix], dtype=np.float32)
        middle_node[0] = np.identity(self.d_matrix)
        for i in range(1, self.d_feature):
            middle_node[i, self.d_output:, 0:self.d_output] = np.diag(self.weight[index, i-1])

        return tf.Variable(middle_node, dtype=tf.float32)

    def _make_special_node(self, index):
        """

        :param index:
        :return:
        """
        
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
        """
        Makes predictions based on the features. Can do batch prediction.
        :param feature: tensorflow Tensor of shape (input_size, batch_size, d_feature)
            The features for which the predictions are to be made.
        :return: tensorflow Tensor of shape (batch_size, d_output)
            The predictions
        """
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
        """
        Creates a feed_dict which assigns the given weights to the MPS' nodes.
        This should be used whenever you want to update the weights of the MPS.
        
        e.g.:
        feed_dict = network.create_feed_dict(weights)
        test_features, test_labels = data_source.test_data
        features = tf.placeholder(tf.float32, shape=[input_size, None, d_feature])
        labels = tf.placeholder(tf.float32, shape=[None, d_output])
        f = network.predict(features)
        confusion_matrix = network.confusion_matrix(f, labels)
        accuracy = network.accuracy(f, labels)
        feed_dict[features] = test_features
        feed_dict[labels] = test_labels
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            conf, acc = sess.run([confusion_matrix, accuracy], feed_dict = feed_dict)
            
        :param weights: list of numpy arrays of length input_length
            The weights that you want to use for the MPS.
        :return: dictionary
            Edit this dictionary for any other things you want to pass in the feed_dict.
        """
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

