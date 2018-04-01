import tensorflow as tf
import numpy as np
from mps.mps import MPS

class shortMPS(MPS):

    def __init__(self, d_feature, d_output, input_size, special_node_loc=None, round=False,):
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
        self.d_matrix = 1
        self.d_feature = d_feature
        self.d_output = d_output
        self._special_node_loc = special_node_loc
        self.round = round
        # if special_node_loc is None:
        #     self._special_node_loc = int(np.floor(self.input_size / 2))
        # else:
        #     self._special_node_loc = special_node_loc

    def _setup_nodes(self):
        """

        :return:
        """

        with tf.name_scope("MPSnodes"):

            # Make the end nodes
            self.start_node = self._make_start_node()
            self.end_node = self._make_end_node()

            # Make the node chain
            self.nodes_list = []
            self.nodes = tf.TensorArray(tf.float32, size=0, dynamic_size=True,
                                        clear_after_read=False, infer_shape=False)

            first_node, sv = self._make_first_node()
            self._append_node(0, first_node, [self.d_feature, None, None])

            node, shape = self._make_node(1)
            node = tf.einsum('ij,mjk->mik', sv, node)
            self._append_node(1, node, shape)

            for i in range(2, self.input_size-2):
                node, shape = self._make_node(i)
                self._append_node(i, node, shape)

            node, shape = self._make_node(self.input_size-2)
            last_node, sv = self._make_last_node()
            node = tf.einsum('mij,jk->mik', node, sv)
            self._append_node(self.input_size-2, node, shape)
            self._append_node(self.input_size-1, last_node, [self.d_feature, None, None])

    def _make_start_node(self):
        start_vector = tf.Variable(np.ones([1, 1], dtype=np.float32), trainable=False)
        return start_vector

    def _make_end_node(self):
        end_node = tf.Variable(np.ones([1, 1], dtype=np.float32), trainable=False)
        return end_node

    def _append_node(self, index, node, shape):
        self.nodes_list.append(tf.placeholder_with_default(node, shape))
        self.nodes = self.nodes.write(index, self.nodes_list[-1])

    def _make_first_node(self):
        """

        :return:
        """
        start_vector = np.zeros([self.d_feature, self.d_output * 2], dtype=np.float32)
        start_vector[0, self.d_output:] = 1
        start_vector[1:, 0:self.d_output] = self.weight[0]
        first_node = tf.Variable(start_vector, dtype=tf.float32,
                           trainable=False, name='first_node')
        s, u, v = tf.svd(first_node)
        sv = tf.einsum('i,ji->ij', s, v)
        u = tf.expand_dims(u, 1)
        u = tf.Print(u, [tf.shape(u), u], message='first node shape')

        return u, sv

    def _make_last_node(self):
        """

        :return:
        """

        end_vector = np.zeros([self.d_feature, self.d_output * 2], dtype=np.float32)
        end_vector[0, 0:self.d_output] = 1
        end_vector[1:, self.d_output:] = self.weight[-1]

        last_node = tf.Variable(end_vector, dtype=tf.float32,
                           trainable=False, name='last_node')
        s, u, v = tf.svd(last_node)
        s = tf.expand_dims(s, 0)
        sv = tf.multiply(s, v)
        u = tf.expand_dims(u, 2)
        u = tf.Print(u, [tf.shape(u), u], message='last node shape')

        return u, sv

    def _make_node(self, index):
        if index == self._special_node_loc:
            return self._make_special_node(index)
        else:
            return self._make_middle_node(index)

    def _make_special_node(self, index):

        def _make_matrix(i):
            m = np.zeros([self.d_feature, self.d_output * 2, self.d_output * 2], dtype=np.float32)
            m[0, i, i] = 1
            m[0, i+self.d_output, i+self.d_output] = 1
            m[0, i+self.d_output, i] = self.bias[i]
            m[1:, i+self.d_output, i] = self.weight[index, :, i]

            return m

        stacked = [_make_matrix(i) for i in range(self.d_output)]
        node = np.stack(stacked, axis=0)

        special_node = tf.Variable(node, dtype=tf.float32,
                           trainable=True, name='special')
        shape = [self.d_output, self.d_feature, None, None]

        return special_node, shape

    def _make_middle_node(self, index):
        """

        :param index:
        :return:
        """

        middle_node = np.zeros([self.d_feature, self.d_output * 2, self.d_output * 2], dtype=np.float32)
        middle_node[0] = np.identity(self.d_output * 2)
        for i in range(1, self.d_feature):
            middle_node[i, self.d_output:, 0:self.d_output] = np.diag(self.weight[index, i-1])

        middle = tf.Variable(middle_node, dtype=tf.float32, trainable=True, name='node_{}'.format(index))
        shape = [self.d_feature, None, None]

        return middle, shape

if __name__ == '__main__':
    # Model parameters
    input_size = 784
    shrink = True
    if shrink:
        input_size = 196
    d_feature = 2
    d_output = 11
    batch_size = 1000
    permuted = False

    data_source = MNISTpreprocessing.MNISTDatasource(shrink, permuted=permuted)

    # Initialise the model
    network = shortMPS(d_feature, d_output, input_size)
    network.prepare(data_source)
    feature, label = data_source.test_data
    network.test(feature, label)
