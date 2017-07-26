import tensorflow as tf
import numpy as np
from mps import MPS
import MNISTpreprocessing


class SimpleMPS(MPS):

    def __init__(self, d_feature, d_output, input_size, feature_reg=1.1, reg=0.001):
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
        self.feature_reg = feature_reg
        self.reg = reg

        super().__init__(d_feature, d_output, input_size)

    def predict(self, feature):

        with tf.name_scope('predict'):
            with tf.name_scope('calculate_C1'):
                C1 = tf.matmul(feature[0], self.nodes[0])
                for i in range(1, self._special_node_loc):
                    node = self.nodes[i]
                    contracted_node = tf.einsum('tn,nij->tij', feature[i], node)
                    C1 = tf.einsum('ti,tij->tj', C1, contracted_node)

            with tf.name_scope('calculate_C2'):
                C2 = tf.matmul(feature[-1], self.nodes[-1])
                for i in range(self.input_size-2, self._special_node_loc, -1):
                    node = self.nodes[i]
                    contracted_node = tf.einsum('tn,nij->tij', feature[i], node)
                    C2 = tf.einsum('tij,tj->ti', contracted_node, C2)

            special_node = self.nodes[self._special_node_loc]
            contracted_node = tf.einsum('tn,lnij->tlij',
                                        feature[self._special_node_loc],
                                        special_node)
            C2 = tf.einsum('tlij,tj->tli', contracted_node, C2)
            f = tf.einsum('ti,tli->tl', C1, C2)
        return f

    def regularisation(self):
        reg_weights = tf.constant([1] + [self.feature_reg] * (self.d_feature - 1),
                                  dtype=tf.float32)

        with tf.name_scope('penalty'):
            with tf.name_scope('penalty_left'):
                reg_node = tf.einsum('ni,n->ni', self.nodes[0], reg_weights)
                penalty_left = tf.tensordot(reg_node, reg_node, [[0], [0]])

                for i in range(1, self._special_node_loc):
                    node = self.nodes[i]
                    reg_node = tf.einsum('nij,n->nij', self.nodes[i], reg_weights)
                    contracted_node = tf.tensordot(reg_node, reg_node, [[0], [0]])
                    penalty_left = tf.einsum('ij,ikjl->kl', penalty_left, contracted_node)

            with tf.name_scope('penalty_right'):
                reg_node = tf.einsum('ni,n->ni', self.nodes[-1], reg_weights)
                penalty_right = tf.tensordot(reg_node, reg_node, [[0], [0]])

                for i in range(self.input_size-2, self._special_node_loc, -1):
                    node = self.nodes[i]
                    reg_node = tf.einsum('nij,n->nij', self.nodes[i], reg_weights)
                    contracted_node = tf.tensordot(reg_node, reg_node, [[0], [0]])
                    penalty_left = tf.einsum('ij,kilj->kl', penalty_left, contracted_node)

            special_node = self.nodes[self._special_node_loc]
            reg_node = tf.einsum('lnij,n->lnij', special_node, reg_weights)
            contracted_node = tf.tensordot(reg_node, reg_node, [[0, 1], [0, 1]])

            penalty_left = tf.einsum('ij,ikjl->kl', penalty_left, contracted_node)
            penalty = tf.tensordot(penalty_left, penalty_right, [[0, 1], [0, 1]])

        return penalty



    def cost(self, f, label):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=f)) #tf.reduce_mean((f - label)**2)
        reg_penalty = self.regularisation()
        return loss + reg_penalty

    def _setup_nodes(self):
        """

        :return:
        """
        with tf.name_scope("MPSnodes"):
            self.nodes= []
            self.nodes.append(self._make_start_vector())

            for i in range(1, self.input_size - 1):
                if i == self._special_node_loc:
                    self.nodes.append(self._make_special_node(i))
                else:
                    self.nodes.append(self._make_middle_node(i))

            self.nodes.append(self._make_end_vector())


    def _make_start_vector(self):
        """

        :return:
        """
        start_vector = np.zeros([self.d_feature, self.d_matrix], dtype=np.float32)
        start_vector[0, self.d_output:] = 1
        start_vector[1:, 0:self.d_output] = self.weight[0]
        return tf.Variable(start_vector, dtype=tf.float32, trainable=True, name='start')

    def _make_end_vector(self):
        """

        :return:
        """

        end_vector = np.zeros([self.d_feature, self.d_matrix], dtype=np.float32)
        end_vector[0, 0:self.d_output] = 1
        end_vector[0, self.d_output:] = self.bias
        end_vector[1:, self.d_output:] = self.weight[-1]

        return tf.Variable(end_vector, dtype=tf.float32, trainable=True, name='end')

    def _make_special_node(self, index):
        def _make_matrix(i):
            m = np.zeros([self.d_feature, self.d_matrix, self.d_matrix], dtype=np.float32)
            m[0, i, i] = 1
            m[0, i+self.d_output, i+self.d_output] = 1
            m[1:, i+self.d_output, i] = self.weight[index, :, i]

            return m

        stacked = [_make_matrix(i) for i in range(self.d_output)]
        node = np.stack(stacked, axis=0)

        return tf.Variable(node, dtype=tf.float32, trainable=True, name='special')


    def _make_middle_node(self, index):
        """

        :param index:
        :return:
        """

        middle_node = np.zeros([self.d_feature, self.d_matrix, self.d_matrix], dtype=np.float32)
        middle_node[0] = np.identity(self.d_matrix)
        for i in range(1, self.d_feature):
            middle_node[i, self.d_output:, 0:self.d_output] = np.diag(self.weight[index, i-1])

        return tf.Variable(middle_node, dtype=tf.float32, trainable=True, name='G_{}'.format(index))

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
    network = SimpleMPS(d_feature, d_output, input_size)
    network.prepare(data_source)
    feature, label = data_source.next_training_data_batch(1000)
    network.test(feature, label)






