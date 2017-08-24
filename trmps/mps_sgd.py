import tensorflow as tf
import numpy as np
from mps import MPS
import MNISTpreprocessing


class SGDMPS(MPS):
    """
    SimpleMPS is used as the MPS for stochastic gradient descent. 
    Variables: 
    d_feature: int 
        The dimension of the feature vectors
    d_output: int 
        The dimension of the output vectors 
    input_size: int 
        The input size, i.e. the number of matrices composing the matrix product state
    feature_reg: float 
        Parameter for regularising the weights. Set it larger than 1 for better performance.
    reg: float 
        Ratio between loss and the regularisation penalty. Set it smaller than 1 for better performance.
    start_node: tf.Tensor
        Untrainable tf.Variable, the first node of the MPS 
    end_node: tf.Tensor
        Untrainable tf.Variable, the last node of the MPS 
    nodes: list of tf.Tensor 
        List of trainable tf.Variable, the middle nodes of the MPS 

    Usage example: 
    import MNISTpreprocessing

    # Parameters
    input_size = 196
    shrink = True
    d_feature = 2
    d_output = 10
    permuted = False

    # Initialise with linear regression
    data_source = MNISTpreprocessing.MNISTDatasource(shrink, permuted = permuted)
    network = SimpleMPS(d_feature, d_output, input_size)
    network.prepare(data_source=data_source)
    feature, label = data_source.test_data 
    network.test(feature, label)
    """

    def __init__(self, d_feature, d_output, input_size, feature_reg=1.1, reg=0.001, special_node_loc=None):
        """
        Initialises the MPS. Currently, the prepare method must be called immediately
        after this before anything else can be done.
        :param d_feature: integer
            The sizes of the feature vectors.
            (Referred to as 'Local dimension' in the paper)
        :param d_output: integer
            The size of the output. e.g. with 10-class classification,
            expressed as a one-hot vector, this would be 10
        :param input_size: int
            The input size, i.e. the number of matrices composing the matrix product state
        :param feature_reg: float 
            Parameter for regularising the weights. Set it larger than 1 for better performance.
        :param reg: float 
            Ratio between loss and the regularisation penalty. Set it smaller than 1 for better performance.
        """
        # structure parameters
        self.feature_reg = feature_reg
        self.reg = reg

        super().__init__(d_feature, d_output, input_size, special_node_loc=special_node_loc)

    def predict(self, feature):
        batch_size = tf.shape(feature)[1]

        with tf.name_scope('predict'):
            with tf.name_scope('calculate_C1'):
                C1 = tf.tile(self.start_node, [batch_size, 1])
                for i in range(0, self._special_node_loc):
                    node = self.nodes[i]
                    contracted_node = tf.einsum('tn,nij->tij', feature[i], node)
                    C1 = tf.einsum('ti,tij->tj', C1, contracted_node)

            with tf.name_scope('calculate_C2'):
                C2 = tf.tile(self.end_node, [batch_size, 1])
                for i in range(self.input_size-1, self._special_node_loc, -1):
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
                penalty_left = tf.eye(self.d_matrix)

                for i in range(0, self._special_node_loc):
                    node = self.nodes[i]
                    reg_node = tf.einsum('nij,n->nij', self.nodes[i], reg_weights)
                    contracted_node = tf.tensordot(reg_node, reg_node, [[0], [0]])
                    penalty_left = tf.einsum('ij,ikjl->kl', penalty_left, contracted_node)

            with tf.name_scope('penalty_right'):
                penalty_right = tf.eye(self.d_matrix)

                for i in range(self.input_size-1, self._special_node_loc, -1):
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
        return loss + reg_penalty * self.reg 

    def _setup_nodes(self):
        """

        :return:
        """
        with tf.name_scope("MPSnodes"):
            self.start_node = self._make_start_node()
            self.end_node = self._make_end_node()
            self.nodes= []

            for i in range(0, self.input_size):
                if i == self._special_node_loc:
                    self.nodes.append(self._make_special_node(i))
                else:
                    self.nodes.append(self._make_middle_node(i))




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






