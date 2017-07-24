import tensorflow as tf
import numpy as np
from mps import MPS
import MNISTpreprocessing


class SimpleMPS(MPS):

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
        super().__init__(d_feature, d_output, input_size)

    def predict(self, feature):
        C1 = tf.matmul(feature[0], self.nodes[0])
        C1 = tf.Print(C1, [C1])
        for i in range(self.input_size-1):
            node = self.nodes[i] 
            contracted_node = tf.einsum('tn,nij->tij', feature[i], node)
            C1 = tf.einsum('ti,tij->tj', C1, contracted_node)
        C2 = tf.matmul(feature[-1], self.nodes[-1])
        return tf.einsum('ti,til->l', C1, C2)

    def _setup_nodes(self):
        """

        :return:
        """
        with tf.name_scope("MPSnodes"):
            self.nodes= []
            self.nodes.append(self._make_start_vector())

            for i in range(1, self.input_size - 1):
                self.nodes.append(self._make_middle_node(i))
                                                                   
            self.nodes.append(self._make_end_vector())
    

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

        end_vector = np.zeros([self.d_feature, self.d_matrix, self.d_output], dtype=np.float32)


        end_vector[0, 0:self.d_output, :] = np.identity(self.d_output)
        end_vector[0, self.d_output:, :] = np.diag(self.bias)
        end_vector[1:, self.d_output:, :] = np.diag(self.weight[-1])

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






