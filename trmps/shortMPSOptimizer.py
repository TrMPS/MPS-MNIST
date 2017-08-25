from optimizer import MPSOptimizer
from parameterObjects import MPSOptimizerParameters, MPSTrainingParameters
import tensorflow as tf


class shortMPSOptimizer(MPSOptimizer):

    def __init__(self, MPSNetwork, max_size, sweep_range=None, optional_parameters=MPSOptimizerParameters()):

        if sweep_range == None:
            sweep_range = (0, MPSNetwork.input_size-1)
        self._left_lim = sweep_range[0] 
        self._right_lim = sweep_range[1] 
        super().__init__(MPSNetwork, max_size, optional_parameters)


    def train_step(self):
        """
        A single step of training. Interface with this if you need to chain training with other tensorflow operations.
        If not, it is recommended to just use the train function.

        :return: the accuracy as calculated at the end of a training step.
        """
        self.batch_size = tf.shape(self._feature)[1]
        self.acc_lr_reg = self.lr_reg

        with tf.name_scope("train_step"):
            # Create updated_nodes and fill in the first half from current one
            self.updated_nodes = self._duplicate_nodes(self.MPS.nodes, self._left_lim, self.MPS._special_node_loc)
            original_special_node_loc = self.MPS._special_node_loc

            # First half-sweep
            self.updated_nodes = self._sweep_right(self.MPS._special_node_loc, self._right_lim)
            self.MPS.nodes = self.updated_nodes
            self.MPS._special_node_loc = self._right_lim

            # First back-sweep
            self.updated_nodes = self._duplicate_nodes(self.MPS.nodes, 0, 0)
            self.updated_nodes = self._sweep_left(self._right_lim, self._left_lim)
            self.MPS.nodes = self.updated_nodes
            self.MPS._special_node_loc = self._left_lim

            # Second half-sweep
            self.updated_nodes = self._duplicate_nodes(self.MPS.nodes, original_special_node_loc + 1,
                                                       self._right_lim + 1)
            C1 = self.C1s.read(self._left_lim)
            self.C1s = tf.TensorArray(tf.float32, size=self.MPS.input_size, dynamic_size=False, infer_shape=False,
                                      clear_after_read=False)
            self.C1s = self.C1s.write(self._left_lim, C1)
            self.updated_nodes = self._sweep_right(self._left_lim, original_special_node_loc)
            self.MPS.nodes = self.updated_nodes
            self.MPS._special_node_loc = original_special_node_loc

            # accuracy
            f = self.MPS.predict(self._feature)
            accuracy = self.MPS.accuracy(f, self._label)

        return accuracy

    def _duplicate_nodes(self, nodes, from_index, to_index):
        """

        :param self:
        :param nodes:
        :param from_index:
        :param to_index:
        :return:
        """
        '''
        duplicate the nodes in the range (from_index, to_index)
        '''
        with tf.name_scope("duplicatingNodes"):
            size = nodes.size()
            new_nodes = tf.TensorArray(tf.float32, size=size, infer_shape=False, clear_after_read=False)

            _, _, new_nodes = tf.while_loop(cond=lambda i, a, b: tf.less(i, self._left_lim), 
                                            body=self._transfer_to_array, 
                                            loop_vars=[0, nodes, new_nodes], 
                                            name="copy_start_nodes")
            _, _, new_nodes = tf.while_loop(cond=lambda i, a, b: tf.less(i, size), 
                                            body=self._transfer_to_array, 
                                            loop_vars=[self._right_lim+1, nodes, new_nodes],
                                            name="copy_end_nodes")


            from_index = tf.cond(tf.less(from_index, self._left_lim), lambda: self._left_lim, lambda: from_index)
            to_index = tf.cond(tf.greater(to_index, self._right_lim+1), lambda: self._right_lim+1, lambda: to_index)
            cond = lambda index, a, b: tf.less(index, to_index)

            _, _, new_nodes = tf.while_loop(cond=cond, body=self._transfer_to_array,
                                            loop_vars=[from_index, nodes, new_nodes], name="duplicate_loop")
        return new_nodes

