import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tensorflow as tf
import time
import numpy as np
from mps.mps import MPS
import pickle
import utils
from tensorflow.python.client import timeline
from optimizers.parameterObjects import MPSOptimizerParameters, MPSTrainingParameters
from mps.squaredDistanceMPS import sqMPS

class BaseOptimizer(object):

    def __init__(self, MPSNetwork, max_size, optional_parameters=MPSOptimizerParameters()):

        """
        Initialises the optimiser.
        :param MPSNetwork: MPS
            The matrix product state network that will be optimised.
        :param max_size: integer
            The maximum size the tensors composing the MPS can grow to.
        :param optional_parameters: MPSOptimizerParameters
            Optional parameters for the MPSOptimizer.
            See documentation for MPSOptimizerParameters for more detail.
        """
        self.parameters = optional_parameters
        self.path = self.parameters.path
        self.MPS = MPSNetwork
        self.use_hessian = self.parameters.use_hessian
        if self.use_hessian and type(self.MPS) is sqMPS:
            print("Using the hessian for squared distance error is not yet supported, sorry!")
            self.use_hessian = False
        self._armijo_iterations = self.parameters.armijo_iterations
        self.rate_of_change = tf.placeholder(tf.float32, shape=[])
        self.reg = self.parameters.reg
        self.lr_reg = self.parameters.lr_reg
        self.max_size = max_size
        self.armijo_coeff = self.parameters.armijo_coeff
        self.cutoff = self.parameters.cutoff
        self.min_singular_value = self.parameters.min_singular_value
        self._feature = tf.placeholder(tf.float32, shape=[self.MPS.input_size, None, self.MPS.d_feature])
        self._label = tf.placeholder(tf.float32, shape=[None, self.MPS.d_output])
        self._setup_optimization()
        self.verbosity = self.parameters.verbosity
        self.updates_per_step = self.parameters.updates_per_step
        self.costs1 = tf.TensorArray(tf.float32, size=self.MPS.input_size-1, infer_shape=True, clear_after_read=False)
        self.costs2 = tf.TensorArray(tf.float32, size=self.MPS.input_size-1, infer_shape=True, clear_after_read=False)
        _ = self.train_step()

        print("_____   Thomas the Tensor Train    . . . . . o o o o o",
              "  __|[_]|__ ___________ _______    ____      o",
              " |[] [] []| [] [] [] [] [_____(__  ][]]_n_n__][.",
              "_|________|_[_________]_[________]_|__|________)<",
              "  oo    oo 'oo      oo ' oo    oo 'oo 0000---oo\_",
              " ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", sep="\n")

    def train(self, data_source, batch_size, n_step, optional_parameters=MPSTrainingParameters(), plot_func=None):
        """
        Trains the network.
        If it is required to chain the training with other tensorflow steps, do not use this function.
        However, it may be helpful to base it on the way this function is implemented,
        as the way the MPS works is quite unique, so the way things will have to be done is
        somewhat different from how it is usually done in tensorflow.

        :param data_source: (some subclass of) MPSDatasource
            The data/labels that the MPS will be trained on.
        :param batch_size: integer
            The batch size to be used when feeding in data for the sweeps.
        :param n_step: integer
            The number of steps of training that should be performed.
            A step of training consists of a full sweep 'forward' and 'backward'
            such that the output leg is attached to a node at the same position as at the start.
            Typically, (if the batch size is all of the data), then a couple of steps should
            be enough to fully optimise the MPS.
        :param optional_parameters: MPSTrainingParameters
            Optional parameters for training in the MPSOptimizer.
            See documentation for MPSTrainingParameters for more detail.
        :return: nothing
        """

        run_options = []
        run_metadata = []
        if optional_parameters._logging_enabled:
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()

        self.feed_dict = None
        self.test = None
        test_result = utils.list_from(self.updated_nodes, length=self.MPS.input_size)
        self.test = optional_parameters.initial_weights
        initial_lr = optional_parameters.rate_of_change

        train_cost, train_accuracy, train_confusion, _, train_f = self._test_step(self._feature, self._label)

        test_feature, test_label = data_source.test_data

        feature = tf.placeholder(tf.float32, shape=[self.MPS.input_size, None, self.MPS.d_feature])
        label = tf.placeholder(tf.float32, shape=[None, self.MPS.d_output])
        test_cost, test_accuracy, test_confusion, test_f1, test_f = self._test_step(feature, label)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            if optional_parameters._logging_enabled:
                writer = tf.summary.FileWriter("output", sess.graph)
            for i in range(n_step):
                start = time.time()

                rate_of_change = initial_lr / np.sqrt(1-self.lr_reg**(i+1))
                print(rate_of_change)
                (batch_feature, batch_label) = data_source.next_training_data_batch(batch_size)

                self.feed_dict = self.MPS.create_feed_dict(self.test)
                self.feed_dict[self._feature] = batch_feature
                self.feed_dict[self._label] = batch_label
                self.feed_dict[self.rate_of_change] = rate_of_change
                self.feed_dict[feature] = test_feature
                self.feed_dict[label] = test_label
                to_eval = [train_cost, test_result, train_accuracy, test_cost, test_accuracy, test_confusion, test_f1,
                           self.costs1_stacked, self.costs2_stacked, test_f]
                train_c, self.test, train_acc, test_c, test_acc, test_conf, test_f1score, costs1, costs2, test_prediction = sess.run(to_eval,
                                                                                                    feed_dict=self.feed_dict,
                                                                                                    options=run_options,
                                                                                                    run_metadata=run_metadata)

                if optional_parameters._logging_enabled:
                    # writer.add_run_metadata(run_metadata, 'step' + str(i))
                    tl = timeline.Timeline(run_metadata.step_stats)
                    ctf = tl.generate_chrome_trace_format()
                    with open("timeline.json", "w") as f:
                        f.write(ctf)
                    run_metadata = tf.RunMetadata()
                self.MPS.save(self.test, path=self.path, verbose=optional_parameters.verbose_save)
                end = time.time()
                print(time.strftime("%Y-%m-%d %H:%M:%S"))
                print('step {}, training cost {}, accuracy {}. Took {} s'.format(i, train_c, train_acc, end - start))
                print('step {}, testing cost {}, accuracy {}'.format(i, test_c, test_acc))
                print('f1 score: ', test_f1score)
                print('sample prediction: ', test_prediction[0])
                print('sample truth: ', test_label[0])
                print('confusion matrix: \n' + str(test_conf))
                if plot_func is not None:
                    plot_func(self, costs1, costs2, i)
                # print("prediction:" + str(prediction[0]))
            if optional_parameters._logging_enabled:
                writer.close()

    def _test_step(self, feature, label):
        """
        A single step of training
        :param self:
        :param feature:
        :param label:
        :return:
        """
        f = self.MPS.predict(feature)
        cost = self.MPS.cost(f, label)
        accuracy = self.MPS.accuracy(f, label)
        confusion = self.MPS.confusion_matrix(f, label)
        f1 = self.MPS.f1score(f, label, confusion)
        return cost, accuracy, confusion, f1, f

    def _find_C1(self, counter, C1, C1s):
        """

        :param self:
        :param counter:
        :param C1:
        :param C1s:
        :return:
        """
        '''
        finds new C1, and write into C1s[counter]
        '''
        node = self.MPS.nodes.read(counter)
        node.set_shape([self.MPS.d_feature, None, None])
        input_leg = self._feature[counter]
        # contracted_node = tf.einsum('mij,tm->tij', node, input_leg)
        contracted_node = tf.tensordot(input_leg, node, [[1], [0]])
        C1 = tf.einsum('ti,tij->tj', C1, contracted_node)
        C1s = C1s.write(counter+1, C1)
        counter = counter + 1
        return [counter, C1, C1s]

    def _find_C2(self, counter, prev_C2, C2s):
        """

        :param self:
        :param counter:
        :param prev_C2:
        :param C2s:
        :return:
        """
        '''
        finds new C2, and write into C2s[counter]
        '''
        node2 = self.MPS.nodes.read(counter)
        node2.set_shape([self.MPS.d_feature, None, None])
        # contracted_node2 = tf.einsum('mij,tm->tij', node2, self._feature[loc2])  # CHECK einsum
        contracted_node2 = tf.tensordot(self._feature[counter], node2, [[1], [0]])
        new_C2 = tf.einsum('tij,tj->ti', contracted_node2, prev_C2)
        C2s = C2s.write(counter-1, new_C2)
        counter = counter - 1
        return [counter, new_C2, C2s]

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
            self.updated_nodes = self._duplicate_nodes(self.MPS.nodes, 0, self.MPS._special_node_loc)
            original_special_node_loc = self.MPS._special_node_loc

            # First half-sweep
            self.updated_nodes = self._sweep_right(self.MPS._special_node_loc, self.MPS.input_size-1)
            self.MPS.nodes = self.updated_nodes
            self.MPS._special_node_loc = self.MPS.input_size - 1

            # First back-sweep
            self.updated_nodes = self._duplicate_nodes(self.MPS.nodes, 0, 0)
            self.updated_nodes = self._sweep_left()
            self.MPS.nodes = self.updated_nodes
            self.MPS._special_node_loc = 0

            # Second half-sweep
            self.updated_nodes = self._duplicate_nodes(self.MPS.nodes, original_special_node_loc + 1,
                                                       self.MPS.input_size)
            C1 = self.C1s.read(0)
            self.C1s = tf.TensorArray(tf.float32, size=self.MPS.input_size, dynamic_size=False, infer_shape=False,
                                      clear_after_read=False)
            self.C1s = self.C1s.write(0, C1)
            self.updated_nodes = self._sweep_right(0, original_special_node_loc)
            self.MPS.nodes = self.updated_nodes
            self.MPS._special_node_loc = original_special_node_loc

            # accuracy
            f = self.MPS.predict(self._feature)
            accuracy = self.MPS.accuracy(f, self._label)

        self.costs1_stacked = self.costs1.stack()
        self.costs2_stacked = self.costs2.stack()
        return accuracy

    def _sweep_left(self, from_index=None, to_index=0):
        """

        :param self:
        :return:
        """
        if from_index == None:
            from_index = self.MPS.input_size - 1

        # read second from end node
        n1 = self.MPS.nodes.read(self.MPS._special_node_loc)
        n1.set_shape([self.MPS.d_output, self.MPS.d_feature, None, None])

        C2 = self.C2s.read(from_index)
        self.C2s = tf.TensorArray(tf.float32, size=self.MPS.input_size, infer_shape=False, clear_after_read=False)
        self.C2s = self.C2s.write(from_index, C2)
        cond = lambda counter, *args: tf.greater(counter, to_index)
        wrapped = [from_index, self.acc_lr_reg, self.C2s, self.updated_nodes, n1, self.costs2]
        shape_invariants = [tf.TensorShape([]), tf.TensorShape([]),
                            tf.TensorShape(None), tf.TensorShape(None),
                            tf.TensorShape([None, None, None, None]),
                            tf.TensorShape(None)]

        _, self.acc_lr_reg, self.C2s, self.updated_nodes, n1, self.costs2 = tf.while_loop(cond=cond, body=self._update_left,
                                                            loop_vars=wrapped,
                                                            shape_invariants=shape_invariants,
                                                            parallel_iterations=10,
                                                            name="leftSweep")
        self.updated_nodes = self.updated_nodes.write(to_index, n1)
        return self.updated_nodes

    def _sweep_right(self, from_index, to_index):
        """

        :param self:
        :param from_index:
        :param to_index:
        :return:
        """
        n1 = self.MPS.nodes.read(from_index)
        n1.set_shape([self.MPS.d_output, self.MPS.d_feature, None, None])

        cond = lambda counter, *args: tf.less(counter, to_index)
        wrapped = [from_index, self.acc_lr_reg, self.C1s, self.updated_nodes, n1, self.costs1]
        shape_invariants = [tf.TensorShape([]), tf.TensorShape([]),
                            tf.TensorShape(None), tf.TensorShape(None),
                            tf.TensorShape([None, None, None, None]),
                            tf.TensorShape(None)]
        _, self.acc_lr_reg, self.C1s, self.updated_nodes, n1, self.costs1 = tf.while_loop(cond=cond, body=self._update_right,
                                                            loop_vars=wrapped,
                                                            shape_invariants=shape_invariants,
                                                            parallel_iterations=10, name="rightSweep")
        self.updated_nodes = self.updated_nodes.write(to_index, n1)
        return self.updated_nodes

    def _armijo_loop(self, bond, C, lr, cost, delta_bond, gradient_dot_change):

        def _armijo_condition(learning_rate, updated_bond):
            f, updated_cost = self._get_f_and_cost(updated_bond, C)
            _gradient_dot_change = tf.Print(gradient_dot_change, [gradient_dot_change], first_n=self.verbosity,
                                  message="Gradient dot change")
            target = cost - self.armijo_coeff * learning_rate * _gradient_dot_change
            if self.verbosity != 0:
                target = tf.Print(target, [updated_cost, target, cost, f], first_n=self.verbosity,
                                  message="updated_cost, target, current cost, and f", summarize=10)
            return tf.greater(updated_cost, target)

        def _armijo_step(counter, armijo_cond, learning_rate, updated_bond):
            updated_bond = tf.add(bond, learning_rate * delta_bond)
            armijo_cond = _armijo_condition(learning_rate, updated_bond)
            updated_bond = tf.cond(armijo_cond, true_fn=lambda: bond, false_fn=lambda: updated_bond)
            return counter+1, armijo_cond, learning_rate * 0.5, updated_bond

        with tf.name_scope("armijo_loop"):
            cond = lambda c, f, lr, b: tf.logical_and(f, tf.less(c, 20))
            loop_vars = [1, True, lr, bond]
            _, _, lr, updated_bond = tf.while_loop(cond=cond, body=_armijo_step, loop_vars=loop_vars, name="lr_opt")

        return lr, updated_bond

    def _repeatedly_update_bond(self, bond, C):
        def _update_bond_once(n, bond, C, _):
            updated_bond, cost = self._update_bond(bond, C)
            return n+1, updated_bond, C, cost
        with tf.name_scope("update_bond_loop"):
            cond = lambda n, *args: tf.less(n, self.updates_per_step)
            loop_vars = [0, bond, C, 1.0]
            _, updated_bond, _, cost = tf.while_loop(cond=cond, body=_update_bond_once, loop_vars=loop_vars, name="bond_update")
        return updated_bond, cost

    def _make_new_nodes(self, nodes):
        """

        :param self:
        :param nodes:
        :return:
        """
        size = nodes.size()
        new_nodes = tf.TensorArray(tf.float32, size=size, infer_shape=False, clear_after_read=False)
        new_nodes = new_nodes.write(0, nodes.read(size - 1))
        new_nodes = new_nodes.write(size - 1, nodes.read(0))
        return new_nodes

    def _get_f_and_cost(self, bond, C):
        f, h = self._get_f_and_h(bond, C)
        cost = self.MPS.cost(f, self._label)
        if type(self.MPS) is sqMPS:
            h = f
        return h, cost

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

            from_index = tf.cond(tf.less(from_index, 0), lambda: 0, lambda: from_index)
            to_index = tf.cond(tf.greater(to_index, size), lambda: size, lambda: to_index)
            cond = lambda index, a, b: tf.less(index, to_index)

            _, _, new_nodes = tf.while_loop(cond=cond, body=self._transfer_to_array,
                                            loop_vars=[from_index, nodes, new_nodes], name="duplicate_loop")
        return new_nodes

    def _transfer_to_array(self, index, old_nodes, new_nodes):
        """

        :param self:
        :param index:
        :param old_nodes:
        :param new_nodes:
        :return:
        """
        old_node = old_nodes.read(index)
        new_nodes = new_nodes.write(index, old_node)
        index += 1
        return (index, old_nodes, new_nodes)
