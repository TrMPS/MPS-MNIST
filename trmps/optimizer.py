import tensorflow as tf
import time
import numpy as np
from mps import MPS
import pickle
import utils
from tensorflow.python.client import timeline
from parameterObjects import MPSOptimizerParameters, MPSTrainingParameters


def list_from(tensorArray, length):
    """
    list_from is a helper function that produces a list from a tensorArray.
    It is used to extract the results of training in MPSOptimizer.
    :param tensorArray: tensorflow TensorArray
        The tensor array that is to be converted to a list
    :param length: integer
        The length of the TensorArray/the list that is to be created
    :return: list of tensorflow Tensors
        A list containing all the values of the TensorArray as Tensors.
        This has to then be evaluated to get actual values.
    """
    arr = tensorArray
    result_list = []
    with tf.name_scope("createlist"):
        for i in range(length):
            result_list.append(arr.read(i))
    return result_list


class MPSOptimizer(object):
    """
    MPSOptimizer is used to optimize the MPS class for a dataset.
    Implements the DMRG method, as detailed in the paper
    Supervised learning with Quantum-Inspired Tensor Networks
    by E.Miles Stoudenmire and David J.Schwab
    Example usage using the MNIST dataset:

    from optimizer import *
    import MNISTpreprocessing

    # Model parameters
    d_feature = 2
    d_output = 10
    batch_size = 1000
    permuted = False
    shuffled = False
    shrink = True
    input_size = 784
    if shrink:
        input_size = 196

    max_size = 20

    rate_of_change = 10 ** (-7)
    logging_enabled = False

    cutoff = 10 # change this next
    n_step = 10

    data_source = MNISTpreprocessing.MNISTDatasource(shrink = shrink, permuted = permuted, shuffled = shuffled)

    weights = None

    network = MPS(d_feature, d_output, input_size)
    network.prepare(data_source)
    optimizer = MPSOptimizer(network, max_size, None, cutoff=cutoff)
    optimizer.train(data_source, batch_size, n_step,
                    rate_of_change=rate_of_change,
                    logging_enabled=logging_enabled,
                    initial_weights=weights)
    """

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
        self.MPS = MPSNetwork
        self.use_hessian = self.parameters.use_hessian
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
        _ = self.train_step()

        print("_____   Thomas the Tensor Train    . . . . . o o o o o",
              "  __|[_]|__ ___________ _______    ____      o",
              " |[] [] []| [] [] [] [] [_____(__  ][]]_n_n__][.",
              "_|________|_[_________]_[________]_|__|________)<",
              "  oo    oo 'oo      oo ' oo    oo 'oo 0000---oo\_",
              " ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", sep="\n")

    def train(self, data_source, batch_size, n_step, optional_parameters=MPSTrainingParameters()):
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
        test_result = list_from(self.updated_nodes, length=self.MPS.input_size)
        self.test = optional_parameters.initial_weights
        initial_lr = optional_parameters.rate_of_change

        train_cost, train_accuracy, train_confusion, _ = self._test_step(self._feature, self._label)

        test_feature, test_label = data_source.test_data

        feature = tf.placeholder(tf.float32, shape=[self.MPS.input_size, None, self.MPS.d_feature])
        label = tf.placeholder(tf.float32, shape=[None, self.MPS.d_output])
        test_cost, test_accuracy, test_confusion, test_f1 = self._test_step(feature, label)

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
                to_eval = [train_cost, test_result, train_accuracy, test_cost, test_accuracy, test_confusion, test_f1]
                train_c, self.test, train_acc, test_c, test_acc, test_conf, test_f1score = sess.run(to_eval,
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
                with open('weights', 'wb') as fp:
                    pickle.dump(self.test, fp)
                end = time.time()
                print('step {}, training cost {}, accuracy {}. Took {} s'.format(i, train_c, train_acc, end - start))
                print('step {}, testing cost {}, accuracy {}'.format(i, test_c, test_acc))
                print('f1 score: ', test_f1score)
                print('confusion matrix: \n' + str(test_conf))
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
        return cost, accuracy, confusion, f1

    def _setup_optimization(self):
        """
        :param self:
        :return:
        """
        '''
        C1s: size = input_size (as the last one is kept redundant)
        C2s: size = input_size (first one redundant)

        writes C1 from 0 to special_loc-1
        writes C2 from special_loc to size-3 (corresponds to special_loc+2 to size-1 the nodes)
        '''
        feature = self._feature
        nodes = self.MPS.nodes
        special_loc = self.MPS._special_node_loc
        batch_size = tf.shape(feature)[1]

        with tf.name_scope("setup_optimization"):


            C1 = tf.tile(self.MPS.start_node, [batch_size, 1])
            C1s = tf.TensorArray(tf.float32, size=self.MPS.input_size, infer_shape=False, clear_after_read=False)
            C1s = C1s.write(0, C1)
            cond = lambda c, *args: tf.less(c, special_loc)
            _, _, self.C1s = tf.while_loop(cond=cond, body=self._find_C1, loop_vars=[0, C1, C1s],
                                           shape_invariants=[tf.TensorShape([]), tf.TensorShape([None, None]),
                                                             tf.TensorShape(None)],
                                           parallel_iterations=10,
                                           name="initialFindC1")

            C2s = tf.TensorArray(tf.float32, size=self.MPS.input_size, infer_shape=False, clear_after_read=False)
            C2 = tf.tile(self.MPS.end_node, [batch_size, 1])
            C2s = C2s.write(self.MPS.input_size-1, C2)
            cond = lambda counter, *args: tf.greater(counter, special_loc+1)
            _, _, self.C2s = tf.while_loop(cond=cond, body=self._find_C2, loop_vars=[self.MPS.input_size-1, C2, C2s],
                                           shape_invariants=[tf.TensorShape([]), tf.TensorShape([None, None]),
                                                             tf.TensorShape(None)],
                                           parallel_iterations=10,
                                           name="initialFindC2")

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

        return accuracy

    def _sweep_left(self):
        """

        :param self:
        :return:
        """
        # read second from end node
        n1 = self.MPS.nodes.read(self.MPS._special_node_loc)
        n1.set_shape([self.MPS.d_output, self.MPS.d_feature, None, None])

        C2 = self.C2s.read(self.MPS.input_size - 1)
        self.C2s = tf.TensorArray(tf.float32, size=self.MPS.input_size, infer_shape=False, clear_after_read=False)
        self.C2s = self.C2s.write(self.MPS.input_size - 1, C2)
        cond = lambda counter, *args: tf.greater(counter, 0)
        wrapped = [self.MPS.input_size-1, self.acc_lr_reg, self.C2s, self.updated_nodes, n1]
        shape_invariants = [tf.TensorShape([]), tf.TensorShape([]),
                            tf.TensorShape(None), tf.TensorShape(None),
                            tf.TensorShape([None, None, None, None])]

        _, self.acc_lr_reg, self.C2s, self.updated_nodes, n1 = tf.while_loop(cond=cond, body=self._update_left,
                                                            loop_vars=wrapped,
                                                            shape_invariants=shape_invariants,
                                                            parallel_iterations=10,
                                                            name="leftSweep")
        self.updated_nodes = self.updated_nodes.write(0, n1)
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
        wrapped = [from_index, self.acc_lr_reg, self.C1s, self.updated_nodes, n1]
        shape_invariants = [tf.TensorShape([]), tf.TensorShape([]),
                            tf.TensorShape(None), tf.TensorShape(None),
                            tf.TensorShape([None, None, None, None])]
        _, self.acc_lr_reg, self.C1s, self.updated_nodes, n1 = tf.while_loop(cond=cond, body=self._update_right,
                                                            loop_vars=wrapped,
                                                            shape_invariants=shape_invariants,
                                                            parallel_iterations=10, name="rightSweep")
        self.updated_nodes = self.updated_nodes.write(to_index, n1)
        return self.updated_nodes

    def _update_left(self, counter, acc_lr_reg, C2s, updated_nodes, previous_node):
        """

        :param self:
        :param counter:
        :param C2s:
        :param updated_nodes:
        :param previous_node:
        :return:
        """

        with tf.name_scope("update_left"):
            # Read in the nodes
            n1 = previous_node
            n2 = self.MPS.nodes.read(counter - 1)
            n2.set_shape([self.MPS.d_feature, None, None])

            # Calculate the C matrix
            C2 = C2s.read(counter)
            C1 = self.C1s.read(counter - 1)
            C1.set_shape([None, None])
            C2.set_shape([None, None])
            input1 = self._feature[counter - 1]
            input2 = self._feature[counter]

            # Calculate the bond
            bond = tf.einsum('nkj,lmji->lmnik', n2, n1)

            C = self._calculate_C(C2, C1, input2, input1)

            # update the bond
            updated_bond = self._update_bond(bond, C)


            # Decompose the bond
            aj, aj1 = self._bond_decomposition(updated_bond, self.max_size)
            aj = tf.transpose(aj, perm=[0, 2, 1])
            aj1 = tf.transpose(aj1, perm=[1, 2, 3, 0])

            # Transpose the values and add to the new variables
            updated_nodes = updated_nodes.write(counter, aj)
            with tf.name_scope("tensordotcontracted_aj"):
                # contracted_aj = tf.einsum('mij,tm->tij', aj, self._feature[counter])
                contracted_aj = tf.tensordot(self._feature[counter], aj, [[1], [0]])
            with tf.name_scope("einsumC2"):
                C2 = tf.einsum('tij,tj->ti', contracted_aj, C2)
            C2s = C2s.write(counter - 1, C2)
            #counter = tf.Print(counter, [counter])
            updated_counter = counter - 1
            acc_lr_reg = acc_lr_reg * self.lr_reg



        return [updated_counter, acc_lr_reg, C2s, updated_nodes, aj1]

    def _update_right(self, counter, acc_lr_reg, C1s, updated_nodes, previous_node):
        """

        :param self:
        :param counter:
        :param C1s:
        :param updated_nodes:
        :param previous_node:
        :return:
        """
        with tf.name_scope("update_right"):
            # Read in the nodes
            n1 = previous_node
            n2 = self.MPS.nodes.read(counter + 1)
            n2.set_shape([self.MPS.d_feature, None, None])

            # Calculate the C matrix
            C2 = self.C2s.read(counter+1)
            C1 = C1s.read(counter)
            C1.set_shape([None, None])
            C2.set_shape([None, None])
            input1 = self._feature[counter]
            input2 = self._feature[counter + 1]

            # Calculate the bond
            bond = tf.einsum('lmij,njk->lmnik', n1, n2)
            # bond = tf.transpose(tf.tensordot(n1, n2, [[3],[1]]), [0, 1, 3, 2, 4])
            # einsum is actually faster in this case

            C = self._calculate_C(C1, C2, input1, input2)

            # Update the bond
            updated_bond = self._update_bond(bond, C)

            # Decompose the bond
            aj, aj1 = self._bond_decomposition(updated_bond, self.max_size)
            aj1 = tf.transpose(aj1, perm=[1, 2, 0, 3])

            # Transpose the values and add to the new variables
            updated_nodes = updated_nodes.write(counter, aj)

            with tf.name_scope("tensordotcontracted_aj"):
                # contracted_aj = tf.einsum('mij,tm->tij', aj, self._feature[counter])
                contracted_aj = tf.tensordot(self._feature[counter], aj, [[1], [0]])
            with tf.name_scope("einsumC1"):
                C1 = tf.einsum('tij,ti->tj', contracted_aj, C1)
            C1s = C1s.write(counter+1, C1)
            #counter = tf.Print(counter, [counter])
            updated_counter = counter + 1
            acc_lr_reg = acc_lr_reg * self.lr_reg

        return [updated_counter, acc_lr_reg, C1s, updated_nodes, aj1]

    def _calculate_C(self, C1, C2, input1, input2):
        """

        :param self:
        :param C1:
        :param C2:
        :param input1:
        :param input2:
        :return:
        """
        # C = tf.einsum('ti,tk,tm,tn->tmnik', C1, C2, input1, input2)
        d1 = tf.shape(C1)[1]
        d2 = tf.shape(C2)[1]

        with tf.name_scope("calculateC"):
            C1 = tf.reshape(C1, [self.batch_size, 1, 1, d1, 1])
            C2 = tf.reshape(C2, [self.batch_size, 1, 1, 1, d2])
            input1 = tf.reshape(input1, [self.batch_size, self.MPS.d_feature, 1, 1, 1])
            input2 = tf.reshape(input2, [self.batch_size, 1, self.MPS.d_feature, 1, 1])
            intermediate_1 = C1 * C2
            intermediate_2 = input1 * input2
            C = intermediate_1 * intermediate_2

        return C

    def _get_f_and_cost(self, bond, C):
        """

        :param self:
        :param bond:
        :param C:
        :return:
        """
        with tf.name_scope("tensordotf"):
            # f = tf.einsum('lmnik,tmnik->tl', bond, C)
            f = tf.tensordot(C, bond, [[1, 2, 3, 4], [1, 2, 3, 4]])
            h = tf.nn.softmax(f)
        with tf.name_scope("reduce_sumcost"):
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self._label, logits=f))
            # 0.5 * tf.reduce_sum(tf.square(f-self._label))

        return h, cost

    def _calculate_hessian(self, f, C):
        with tf.name_scope('hessian'):
            d1 = tf.shape(C)[-2]
            d2 = tf.shape(C)[-1]
            f_part = tf.reshape(f * (1 - f), [self.batch_size, self.MPS.d_output, 1, 1, 1, 1])
            C_sq = tf.reshape(tf.square(C), [self.batch_size, 1, self.MPS.d_feature, self.MPS.d_feature, d1, d2])
            hessian = tf.reduce_sum(f_part * C_sq, axis=0) + 2 * self.reg

            return hessian

    def _armijo_loop(self, bond, C, lr, cost, delta_bond, gradient_dot_change):

        def _armijo_condition(learning_rate, updated_bond):
            _, updated_cost = self._get_f_and_cost(updated_bond, C)
            target = cost - self.armijo_coeff * learning_rate * gradient_dot_change
            if self.verbosity != 0:
                target = tf.Print(target, [updated_cost, target, cost], first_n=self.verbosity,
                                  message = "updated_cost, target and cost")
            return tf.greater(updated_cost, target)

        def _armijo_step(counter, armijo_cond, learning_rate, updated_bond):
            updated_bond = tf.add(bond, learning_rate * delta_bond)
            armijo_cond = _armijo_condition(learning_rate, updated_bond)
            updated_bond = tf.cond(armijo_cond, true_fn=lambda: bond, false_fn=lambda: updated_bond)
            return counter+1, armijo_cond, learning_rate * 0.5, updated_bond

        with tf.name_scope("armijo_loop"):
            cond = lambda c, f, lr, b: tf.logical_and(f, tf.less(c, 10))
            loop_vars = [1, True, lr, bond]
            _, _, lr, updated_bond = tf.while_loop(cond=cond, body=_armijo_step, loop_vars=loop_vars, name="lr_opt")

        return lr, updated_bond

    def _update_bond(self, bond, C):
        # obtain the original cost
        # bond = tf.Print(bond, [counter, tf.shape(bond)])
        f, cost = self._get_f_and_cost(bond, C)
        h = 1.0
        if self.use_hessian:
            h = self._calculate_hessian(f, C)

        # perform gradient descent on the bond
        with tf.name_scope("tensordotgradient"):
            gradient = tf.tensordot(self._label - f, C, [[0], [0]]) - 2 * self.reg * bond
            delta_bond = gradient / h
        gradient_dot_change = tf.tensordot(gradient,
                                           delta_bond,
                                           [[0, 1, 2, 3, 4],[0, 1, 2, 3, 4]])/tf.cast(self.batch_size, tf.float32)
        lr = self.rate_of_change
        lr, updated_bond = self._armijo_loop(bond, C, lr, cost, delta_bond, gradient_dot_change)

        _, cost1 = self._get_f_and_cost(updated_bond, C)
        if self.verbosity != 0:
            updated_bond = tf.Print(updated_bond, [cost1], message='updated cost', first_n=self.verbosity)
        cond_change_bond = tf.less(cost1, cost)
        updated_bond = tf.cond(cond_change_bond, true_fn=(lambda: updated_bond),
                               false_fn=(lambda: tf.Print(bond, [cost, cost1], message='Gradient may be too big/too small')))

        return updated_bond

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

    def _bond_decomposition(self, bond, max_size, min_size=3):
        """

        :param self:
        :param bond:
        :param max_size:
        :param min_size:
        :param min_singular_value:
        :return:
        """
        """
        Decomposes bond, so that the next step can be done.
        :param bond:
        :param m:
        :return:
        """
        with tf.name_scope("bond_decomposition"):
            bond_reshaped = tf.transpose(bond, perm=[1, 3, 0, 2, 4])
            # bond_reshaped = tf.Print(bond_reshaped, [tf.shape(bond_reshaped), tf.shape(bond)], summarize = 1000, message = "bond reshaped, bond")

            dims = tf.shape(bond_reshaped)
            l_dim = dims[0] * dims[1]
            r_dim = dims[2] * dims[3] * dims[4]
            bond_flattened = tf.reshape(bond_reshaped, [l_dim, r_dim])
            s, u, v = tf.svd(bond_flattened)
            filtered_u = utils.check_nan(u, 'u', replace_nan=True)
            filtered_v = utils.check_nan(v, 'v', replace_nan=True)

            filtered_s = tf.boolean_mask(s, tf.greater(s, self.min_singular_value))
            s_size = tf.size(filtered_s)
            # s_size = tf.Print(s_size, [s_size], message='bond dim: ')

            case1 = lambda: min_size
            case2 = lambda: max_size
            case3 = lambda: s_size
            m = tf.case({tf.less(s_size, min_size): case1, tf.greater(s_size, max_size): case2}, default=case3,
                        exclusive=True)
            # if self.verbosity != 0:
            #     m = tf.Print(m, [m, s[m-5:m]], first_n=self.verbosity, summarize=5, message='bond: ')

            # make s into a matrix
            s_mat = tf.diag(s[0:m])

            # make u, v into suitable matrices
            u_cropped = filtered_u[:, 0:m]
            v_cropped = tf.transpose(filtered_v[:, 0:m])

            # make a_
            a_prime_j = tf.reshape(u_cropped, [dims[0], dims[1], m])

            sv = tf.matmul(s_mat, v_cropped)
            a_prime_j1 = tf.reshape(sv, [m, dims[2], dims[3], dims[4]])
            # a_prime_j1 = tf.transpose(a_prime_j1_mixed, perm=[1, 2, 0, 3])
            # will do this in the update_right/update_left functions from now on as else transpose twice for udpate_left

        return (a_prime_j, a_prime_j1)
