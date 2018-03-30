MPS
==========

Introduction to the MPS class
--------------------------

MPS represents a 'Matrix Product State', which can be optimised (using MPSOptimizer) to create a model which performs well for certain machine learning tasks, and the implementation generally follows the paper Supervised learning with Quantum-Inspired Tensor Networks by E.Miles Stoudenmire and David J.Schwab. However, there are some key differences, including that the cost function in this case is the cross entropy. For a version that implements the squared error cost function, see the subclass sqMPS. The class can also be used on its own to perform inference.

.. code-block:: python

    from trmps import *
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

This creates an MPS, loads in some pre-trained weights, then performs some inference, for the accuracy is then printed out.

The SGDMPS subclass should be used when the MPS is goign to be optimized with stochastic gradient descent.

Subclasses
-------

.. toctree::
   :maxdepth: 1

   sqMPS
   SGDMPS

Properties
---------

*input_size: int (read only)*
 The input size, i.e. the number of matrices composing the matrix product state.
*d_matrix: int (read only)*
 The initial sizes of the matrices
*d_feature: int (read only)*
 The sizes of the feature vectors. (Referred to as 'Local dimension' in the paper). Try to keep this low (if possible), as the optimisation algorithm scales as (d_feature)^3.
*d_output: int (read only)*
 The size of the output. e.g. with 10-class classification, expressed as a one-hot vector, this would be 10
*nodes: tf.TensorArray (read only)*
 A TensorArray containing all of the weights for this model.
*feature: tf.Placeholder of shape (input_size, batch_size, d_feature)*
 The features that will be evaluated.

Creating a new MPS
-----------
__init__(d_feature, d_output, input_size, special_node_loc=None)
^^^^^^^^
 Initialises the MPS. The prepare method must be called after this before  anything else can be done.

 *d_feature: integer*
  The sizes of the feature vectors. (Referred to as 'Local dimension' in the  paper). Try to keep this low (if possible), as the optimisation algorithm  scales as (d_feature)^3.

 *d_output: integer*
  The size of the output. e.g. with 10-class classification, expressed as a  one-hot vector, this would be 10.

 *input_size: int*
  The input size, i.e. the number of matrices composing the matrix product  state.

 *special_node_loc: int or None*
  The location of the "special node", i.e. the tensor which has the extra index  on it, which is where we obtain the prediction from. If loading in weights,  make sure that this location coincides with where the special node location  is in the weights you are loading in. If this value is set to None, then if  in the prepare method, no data_source is passed in, the special node location  is set automatically to be at the middle of the MPS. If this value is set to  None and a data_source is passed into the prepare method, then the special  node location is set to be the location where the weight from linear  regression was largest.

from_file(path="MPSconfig")
^^^^^^^^^
 Initialises the MPS from the file at the path specified. If the file only contains the configuration of the MPS, and not the weights, the prepare method must be called after this before anything else can be done. More information on the file format can be found `here <https://github.com/TrMPS/MPS-MNIST/blob/master/Notes%20for%20developement.md>`_.

 *path: string*
  Specifies the path where the MPS configuration is stored.

prepare(data_source=None, iterations=1000, learning_rate=0.05)
^^^^^^^^
 Prepares the MPS. Optionally uses linear regression, which can be thought of  as pre-training the network, and dramatically shortens the required training  time. This function must be called after the initialiser, before anything can  be done with the MPS.

 *data_source: (some subclass of) MPSDatasource, or None*
  The data/labels that the MPS will be trained on. If this is set to None, no  pre-training is performed, and so the predictions start out essentialy being  random.

 *iterations: integer*
  The number of iterations for which the linear regression model is trained. If  the data_source is None, this is value is ignored.

 *learning_rate: float*
  The learning rate to use when training with linear regression. If the  data_source is None, this is value is ignored.

Using the MPS
---------
create_feed_dict(weights)
^^^^^^^^
 Creates a feed_dict which assigns the given weights to the MPS' nodes. This  should be used whenever you want to update the weights of the MPS, and when  sess.run() in tensorflow is used to perform any actions with the MPS, this  feed_dict should be used as the feed_dict. An example of its usage is as  follows:

 .. code-block:: python

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

 *weights: list of numpy arrays of length input_length*
  The weights that you want to use for the MPS.

 *returns: a dictionary*
  Edit this dictionary for any other things you want to pass in the feed_dict.

predict(feature)
^^^^^^^^^
 A tensorflow operation that takes predictions based on the features. Can do batch prediction.

 *feature: tensorflow Tensor of shape (input_size, batch_size, d_feature)*
  The features for which the predictions are to be made.

 *returns: tensorflow Tensor of shape (batch_size, d_output)*
  The predictions


Testing the MPS
---------
test(test_feature, test_label)
^^^^^^^^
 A function to test the MPS. This function creates a tensorflow session and performs some testing of the MPS, using the current MPS. At the end of testing, it prints out the cost, accuracy, and a sample prediction. Does not support passing in pre-trained weights, so can only be used to test an MPS as it has been initialised. This will be amended in a future update. In the mean time, use the other functions in this section, which make tensorflow operations, to test certain aspects of the MPS.

 *test_feature: a numpy array of type float32 of shape (input_size, batch_size, d_feature)*
  The features for which the testing is to be done.

 *test_label: a numpy array of shape (batch_size, d_output)*
  The 'correct' labels against which the predictions from the test_feature will be judged.


accuracy(f, labels)
^^^^^^^^^
 Computes the accuracy given the predictions(f), and the correct labels.

 *f: tensorflow Tensor of shape (batch_size, d_output)*
  The predictions that are to be judged. Usually found using the predict  function.
 *labels: tensorflow Tensor of shape (batch_size, d_output)*
  The correct labels.
 *returns: a tensorflow scalar*
  The accuracy of the predictions

cost(f, labels)
^^^^^^^^^
 Computes the cost (softmax cross entropy with logits) given the predictions(f), and the correct labels.

 *f: tensorflow Tensor of shape (batch_size, d_output)*
  The predictions that are to be judged.
 *labels: tensorflow Tensor of shape (batch_size, d_output)*
  The correct labels.
 *returns: a tensorflow scalar*
  The cost of the predictions, as judged by softmax cross entropy with logits.

confusion_matrix(f, labels)
^^^^^^^
 Computes the confusion matrix given the predictions(f), and the correct labels.

 *f: tensorflow Tensor of shape (batch_size, d_output)*
  The predictions that are to be judged.
 *labels: tensorflow Tensor of shape (batch_size, d_output)*
  The correct labels.
 *returns: a tensorflow Tensor of shape (d_output, d_output)*
  The confusion matrix.

f1score(f, labels, _confusion_matrix=None)
^^^^^^^^
 Computes the `F1 score <https://en.wikipedia.org/wiki/F1_score>`_ for the predictions(f) given the correct labels.

 *f: tensorflow Tensor of shape (batch_size, d_output)*
  The predictions that are to be judged.
 *labels: tensorflow Tensor of shape (batch_size, d_output)*
  The correct labels.
 *_confusion_matrix: a tensorflow Tensor of shape (d_output, d_output), or None*
  If the confusion matrix has already been evaluated elsewhere for other reasons, it can be passed in via this parameter, so that it does not have to be evaluated again. If passing in a confusion matrix, then f and labels are never used.

Storing the MPS
---------

save(weights=None, path="MPSconfig")
^^^^^^^^^
 Saves the MPS configuration at the path given. If weights are given, then the weights will be stored as well. A new MPS can be created from the saved weights using from_file.

 *weights: list of numpy arrays*
  A list containing the trained weights of the MPS that are to be saved. If no value is passed in, only the configuration of the MPS will be stored.
 *path: string*
  Specifies the path where the MPS configuration is stored.



