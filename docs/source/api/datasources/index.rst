Datasources
====================================

Introduction to datasources for MPS
--------------------------

To train an MPS, use MPSDatasource or a subclass of it. MPSDatasource offers the interfaces for the MPS class and the MPSOptimizer class to get data. It can be used directly by using passing in the training data and test data, or can be subclassed to implement custom loading behaviour. If subclassing, should implement _load_test_data and _load_train_data which will be called at appropriate timings in the datasource's lifecycle. Use as follows:

.. code-block:: python

  from preprocessing import MPSDatasource

A number of subclasses have been implemented for different datasources, which each implement the same API as MPSDatasource, and so can be fed directly into an MPS or MPSOptimizer class. The implemented ones are:

* The `MNIST dataset <http://yann.lecun.com/exdb/mnist/>`_, implemented as MNISTDatasource in MNISTpreprocessing.
* The `activity dataset <http://ps.ewi.utwente.nl/Datasets.php>`_, a dataset of accelerometer data of people walking, running, sitting, standing, jogging, biking and walking upstairs/downstairs. Implemented as activityDatasource in activitypreprocessing. This dataset requires the modules unrar and rarfile.
* The `cardiogram dataset <https://physionet.org/challenge/2017/training2017.zip>`_.
* The `Movie review dataset <http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz>`_.

In each case, the labels are stored as one-hot vectors. The API reference for MPSDatasource is as follows:

Properties
--------

test_data (read only)
^^^^^^^^^
 The test data. Loads the test data if it doesn't exist. Note that this the test data and labels are in the form ([MPS input size, batch, other dimensions], [batch, classifications]) in accordance with how it is used in the MPS and MPSOptimizer classes. If the data is required in the form ([batch, MPS input size, other dimensions], [batch, classifications]), the property _test_data should be used.

training_data (read only)
^^^^^^
 The training data. Loads the training data if it doesn't exist. Note that this the test data and labels are in the form ([MPS input size, batch, other dimensions], [batch, classifications]) in accordance with how it is used in the MPS and MPSOptimizer classes. If the data is required in the form ([batch, MPS input size, other dimensions], [batch, classifications]), the property _test_data should be used.

num_test_samples (read only)
^^^^^
 The number of test samples

num_train_samples (read only)
^^^^^
 The number of training samples.

Creating a new datasource
---------

__init__(_expected_shape = None, shuffled = False, _training_data = None, _test_data = None)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  Creates a newly initialised datasource with the specified training or test data. If _load_test_data and _load_training_data are implemented, they are called as well, and saving/loading is handled there.

  *_expected_shape: tuple*
    The expected shape of an individual element of training data
  *shuffled: boolean*
    Pass true to shuffle the dataset.
  *_training_data: (numpy array, numpy array)*
    The training data for the MPS object, in the shape
    ([batch, MPS input size, other dimensions], [batch, classifications])
  *_test_data: (numpy array, numpy array)*
    The test data for the MPS object, in the shape
    ([batch, MPS input size, other dimensions], [batch, classifications])

Getting data for training
----------

next_training_data_batch(batch_size)
^^^^^^^^^^^^
  Gets the next n data and label points from the training dataset, where n = batch_size.

  *batch_size: integer*
    The number of data and label points to return
  *returns: (numpy array, numpy array)*
    The training data, labels in the form ([MPS input size, batch, other dimensions], [batch, classifications])

Creating a subclass
------------
If you want to work with your own dataset, outside of the ones provided, it will most likely be easiest to subclass MPSDatasource. For concrete examples, look at how e.g. activityDatasource is implemented. In general, however, the following functions should be implemented, which will then be called at appropriate times in the object's lifecycle.

_load_test_data()
^^^^^^^^^^
 Get test data (perhaps from remote server) and preprocess into shape [batch, expected shape of element]. self._test_data[0] should then be set to the testing data, and self._test_data[1] the testing labels. The superclass' _load_test_data should then be called, which will save the loaded data.

_load_training_data()
^^^^^^^^^^^^^
 Get training data (perhaps from remote server) and preprocess into shape [batch, expected shape of element]. self._training_data[0] should then be set to the training data, and self._training_data[1] the training labels. The superclass' _load_training_data should then be called, which will save the loaded data.


