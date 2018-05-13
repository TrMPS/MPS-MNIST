import tensorflow as tf
import numpy as np
import sys
import os

#TODO: Implement shuffling
#TODO: _available_training_lengths can be implemented more elegantly as a property

class UMPSDatasource(object):

    """
    MPSDatasource offers the interfaces for the MPS class and the MPSOptimiser class to get data.
    Can be used directly by using passing in the training data and test data, or can be subclassed
    to implement custom loading behaviour. If subclassing, should implement
    _load_test_data and _load_train_data
    which will be called at appropriate timings in the datasource's lifecycle.
    """
    def __init__(self, _expected_d_input=None, shuffled=False, _training_data=None, _test_data=None):
        """
        Creates a newly initialised datasource with the specified training or test data.
        If _load_test_data and _load_training_data are implemented, they are called as well,
        and saving/loading is handled there.

        :param _expected_shape: tuple
            The expected shape of an individual element of training data
        :param shuffled: boolean
            Pass true to shuffle the dataset.
        :param _training_data: {int:(numpy array, numpy array)}
            The training data for the MPS object, in the shape
            {length:([batch, MPS input size, other dimensions], [batch, classifications])}
        :param _test_data: {int:(numpy array, numpy array)}
            The test data for the MPS object, in the shape
            {length:([batch, MPS input size, other dimensions], [batch, classifications])}
        """
        self._training_data = _training_data
        self._test_data = _test_data
        self._num_training_samples = None
        self._num_test_samples = None
        self._available_training_lengths = []
        self._available_test_lengths = []
        self._training_data_path = os.path.join(type(self).__name__, "training_data.npy")
        if not os.path.isdir(type(self).__name__):
            os.mkdir(type(self).__name__)

        self._expected_d_input = _expected_d_input
        if self._training_data is None:
            if os.path.isfile(self._training_data_path):
                self._training_data = np.load(self._training_data_path).item()
                for _, value in self._training_data.items():
                    print(value[0].shape)
                    print(_expected_d_input)
                    if value[0].shape[2] != _expected_d_input:
                        self._training_data = None
                    break
        self._test_data_path = os.path.join(type(self).__name__, "testing_data.npy")
        if self._test_data is None:
            if os.path.isfile(self._test_data_path):
                self._test_data = np.load(self._test_data_path).item()
                for _, value in self._test_data.items():
                    if value[0].shape[2] != _expected_d_input:
                        self._test_data = None
                    break
        if self._test_data is None:
            self._load_test_data()
        if self._training_data is None:
            self._load_training_data()
        if shuffled:
            print("Shuffling not supported at this point!")
        self.current_index = {}
        for key, _ in self._training_data.items():
            self.current_index[key] = 0
        self._initialise_available_training_lengths()
        self._initialise_available_test_lengths()
        self._swapped_test_data = None
        self._swapped_training_data = None

    def _initialise_available_training_lengths(self):
        for key, _ in self._training_data.items():
            self._available_training_lengths.append(key)
        self._available_training_lengths = np.array(self._available_training_lengths)

    def _initialise_available_test_lengths(self):
        for key, _ in self._test_data.items():
            self._available_test_lengths.append(key)
        self._available_test_lengths = np.array(self._available_test_lengths)


    @property
    def test_data(self):
        """
        This property returns the test data, and loads the test data if it doesn't exist.
        Note that this function returns the test data and labels in the form
        ([MPS input size, batch, other dimensions], [batch, classifications])
        in accordance with how it is used in the MPS and MPSOptimizer classes.

        If the data is required in the form
        ([batch, MPS input size, other dimensions], [batch, classifications]),
        the variable _test_data should be used

        :return: (numpy array, numpy array)
            The test data, labels in the form
            ([MPS input size, batch, other dimensions], [batch, classifications])
        """
        if self._test_data is None:
            self._load_test_data()
        if self._swapped_test_data is None:
            self._swapped_test_data = {}
            for key, value in self._test_data.items():
                self._swapped_test_data[key] = value
        return self._swapped_test_data

    def _load_test_data(self):
        """
        Get test data (perhaps from remote server) and preprocess in shape [batch, expected shape of element].
        Remember to call this from a subclass to save the things.
        :return: nothing
        """
        self._save_test_data()

    def _save_test_data(self):
        np.save(self._test_data_path, self._test_data)

    @property
    def training_data(self):
        """
        This property returns the training data, and loads the training data if it doesn't exist.
        Note that this function returns the training data and labels in the form
        ([MPS input size, batch, other dimensions], [batch, classifications])
        in accordance with how it is used in the MPS and MPSOptimizer classes.

        If the data is required in the form
        ([batch, MPS input size, other dimensions], [batch, classifications]),
        the variable _training_data should be used

        :return: (numpy array, numpy array)
            The training data, labels in the form
            ([MPS input size, batch, other dimensions], [batch, classifications])

        """
        if self._training_data is None:
            self._load_training_data()
        if self._swapped_training_data is None:
            self._swapped_training_data = {}
            for key, value in self._training_data.items():
                self._swapped_training_data[key] = value
        return self._swapped_training_data

    def _load_training_data(self):
        """
        Get training data (perhaps from remote server) and preprocess in shape [batch, expected shape of element]
        Remember to call this from a subclass to save the things.
        :return: nothing
        """
        self._save_training_data()

    def _save_training_data(self):
        np.save(self._training_data_path, self._training_data)

    def shuffle(self):
        print("Shuffling isn't supported yet, please don't call rmpsdatasource.shuffle()")

    def test_data_of_length(self, expected_length):
        if self._test_data is None:
            self._load_test_data()
        if expected_length in self._test_data:
            actual_length = expected_length
        else:
            differences = np.abs(self._available_test_lengths - expected_length)
            mininimum_loc = np.argmin(differences)
            actual_length = self._available_test_lengths[mininimum_loc]
        data, labels = self._test_data[actual_length]
        data = np.array(data)
        data = np.swapaxes(data, 0, 1)

        return (actual_length, (data, labels))

    def next_training_data_batch(self, batch_size, expected_length):
        """
        Gets the next n data and label points from the training dataset,
        where n = batch_size
        length is the expected length of the inputs
        :param batch_size: integer
            The number of data and label points to return
        :return: (numpy array, numpy array)
            The training data, labels in the form
            ([MPS input size, batch, other dimensions], [batch, classifications])
        """
        if self._training_data is None:
            self._load_training_data()
        if expected_length in self._training_data:
            actual_length = expected_length
        else:
            differences = np.abs(self._available_training_lengths - expected_length)
            mininimum_loc = np.argmin(differences)
            actual_length = self._available_training_lengths[mininimum_loc]
        all_data, all_labels = self._training_data[actual_length]
        if batch_size > len(all_data):
            print("Probably shouldn't do this; your batch size is greater than the size of the dataset")
        data = None
        labels = None
        while batch_size > 0:
            if len(all_data) - self.current_index[actual_length] < batch_size:
                # print("A" + str(self.current_index))
                batch_size -= (len(all_data) - self.current_index[actual_length])
                if self.current_index[actual_length] != len(all_data):
                    if data is None:
                        data = np.array(all_data[self.current_index[actual_length]:])
                        labels = np.array(all_labels[self.current_index[actual_length]:])
                    else:
                        data = np.concatenate((data, all_data[self.current_index[actual_length]:]), axis=0)
                        labels = np.concatenate((labels, all_labels[self.current_index[actual_length]:]), axis=0)
                self.current_index[actual_length] = 0
            else:
                # print("B" + str(self.current_index))
                if data is None:
                    data = all_data[self.current_index[actual_length]:self.current_index[actual_length] + batch_size]
                    labels = np.array(all_labels[self.current_index[actual_length]:self.current_index[actual_length] + batch_size])
                else:
                    data = np.concatenate((data, all_data[self.current_index[actual_length]:self.current_index[actual_length] + batch_size]), axis=0)
                    labels = np.concatenate((labels, all_labels[self.current_index[actual_length]:self.current_index[actual_length] + batch_size]),
                                            axis=0)
                self.current_index[actual_length] += batch_size
                batch_size = 0
        data = np.array(data)
        data = np.swapaxes(data, 0, 1)
        return (actual_length, (data, labels))

    @property
    def num_train_samples(self):
        """
        Property giving the number of training samples for each key

        :return: integer
            The number of training samples
        """
        if self._num_training_samples is None:
            self._num_training_samples = {}
            for key, value in self._training_data.items():
                self._num_training_samples[key] = len(value[0])
        return self._num_training_samples

    @property
    def num_test_samples(self):
        """
        Property giving the number of test samples

        :return: integer
            The number of test samples
        """
        if self._num_test_samples is None:
            self._num_training_samples = {}
            for key, value in self._test_data.items():
                self._num_test_samples[key] = len(value[0])
        return self._num_test_samples
