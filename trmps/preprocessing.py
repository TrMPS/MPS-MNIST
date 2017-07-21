import tensorflow as tf
import numpy as np
import sys
import os

class MPSDatasource(object):
    """
	MPSDatasource offers the interfaces for the MPS class and the MPSOptimiser class to get data.
	Can be used directly by using passing in the training data and test data, or can be subclassed
	to implement custom loading behaviour. If subclassing, should implement
	_load_test_data and _load_train_data
	which will be called at appropriate timings in the datasource's lifecycle.
    """
    def __init__(self, _expected_shape = None, shuffled = False, _training_data = None, _test_data = None):
        """
		Creates a newly initialised datasource with the specified training or test data.
		If _load_test_data and _load_training_data are implemented, they are called as well,
		and saving/loading is handled there.
		
        :param _expected_shape: tuple 
        	The expected shape of an individual element of training data
        :param shuffled: boolean
        	Pass true to shuffle the dataset.
        :param _training_data: (numpy array, numpy array)
        	The training data for the MPS object, in the shape 
        	([batch, MPS input size, other dimensions], [batch, classifications])
        :param _test_data: (numpy array, numpy array)
        	The test data for the MPS object, in the shape 
        	([batch, MPS input size, other dimensions], [batch, classifications])
        """
        self._training_data = _training_data
        self._test_data = _test_data
        self._training_data_path = "training_data" + type(self).__name__ + ".npy"
        self._training_labels_path = "training_labels" + type(self).__name__ + ".npy"
        self._expected_shape = _expected_shape
        if self._training_data is None:
	        if os.path.isfile(self._training_data_path):
	            self._training_data = (np.load(self._training_data_path), np.load(self._training_labels_path))
	            print(self._training_data[0][0].shape)
	            print(self._expected_shape)
	            if self._training_data[0][0].shape != self._expected_shape:
	                self._training_data = None
        self._test_data_path = "testing_data" + type(self).__name__ + ".npy"
        self._test_labels_path = "testing_labels" + type(self).__name__ + ".npy"
        if self._test_data is None:
	        if os.path.isfile(self._test_data_path):
	            self._test_data = (np.load(self._test_data_path), np.load(self._test_labels_path))
	            if self._test_data[0][0].shape != self._expected_shape:
	                self._test_data = None
        self.current_index = 0
        if self._test_data is None:
            self._load_test_data()
        if self._training_data is None:
            self._load_training_data()
        if shuffled:
            data, labels = self._training_data
            permutation = np.random.permutation(len(data))
            self._training_data = data[permutation], labels[permutation]
        self._swapped_test_data = None
        self._swapped_training_data = None
        
    
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
            self._swapped_test_data = np.swapaxes(self._test_data[0], 0, 1)
        return self._swapped_test_data, self._test_data[1]
    
    def _load_test_data(self):
        """
        Get test data (perhaps from remote server) and preprocess in shape [batch, expected shape of element].
        Remember to call this from a subclass to save the things.
        :return: nothing
        """
        np.save(self._test_data_path, self._test_data[0])
        np.save(self._test_labels_path, self._test_data[1])
        
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
            self._swapped_training_data = np.swapaxes(self._training_data[0], 0, 1)
        return self._swapped_training_data, self._training_data[1]
        
    def _load_training_data(self):
        """
        Get training data (perhaps from remote server) and preprocess in shape [batch, expected shape of element]
        Remember to call this from a subclass to save the things.
        :return: nothing
        """
        np.save(self._training_data_path, self._training_data[0])
        np.save(self._training_labels_path, self._training_data[1])

        
    def next_training_data_batch(self, batch_size):
        """
		Gets the next n data and label points from the training dataset,
		where n = batch_size
		
        :param batch_size: integer
        	The number of data and label points to return
        :return: (numpy array, numpy array)
	        The training data, labels in the form 
	        ([MPS input size, batch, other dimensions], [batch, classifications])
        """
        if self._training_data == None:
            self._load_test_data()
        all_data, all_labels = self._training_data
        if batch_size > len(all_data):
            print("Probably shouldn't do this; your batch size is greater than the size of the dataset")
        data = None
        labels = None
        while batch_size > 0:
            if len(all_data) - self.current_index < batch_size:
                # print("A" + str(self.current_index))
                batch_size -= (len(all_data) - self.current_index)
                if self.current_index != len(all_data):
                    if data is None:
                        data = np.array(all_data[self.current_index:])
                        labels = np.array(all_labels[self.current_index:])
                    else:
                        data = np.concatenate((data, all_data[self.current_index:]), axis=0)
                        labels = np.concatenate((labels, all_labels[self.current_index:]), axis=0)
                self.current_index = 0
            else:
                # print("B" + str(self.current_index))
                if data is None:
                    data = all_data[self.current_index:self.current_index + batch_size]
                    labels = np.array(all_labels[self.current_index:self.current_index + batch_size])
                else:
                    data = np.concatenate((data, all_data[self.current_index:self.current_index + batch_size]), axis=0)
                    labels = np.concatenate((labels, all_labels[self.current_index:self.current_index + batch_size]),
                                            axis=0)
                self.current_index += batch_size
                batch_size = 0
        data = np.array(data)
        data = np.swapaxes(data, 0, 1)
        return (data, labels)
        
    @property
    def num_train_samples(self):
        """
		Property giving the number of training samples
		
        :return: integer
        	The number of training samples
        """
        return len(self._training_data[0])
        
    @property
    def num_test_samples(self):
        """
		Property giving the number of test samples
		
        :return: integer
        	The number of test samples
        """
        return len(self._test_data[0])