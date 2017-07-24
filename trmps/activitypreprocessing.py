import tensorflow as tf
import numpy as np
import sys
import os
from preprocessing import *
import urllib.request
from rarfile import RarFile
import csv
from utils import spinner
from enum import Enum

# Adapted from https://stackoverflow.com/questions/29831489/numpy-1-hot-array
def convert_to_onehot(vector, num_classes=None):
    """
    Converts an input 1-D vector of integers into an output
    2-D array of one-hot vectors, where an i'th input value
    of j will set a '1' in the i'th row, j'th column of the
    output array.

    Example:
        v = np.array((1, 0, 4))
        one_hot_v = convertToOneHot(v)
        print one_hot_v

        [[0 1 0 0 0]
         [1 0 0 0 0]
         [0 0 0 0 1]]
    """
    assert isinstance(vector, np.ndarray)
    assert len(vector) > 0

    if num_classes is None:
        num_classes = np.max(vector)+1
    else:
        assert num_classes > 0
        assert num_classes >= np.max(vector)

    result = np.zeros(shape=(len(vector), num_classes))
    result[np.arange(len(vector)), vector] = 1
    return result.astype(int)
    
class activityLabels(Enum):
    """
    An enum to hold the different types of activity available in the activity dataset:
    http://ps.ewi.utwente.nl/Blog/Sensors_Activity_Recognition_DataSet_Shoaib.rar
    Includes an element "upsatirs" because the authors of the dataset spelt it wrong -_-
    """
    walking = 0
    standing = 1
    jogging = 2
    sitting = 3
    biking = 4
    upstairs = 5
    upsatirs = 5
    downstairs = 6


class activityDatasource(MPSDatasource):
    """
    activityDatasource is a subclass of MPSDatasource which implements data loading for the
    Dataset of walking, running, sitting, standing, jogging, biking, walking upstairs and walking downstairs activities
    as found on
    http://ps.ewi.utwente.nl/Datasets.php
    
    Use as you would use any subclass of MPSDatasource.
    This class requires that unrar and rarfile are installed.
    """
    def __init__(self, shuffled = False):
        """
        As the data needs to be downloaded from the website, the initialiser handles that as well,
        then calls _load_all_data to handle loading the contents of the downloaded data.
        :param shuffled: boolean
            Pass true to shuffle the dataset.
            This parameter is not very meaningful in this case because _load_all_data shuffles
            the data anyway.
        """
        self.data_length = 200
        self.training_fraction = 0.2
        expected_shape = (int(self.data_length/2), 4)
        self._compressed_data_path = "ActivityData.rar"
        self._uncompressed_data_path = "DataSet/"
        self._all_data = None
        self._all_data_path = "all_data" + type(self).__name__ + ".npy"
        self._all_labels_path = "all_labels" + type(self).__name__ + ".npy"
        if os.path.isfile(self._all_data_path):
            self._all_data = (np.load(self._all_data_path), np.load(self._all_labels_path))
            if self._all_data[0][0].shape != expected_shape:
                self._all_data = None
        if not os.path.isfile(self._compressed_data_path):
            urllib.request.urlretrieve("http://ps.ewi.utwente.nl/Blog/Sensors_Activity_Recognition_DataSet_Shoaib.rar", self._compressed_data_path)
        if not os.path.isfile(self._uncompressed_data_path):
        # REMEMBER THAT THIS REQUIRES UNRAR INSTALLED
            with RarFile("ActivityData.rar") as rf:
                rf.extractall()
        super().__init__(expected_shape, shuffled)
        
    def _load_all_data(self):
        """
        _load_all_data is responsible for reading the .csv files downloaded in the initialisation.
        The results are saved into _all_data
        :return: nothing
        """
        _all_datapoints = []
        _all_labels = []
        counter = 0
        factor = 2/self.data_length
        new_length = int(self.data_length/2)
        ones = np.ones(new_length)
        _spinner = spinner()
        for i in range(10):
        
            #_spinner.print_spinner(0.0)
            percentage = int((i / 10) * 100)
            
            filename = self._uncompressed_data_path + "Participant_" + str(i+1) +".csv"
            with open(filename, 'r') as f:
                reader = csv.reader(f)
                header = next(reader)
                headings = next(reader)
                jump_index = 0
                index = 0
                prev_row_label = None
                row_label = 0
                data = []
                for index, row in enumerate(reader):
                
                    _spinner.print_spinner(percentage)
                    
                    if index >= jump_index:
                        if index != 0 and (index) % self.data_length == 0:
                            data = np.abs(np.fft.rfft(data, axis = 0)*factor)[:-1]
                            data = np.column_stack((ones, data))
                            _all_datapoints.append(np.array(data))
                            _all_labels.append(row_label.value)
                            data = []
                            prev_row_label = None
                        data.append(np.array([np.float32(row[1]), np.float32(row[2]), np.float32(row[3])]))
                        row_label = activityLabels[row[-1]]
                        if prev_row_label is not None:
                            if row_label != prev_row_label:
                                modulo = index % self.data_length
                                jump_index = (modulo + 1) * self.data_length
                        prev_row_label = row_label
        _all_datapoints = np.array(_all_datapoints)
        _all_labels = convert_to_onehot(np.array(_all_labels))
        permutation = np.random.permutation(len(_all_datapoints))
        #_all_datapoints = _all_datapoints[permutation]
        #_all_labels = _all_labels[permutation]
        #_all_datapoints[:,:,1:] = np.tanh(_all_datapoints[:,:,1:])
        
        _spinner.print_spinner(100.0)
        
        print(_all_datapoints.shape)
        print(_all_labels.shape)
        print(_all_labels[0])
        self._all_data = (_all_datapoints, _all_labels)
        np.save(self._all_data_path, _all_datapoints)
        np.save(self._all_labels_path, _all_labels)
    
    
    def _load_test_data(self):
        """
        Takes part of _all_data (non-overlapping with training data)
        and uses that as the test data
        :return: nothing
        """
        if self._all_data is None:
            self._load_all_data()
        test_index = int(self.training_fraction * len(self._all_data[0]))
        self._test_data = self._all_data[0][:test_index], self._all_data[1][:test_index]
        super()._load_test_data()
        
    def _load_training_data(self):
        """
        Takes part of _all_data (non-overlapping with test data)
        and uses that as the training data
        :return: nothing
        """
        if self._all_data is None:
            self._load_all_data()
        test_index = int(self.training_fraction * len(self._all_data[0]))
        self._training_data = self._all_data[0][test_index:], self._all_data[1][test_index:]
        super()._load_training_data()
        
if __name__ == "__main__":
    data_source = activityDatasource(shuffled = False)
    data, labels = data_source.next_training_data_batch(1000)
    print(data.shape)
    print(len(labels))
    data, labels = data_source.next_training_data_batch(1000)
    print(data.shape)
    print(len(labels))
    data, labels = data_source.next_training_data_batch(1000)
    print(len(labels))
    data, labels = data_source.next_training_data_batch(500)
    print(len(labels))
