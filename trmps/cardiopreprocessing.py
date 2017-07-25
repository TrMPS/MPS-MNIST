import tensorflow as tf
import numpy as np
import sys
import os
from preprocessing import *
import csv
from utils import spinner
from utils import convert_to_onehot
import urllib
from enum import Enum
from zipfile import ZipFile
import scipy.io
from pathlib import Path

class cardioLabels(Enum):
    noisy = 0
    N = 1
    normal = 1
    O = 2
    other = 2
    A = 3
    AF = 3
    
class cardioDatasource(MPSDatasource):
    
    def __init__(self, shuffled = False):
        if not os.path.isdir(type(self).__name__):
            os.mkdir(type(self).__name__)
        self.data_length = 1800
        self.training_fraction = 0.2
        expected_shape = (int(self.data_length/2), 4)
        self._compressed_data_path = "CardioData.zip"
        self._uncompressed_data_path = os.path.join(type(self).__name__, "training2017/")
        self._all_data = None
        self._all_data_path = os.path.join(type(self).__name__, "all_data.npy")
        self._all_labels_path = os.path.join(type(self).__name__, "all_labels.npy")
        if os.path.isfile(self._all_data_path):
            self._all_data = (np.load(self._all_data_path), np.load(self._all_labels_path))
            if self._all_data[0][0].shape != expected_shape:
                self._all_data = None
        if not os.path.isfile(self._compressed_data_path):
            urllib.request.urlretrieve("https://physionet.org/challenge/2017/training2017.zip", self._compressed_data_path)
        if not os.path.isfile(self._uncompressed_data_path):
            with ZipFile(self._compressed_data_path) as zf:
                zf.extractall(type(self).__name__)
        super().__init__(expected_shape, shuffled)
        
    def _load_all_data(self):
        _all_datapoints = []
        _all_labels = []
        counter = 0
        new_length = int(self.data_length/2)
        ones = np.ones(new_length)
        _spinner = spinner(200)
        counter = np.array([0, 0, 0, 0])
        csv_filename = self._uncompressed_data_path + "REFERENCE.csv"
        with open(csv_filename, 'r') as f:
            reader = csv.reader(f)
            for index, row in enumerate(reader):
                current_loc = self.data_length
                percentage = int(100 * index/8528.0)
                _spinner.print_spinner(percentage)
                record = self._uncompressed_data_path + row[0] + ".mat"
                mat_data = scipy.io.loadmat(record)
                samples = mat_data["val"]
                samples = samples.flatten()
                len_left = len(samples)
                data = samples[:self.data_length]
                data = np.abs(np.fft.rfft(data))[:-1]
                factor = 1/np.amax(data)
                data = data * factor
                data = np.column_stack((ones, data))
                label = cardioLabels.noisy
                if row[1] != "~":
                    label = cardioLabels[row[1]]
                _all_labels.append(label.value)
                _all_datapoints.append(data)
                counter[label.value] = counter[label.value] + 1
                len_left -= self.data_length
                while len_left > self.data_length and counter[label.value]<np.amax(counter):
                    data = samples[current_loc:current_loc+self.data_length]
                    data = np.abs(np.fft.rfft(data))[:-1]
                    factor = 1/np.amax(data)
                    data = data * factor
                    data = np.column_stack((ones, data))
                    label = cardioLabels.noisy
                    if row[1] != "~":
                        label = cardioLabels[row[1]]
                    _all_labels.append(label.value)
                    _all_datapoints.append(data)
                    counter[label.value] = counter[label.value] + 1
                    current_loc += self.data_length + 1
                    len_left -= self.data_length
        _all_datapoints = np.array(_all_datapoints)
        _all_labels = convert_to_onehot(np.array(_all_labels))
        
        _spinner.print_spinner(100.0)
        
        print(_all_datapoints.shape)
        print(_all_labels.shape)
        print(_all_labels[0])
        self._all_data = (_all_datapoints, _all_labels)
        np.save(self._all_data_path, _all_datapoints)
        np.save(self._all_labels_path, _all_labels)
    
    
    def _load_test_data(self):
        if self._all_data is None:
            self._load_all_data()
        test_index = int(self.training_fraction * len(self._all_data[0]))
        self._test_data = self._all_data[0][:test_index], self._all_data[1][:test_index]
        super()._load_test_data()
        
    def _load_training_data(self):
        if self._all_data is None:
            self._load_all_data()
        test_index = int(self.training_fraction * len(self._all_data[0]))
        self._training_data = self._all_data[0][test_index:], self._all_data[1][test_index:]
        super()._load_training_data()
        
if __name__ == "__main__":
    data_source = cardioDatasource(shuffled = False)
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

        