import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from preprocessing.preprocessing import *
import numpy as np
from utils import spinner
from utils import convert_to_onehot
from enum import Enum

class inputEncoding(Enum):
    none=0
    linear=1
    trig=2

def _create_random_data(sequence_length, encoding=inputEncoding.none):
    if encoding == inputEncoding.none:
        sequence = np.zeros((sequence_length, 1))
    else:
        sequence = np.zeros((sequence_length, 2))
    for i in range(sequence_length):
        rand_int = np.random.randint(low=0, high=2)
        if encoding == inputEncoding.none:
            sequence[i] = rand_int
        elif encoding == inputEncoding.linear:
            sequence[i][0] = 1
            sequence[i][1] = rand_int
        elif encoding == inputEncoding.trig:
            sequence[i][0] = 1-rand_int
            sequence[i][1] = rand_int
        else:
            raise Exception("Specified encoding does not exist!")
    rand_int = np.random.randint(low=0, high=2)
    classification = [1-rand_int, rand_int]
    return sequence, classification

class RandomSequenceDatasource(MPSDatasource):
    def __init__(self, sequence_length, encoding=inputEncoding.none, num_samples=10000):
        self.sequence_length = sequence_length
        self.encoding = encoding
        self.num_samples=num_samples
        if encoding == inputEncoding.none:
            expected_shape = (sequence_length, 1)
        elif encoding == inputEncoding.linear or encoding == inputEncoding.trig:
            expected_shape = (sequence_length, 2)
        super().__init__(expected_shape, shuffled=False)

    def _load_test_data(self):
        test_data = []
        test_labels = []
        _spinner = spinner(jump=300)
        for i in range(self.num_samples):
            data, label = _create_random_data(self.sequence_length, self.encoding)
            test_data.append(data)
            test_labels.append(label)
            percentage = float(i) / float(self.num_samples)
            _spinner.print_spinner(percentage * 100)
        _spinner.print_spinner(100.0)
        self._test_data = np.array(test_data), np.array(test_labels)
        super()._save_test_data()

    def _load_training_data(self):
        if self._test_data is None:
            self._load_test_data()
        self._training_data = self._test_data
        super()._save_training_data()

if __name__ == "__main__":
    # If main, processes the images and also prints the number of images
    sequence_length = 30
    num_samples = 10000
    encoding = inputEncoding.trig
    data_source = RandomSequenceDatasource(sequence_length, encoding, num_samples)
    data, labels = data_source.next_training_data_batch(1000)
    print(data.shape)
    print(data[:,0,:])
    print(labels.shape)
    print(labels[0])
    print(len(labels))
    print(labels[0])
    data, labels = data_source.next_training_data_batch(1000)
    print(data.shape)
    print(len(labels))
    print(labels[0])
    data, labels = data_source.next_training_data_batch(1000)
    print(len(labels))
    print(labels[0])
    data, labels = data_source.next_training_data_batch(500)
