from preprocessing import *
import numpy as np
from utils import spinner
from utils import convert_to_onehot

def _create_bracket_data(sequence_length, num_bracket_types, num_noise_types, max_unmatched, normalised=True):
    total = (num_bracket_types * 2) + num_noise_types
    last_bracket_num = num_bracket_types * 2
    sequence = np.zeros((sequence_length, total))
    num_unopened = np.zeros(num_bracket_types)
    num_unclosed = np.zeros(num_bracket_types)
    for i in range(sequence_length):
        rand_int = np.random.randint(low=0, high=total)
        sequence[i][rand_int] = 1.0
        if rand_int < num_bracket_types:
            num_unclosed[rand_int] = num_unclosed[rand_int] + 1
        elif rand_int < last_bracket_num:
            if num_unclosed[rand_int - num_bracket_types] > 0:
                num_unclosed[rand_int - num_bracket_types] = num_unclosed[rand_int - num_bracket_types] - 1
            else:
                num_unopened[rand_int - num_bracket_types] = num_unopened[rand_int - num_bracket_types] + 1
    num_unmatched = np.minimum(np.add(num_unopened, num_unclosed), max_unmatched)
    if normalised:
        num_unmatched = num_unmatched / max_unmatched
    return sequence, num_unmatched

class BracketMatchingDatasource(MPSDatasource):
    def __init__(self, sequence_length, num_bracket_types, num_noise_types, max_unmatched, num_test_data, num_train_data,
                 normalised=True):
        self.sequence_length = sequence_length
        self.num_bracket_types = num_bracket_types
        self.num_noise_types = num_noise_types
        self.max_unmatched = max_unmatched
        self.num_train_data = num_train_data
        self.num_test_data = num_test_data
        self.normalised = normalised
        expected_shape = (sequence_length, (num_bracket_types*2) + num_noise_types)
        super().__init__(expected_shape, shuffled=False)

    def _load_test_data(self):
        test_data = []
        test_labels = []
        _spinner = spinner(jump=300)
        for i in range(self.num_test_data):
            data, label = _create_bracket_data(self.sequence_length, self.num_bracket_types, self.num_noise_types,
                                               self.max_unmatched, self.normalised)
            test_data.append(data)
            test_labels.append(label)
            percentage = float(i) / float(self.num_test_data)
            _spinner.print_spinner(percentage * 100)
        _spinner.print_spinner(100.0)
        self._test_data = np.array(test_data), np.array(test_labels)
        super()._save_test_data()

    def _load_training_data(self):
        train_data = []
        train_labels = []
        _spinner = spinner(jump=300)
        for i in range(self.num_train_data):
            data, label = _create_bracket_data(self.sequence_length, self.num_bracket_types, self.num_noise_types,
                                               self.max_unmatched, self.normalised)
            train_data.append(data)
            train_labels.append(label)
            percentage = float(i) / float(self.num_train_data)
            _spinner.print_spinner(percentage * 100)
        _spinner.print_spinner(100.0)
        self._training_data = np.array(train_data), np.array(train_labels)
        super()._save_training_data()

if __name__ == "__main__":
    # If main, processes the images and also prints the number of images
    sequence_length = 30
    num_bracket_types = 1
    num_noise_types = 0
    max_unmatched = 10
    num_test_data = 10000
    num_train_data = 10000
    data_source = BracketMatchingDatasource(sequence_length, num_bracket_types, num_noise_types,
                                            max_unmatched, num_test_data, num_train_data)
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
