import numpy as np
import tensorflow as tf
import pandas as pd
import os
import utils

# Download the data
if not os.path.isdir('aclImdb'):
    url = 'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
    name = 'aclImdb_v1.tar.gz'
    utils.getunzipped(url, name)

# Extract data into lists 
def _get_data(directory):

    files = [os.path.join(directory, f) for f in os.listdir(directory)]
    length = len(files)
    text_list = []
    label_list = []
    for f_name in files:
        _, label = f_name[:-4].split('_')
        label_list.append(label) 

        file = open(f_name, 'r')
        text_list.append(file.read())
        file.close()

    return text_list, label_list

max_review_length = 50

train_texts_pos, train_labels_pos = _get_data('aclImdb/train/pos')
train_texts_neg, train_labels_neg = _get_data('aclImdb/train/neg')
train_texts = train_texts_pos + train_texts_neg 
train_labels = train_labels_pos + train_texts_neg
train_lengths = [min(max_review_length, len(x.split(' '))) for x in train_texts]

test_texts_pos, test_labels_pos = _get_data('aclImdb/test/pos')
test_texts_neg, test_labels_neg = _get_data('aclImdb/test/neg')
test_texts = test_texts_pos + test_texts_neg 
test_labels = test_labels_pos + test_labels_neg
test_lengths = [min(max_review_length, len(x.split(' '))) for x in test_texts]

# Convert text to numbers 
preprocessor = tf.contrib.learn.preprocessing.VocabularyProcessor(max_review_length,min_frequency=3)
vocab_length = len(preprocessor.vocabulary_)

train_data = np.array(list(preprocessor.fit_transform(train_texts)))
test_data = np.array(list(preprocessor.transform(test_texts)))

del train_texts 
del test_texts 

# train an n-gram model on the training data

data_index = 0 
def generate_batch(batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    buffer = collections.deque(maxlen=span)
    if data_index + span > len(data):
        data_index = 0
    buffer.extend(train_data[data_index:data_index + span])
    data_index += span
    for i in range(batch_size // num_skips):
        target = skip_window  # target label at the center of the buffer
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
                targets_to_avoid.append(target)
                batch[i * num_skips + j] = buffer[skip_window]
                labels[i * num_skips + j, 0] = buffer[target]
        if data_index == len(train_data):
            buffer[:] = train_data[:span]
            data_index = span
        else:
            buffer.append(train_data[data_index])
            data_index += 1
            
    # Backtrack a little bit to avoid skipping words in the end of a batch
    data_index = (data_index + len(train_data) - span) % len(train_data)

    return batch, labels

batch_size = 100
num_skips = 2
skip_window = 1 
batch, labels = generate_batch(batch_size, num_skips, skip_window)
print(preprocessor.reverse(batch[0]))
print(preprocessor.reverse(labels[0]))





















