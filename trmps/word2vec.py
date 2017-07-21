import numpy as np
import tensorflow as tf
import pandas as pd
import os
import utils
from preprocessing import MPSDatasource
import collections



class MovieReviewDatasource(MPSDatasource):

    def __init__(self, expected_shape=None, shuffled=False, embedding_size=50, max_doc_length=100):
        self._embedding_size = embedding_size
        self._max_doc_length = max_doc_length
        super().__init__(expected_shape, shuffled)

    def _load_all_data(self):

        # download the data from url if necessary
        if not os.path.isdir('aclImdb'):
            url = 'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
            name = 'aclImdb_v1.tar.gz'
            utils.getunzipped(url, name)

        # read in the training and test data 
        train_texts, train_labels = self._extract_data_from_files('train')
        test_texts, test_labels = self._extract_data_from_files('test')

        # convert the training and test data to vectors 
        train_vecs, test_vecs = self._word2vec(train_texts, test_texts)
        del train_texts, test_texts
        train_vecs = self._reshape_data(train_vecs)
        test_vecs = self._reshape_data(test_vecs)

        self._training_data = (train_vecs, train_labels)
        self._test_data = (test_vecs, test_labels)

        # save the data 
        super()._load_training_data()
        super()._load_test_data()

    def _load_training_data(self):
        self._load_all_data()

    def _load_test_data(self):
        self._load_all_data()

    def _extract_data_from_files(self, tag):
        '''
        class: 
            1, 2, 3, 4, 7, 8, 9, 10
        '''
        texts_pos, labels_pos = self._load_data_to_list(os.path.join('aclImdb', tag, 'pos'))
        texts_neg, labels_neg = self._load_data_to_list(os.path.join('aclImdb', tag, 'neg'))
        texts = texts_pos + texts_neg
        labels = np.array((labels_pos + labels_neg), dtype=np.int)

        one_hot_labels = np.zeros([labels.size, 8])
        one_hot_labels[labels == 1][:, 0] = 1
        one_hot_labels[labels == 2][:, 1] = 1 
        one_hot_labels[labels == 3][:, 2] = 1
        one_hot_labels[labels == 4][:, 3] = 1 
        one_hot_labels[labels == 7][:, 4] = 1
        one_hot_labels[labels == 8][:, 5] = 1 
        one_hot_labels[labels == 9][:, 6] = 1
        one_hot_labels[labels == 10][:, 7] = 1 

        return texts, one_hot_labels

    def _reshape_data(self, data):
        # make the shape into (input_size, num_data, d_feature)
        num_data = data.shape[0]
        data = data.reshape([num_data, self._expected_shape])
        ones = np.ones([num_data, self._expected_shape])
        data = np.dstack((ones, data))
        return data 
        
    def _word2vec(self, train_texts, test_texts):

        # convert the words to ids from most frequent to least 
        self._preprocessor = tf.contrib.learn.preprocessing.VocabularyProcessor(self._max_doc_length, 
                                                                                min_frequency=3)
        train_ids = np.array(list(self._preprocessor.fit_transform(train_texts)))
        test_ids = np.array(list(self._preprocessor.transform(test_texts)))
        vocab_size = len(self._preprocessor.vocabulary_)

        self._data_index = 0 
        batch_size = 100
        num_skips = 2
        num_steps = int(np.floor(train_ids.size*num_skips/batch_size))
        skip_window = 1 
        data_for_skip_gram = train_ids.flatten()
        data_for_skip_gram = data_for_skip_gram[data_for_skip_gram > 0] # filter out zeros
        

        skip_gram = SkipGramModel(batch_size, num_skips, skip_window, 
                                         self._embedding_size, vocab_size)
        self._embedding_matrix = skip_gram.train_model(data_for_skip_gram, num_steps)
        train_vecs, test_vecs = self._embedding_look_up(train_ids, test_ids)

        del train_ids, test_ids
        return train_vecs[0], test_vecs[0]


    def plot_with_tsne(plot_only=200, filename='tsne.png'):

        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE

        plot_only = 200
        low_dim_embs = tsne.fit_transform(self.embedding_matrix[:plot_only, :])
        words = next(self._preprocessor.reverse([list(range(plot_only))]))

        plt.figure(figsize=(18, 18))  # in inches
        for i, word in enumerate(words):
            x, y = self.embedding_matrix[i, :]
            plt.scatter(x, y)
            plt.annotate(word,
                         xy=(x, y),
                         xytext=(5, 2),
                         textcoords='offset points',
                         ha='right',
                         va='bottom')

        plt.savefig(filename)

    def _embedding_look_up(self, train_ids, test_ids):

        graph = tf.Graph()
        with graph.as_default(), tf.device('/cpu:0'):
            with tf.name_scope('embedding_lookup'):
                embedding = tf.constant(self._embedding_matrix, dtype=tf.float32)
                ids = tf.placeholder(tf.int32, shape=[None, self._max_doc_length],)
                word_vectors = tf.nn.embedding_lookup(embedding, ids) 

        with tf.Session(graph=graph) as session:

            train_vecs = session.run([word_vectors], feed_dict={ids: train_ids})
            test_vecs = session.run([word_vectors], feed_dict={ids: test_ids})

        return train_vecs, test_vecs


    def _load_data_to_list(self, directory):

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


class SkipGramModel(object):

    def __init__(self, batch_size, num_skips, skip_window, _embedding_size, vocab_size):
        self.batch_size = batch_size
        self.num_skips = num_skips
        self.skip_window = skip_window
        self._embedding_size = _embedding_size
        self.vocab_size = vocab_size

    def train_model(self, train_data, num_steps):
        print('Training a skipgram model')
        self._train_data = train_data
        num_sampled = 64    # Number of negative examples to sample.

        graph = tf.Graph()

        with graph.as_default():

            # make input data
            train_inputs = tf.placeholder(tf.int32, shape=[self.batch_size])
            train_labels = tf.placeholder(tf.int32, shape=[self.batch_size, 1])

            # pinned to the cpu due to missing gpu impl
            with tf.device('/cpu:0'): 
                # Look up embeddings for inputs.
                embeddings = tf.Variable(
                        tf.random_uniform([self.vocab_size, self._embedding_size], -1.0, 1.0))
                embed = tf.nn.embedding_lookup(embeddings, train_inputs)

                # Construct the variables for the NCE loss
                nce_weights = tf.Variable(
                        tf.truncated_normal([self.vocab_size, self._embedding_size],
                                             stddev=1.0 / np.sqrt(self._embedding_size)))
                nce_biases = tf.Variable(tf.zeros([self.vocab_size]))

            # Compute the average NCE loss for the batch.
            # tf.nce_loss automatically draws a new sample of the negative labels each
            # time we evaluate the loss.
            loss = tf.reduce_mean(
                    tf.nn.nce_loss(weights=nce_weights,
                                   biases=nce_biases,
                                   labels=train_labels,
                                   inputs=embed,
                                   num_sampled=num_sampled,
                                   num_classes=self.vocab_size))

            # Construct the SGD optimizer using a learning rate of 1.0.
            optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

            # Compute the cosine similarity between minibatch examples and all embeddings.
            norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
            normalized_embeddings = embeddings / norm

            # Add variable initializer.
            init = tf.global_variables_initializer()

        with tf.Session(graph=graph) as session:

            # We must initialize all variables before we use them.
            init.run()
            print('Initialized')

            average_loss = 0
            self._data_index = 0 

            for step in range(num_steps):
                batch_inputs, batch_labels = self._generate_batch(
                        self.batch_size, self.num_skips, self.skip_window)
                feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

                # We perform one update step by evaluating the optimizer op (including it
                # in the list of returned values for session.run()
                _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
                average_loss += loss_val

                if step % 1000 == 0:
                    if step > 0:
                        average_loss /= 1000
                        # The average loss is an estimate of the loss over the last 2000 batches.
                        print('Average loss at step ', step, ': ', average_loss)
                        average_loss = 0

            final_embeddings = normalized_embeddings.eval()
        return final_embeddings


    def _generate_batch(self, batch_size, num_skips, skip_window):
        
        assert batch_size % num_skips == 0
        assert num_skips <= 2 * skip_window
        batch = np.ndarray(shape=(batch_size), dtype=np.int32)
        labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
        span = 2 * skip_window + 1  # [ skip_window target skip_window ]
        queue = collections.deque(maxlen=span)
        if self._data_index + span > len(self._train_data):
            self._data_index = 0
        queue.extend(self._train_data[self._data_index:self._data_index + span])
        self._data_index += span
        for i in range(batch_size // num_skips):
            target = skip_window  # target label at the center of the queue
            targets_to_avoid = [skip_window]
            for j in range(num_skips):
                while target in targets_to_avoid:
                    target = np.random.randint(0, span)
                targets_to_avoid.append(target)
                batch[i * num_skips + j] = queue[skip_window]
                labels[i * num_skips + j, 0] = queue[target]
            if self._data_index == len(self._train_data):
                queue.extend(self._train_data[:span])
                self._data_index = span
            else:
                queue.append(self._train_data[self._data_index])
                self._data_index += 1
                        
        # Backtrack a little bit to avoid skipping words in the end of a batch
        self._data_index = (self._data_index + len(self._train_data) - span) % len(self._train_data)

        
        return batch, labels

if __name__ == '__main__':
    max_size = 100
    embedding_size = 30 
    datasource = MovieReviewDatasource(expected_shape=max_size*embedding_size, 
                                       embedding_size=embedding_size, 
                                       max_doc_length=max_size)























