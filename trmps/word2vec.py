import numpy as np
import tensorflow as tf
import pandas as pd
import os
import utils
from preprocessing import MPSDatasource



class MovieReviewDatasource(MPSDatasource):

    def __init__(self, expected_shape=None, shuffled=False, embedding_size=50):
        super().__init__(expected_shape, shuffled)
        self.embedding_size = embedding_size
        self._embedding_path = ''

    def _load_training_data(self):

        if not os.path.isdir('aclImdb'):
            url = 'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
            name = 'aclImdb_v1.tar.gz'
            utils.getunzipped(url, name)

        train_texts_pos, train_labels_pos = self._load_data_to_list('aclImdb/train/pos')
        train_texts_neg, train_labels_neg = self._load_data_to_list('aclImdb/train/neg')
        train_texts = train_texts_pos + train_texts_neg 
        train_labels = np.array((train_labels_pos + train_labels_neg), dtype=np.float32)

        data = self._train_word2vec(train_texts)
        del train_texts

        self._training_data = (data, train_labels)
        np.save(self._test_data_path, self._test_data[0])
        np.save(self._test_labels_path, self._test_data[1])

    def _train_word2vec(self, train_texts):

        # convert the words to ids from most frequent to least 
        self._preprocessor = tf.contrib.learn.preprocessing.VocabularyProcessor(self._expected_shape, 
                                                                                                                                                         min_frequency=3)
        data = np.array(list(self._preprocessor.fit_transform(train_texts)))
        vocab_size = len(self._preprocessor.vocabulary_)
        print(data.shape)

        # test with some data 
        print(next(self._preprocessor.reverse(data[0, None])))

        self._data_index = 0 
        batch_size = 128
        num_skips = 2
        skip_window = 1 

        # skip_gram = SkipGramModel(batch_size, num_skips, skip_window, 
        #                                 self.embedding_size, vocab_size)
        # embedding_matrix = skip_gram.train_model(data)
        return data #, embedding_matrix


    def plot_with_tsne(plot_only=200, filename='tsne.png'):

        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE

        plot_only = 200
        low_dim_embs = tsne.fit_transform(self.embedding_matrix[:plot_only, :])
        words = [preprocessor.reverse[i] for i in xrange(plot_only)] # this won't work yet!!!!!!!

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

    def __init__(self, batch_size, num_skips, skip_window, embedding_size, vocab_size):
        self.batch_size = batch_size
        self.num_skips = num_skips
        self.skip_window = skip_window
        self.embedding_size = embedding_size
        self.vocab_size = vocab_size

    def train_model(self, train_data):
        self._train_data = train_data
        num_sampled = 64    # Number of negative examples to sample.

        graph = tf.Graph()

        with graph.as_default():

            # make input data
            train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
            train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])

            # pinned to the cpu due to missing gpu impl
            with tf.device('/cpu:0'): 
                # Look up embeddings for inputs.
                embeddings = tf.Variable(
                        tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0))
                embed = tf.nn.embedding_lookup(embeddings, train_inputs)

                # Construct the variables for the NCE loss
                nce_weights = tf.Variable(
                        tf.truncated_normal([self.vocab_size, self.embedding_size],
                                             stddev=1.0 / np.sqrt(self.embedding_size)))
                nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

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

        num_steps = 100001

        with tf.Session(graph=graph) as session:

            # We must initialize all variables before we use them.
            init.run()
            print('Initialized')

            average_loss = 0
            self._data_index = 0 

            for step in xrange(num_steps):
                batch_inputs, batch_labels = self._generate_batch(
                        batch_size, num_skips, skip_window)
                feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

                # We perform one update step by evaluating the optimizer op (including it
                # in the list of returned values for session.run()
                _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
                average_loss += loss_val

                if step % 2000 == 0:
                    if step > 0:
                        average_loss /= 2000
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
            buffer = collections.deque(maxlen=span)
            if self._data_index + span > len(self._train_data):
                self._data_index = 0
            buffer.extend(self._train_data[self._data_index:self._data_index + span])
            self._data_index += span
            for i in range(batch_size // num_skips):
                target = skip_window  # target label at the center of the buffer
                targets_to_avoid = [skip_window]
                for j in range(num_skips):
                    while target in targets_to_avoid:
                        target = random.randint(0, span - 1)
                        targets_to_avoid.append(target)
                        batch[i * num_skips + j] = buffer[skip_window]
                        labels[i * num_skips + j, 0] = buffer[target]
                if self._data_index == len(self._train_data):
                    buffer[:] = self._train_data[:span]
                    self._data_index = span
                else:
                    buffer.append(self._train_data[self._data_index])
                    self._data_index += 1
                            
            # Backtrack a little bit to avoid skipping words in the end of a batch
            self._data_index = (self._data_index + len(self._train_data) - span) % len(self._train_data)

            return batch, labels

if __name__ == '__main__':
    max_size = 100
    embedding_size = 30 
    print('hi')
    datasource = MovieReviewDatasource(expected_shape=max_size, embedding_size= embedding_size)
    datasource._load_training_data()























