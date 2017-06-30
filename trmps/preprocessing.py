import tensorflow as tf
import sys
from tensorflow.examples.tutorials.mnist import input_data


def preprocess_images():
    # Function to process images into format from paper, also currently just returns the images for zeroes and ones,
    # so that we can create a binary classifier first.
    # Returned as a list of [images, classifications]

    # Import MNIST dataset
    print("Importing data")

    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    sess = tf.Session()
    data = []

    # Tensorflow operators / placeholders to resize the data from MNIST to format from paper
    # Resize images from 28*28 to 14*14
    image = tf.placeholder(tf.float32, shape=[784])
    reshaped_image = tf.reshape(image, [-1, 28, 28, 1])
    pool = tf.nn.avg_pool(reshaped_image, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')
    pooled_image = tf.placeholder(tf.float32, shape=[1, 14, 14, 1])
    snaked_image = tf.reshape(pooled_image, shape=[196])
    print("0 % done", end="")

    # Loop through all the elements in the dataset and resize
    with sess.as_default():
        sess.run(tf.global_variables_initializer())
        counter = 0
        for i in range(20):
            sys.stdout.flush()
            print("\r" + str(int((i / 20) * 100)) + " % done", end="")
            batch = mnist.train.next_batch(3000)
            images = batch[0]
            for index, element in enumerate(images):
                pooled = sess.run(pool,
                                  feed_dict={reshaped_image: sess.run(reshaped_image,
                                                                      feed_dict={image: element})})
                data.append([sess.run(snaked_image, feed_dict={pooled_image: pooled}), batch[1][index]])
                if index % 300 == 0:
                    # Spinner to show progress 
                    if counter == 0:
                        print("\r" + str(int((i / 20) * 100)) + " % done", end="|")
                        counter += 1
                    elif counter == 1:
                        print("\r" + str(int((i / 20) * 100)) + " % done", end="/")
                        counter += 1
                    elif counter == 2:
                        print("\r" + str(int((i / 20) * 100)) + " % done", end="-")
                        counter += 1
                    elif counter == 3:
                        print("\r" + str(int((i / 20) * 100)) + " % done", end="\\")
                        counter = 0
    sys.stdout.flush()
    print("\r" + str(100) + " % done")

    zeroes_and_ones = np.array(list(filter(lambda x: x[1][0] == 1. or x[1][1] == 1., data)))
    return zeroes_and_ones

if __name__ == "__main__":
    # If main, processes the images and also prints the number of images
    print(len(preprocess_images()))

