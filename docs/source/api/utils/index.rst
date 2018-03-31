utils
====================================

utils includes classes and functions that were found to be useful for more than one part of TrMPS and was not found to belong in any particular other part.

The spinner class
-------------------

spinner is a class that is used for showing a spinning indicator to show progress during things such as loading data from the internet for datasources. An example of its usage is as follows:

.. code-block:: python


    _spinner = spinner(jump = 300)
    for i in range(20):
        percentage = int((i / 20) * 100)
        images, labels = get_more_data()
        for index, image in enumerate(images):
            data.append(preprocess(image))
            labels.append(np.array(batch[1][index]))
            _spinner.print_spinner(percentage)

__init__(MPSNetwork, jump = 400)
^^^^^^^^
 *jump: integer*
  The spinner is often used for showing progress in a situation where the spinner is called many more times than you would want the spinner to update on screen, so the jump parameter is used to control how often the displayed spinner is updated. i.e.,The larger this value is, the less often the spinner is printed.

print_spinner(percentage)
^^^^^^^^
  A function to print the percentage and the spinner.

  *percentage: any number type*
   The percentage that will be printed out along with the printer

Other functions
----------------

check_nan(tensor, name, replace_nan=True)
^^^^^^^^

 A tensorflow operation to check if there are NaN values in *tensor*, and if *replace_nan*, also replaces them with 0s. Will also print out the name of the tensor that is NaN, where the name is determined by *name*.

 *tensor: Tensor*
  The tensor which will be checked for NaN values.
 *name: string*
  The name of the tensor, which will be printed out if there exist NaN values.
 *replace_nan: boolean*
  If replace_nan, the NaN values are replaced with zeroes. If false, the NaN values are not replaced, and the operation just logs when NaN values exist.

getunzipped(url, name)
^^^^^^^^^
 Downloads the file at *url* and unzips it into a file with the name *name*.

 *url: string*
  The url of the file to be retrieved.
 *name: string*
  The name of the file into which the results will be downloaded.

list_from(tensorArray, length)
^^^^^^^^^^
 list_from is a tensorflow operation that produces a list from a tensorArray. It is used to extract the results of training in MPSOptimizer.

 *tensorArray: tensorflow TensorArray*
  The tensor array that is to be converted to a list.
 *length: integer*
  The length of the list that is to be created. Needs to be smaller than the length of the TensorArray else there will be an out of bounds error.
 *returns: list of tensorflow Tensors*
  A list containing all the values of the TensorArray as Tensors. This has to then be evaluated to get actual values.

convert_to_onehot(vector, num_classes=None)
^^^^^^^^^^
 Converts an input 1-D vector of integers into an output 2-D array of one-hot vectors, where an i'th input value of j will set a '1' in the i'th row, j'th column of the output array.

.. code-block:: python


    v = np.array((1, 0, 4))
    one_hot_v = convertToOneHot(v)
    print(one_hot_v)

    output:
    [[0 1 0 0 0]
     [1 0 0 0 0]
     [0 0 0 0 1]]


into_image(snaked_image)
^^^^^^^^^
 Converts a square image that has been turned into a 1D array back into a square image.

 *snaked_image: numpy array of size (any integer) squared*
  The 1D array that will be converted into a 2D square numpy array that will represent the image.

 *returns: numpy array of size (n, n)*
  The snaked image converted back into a square one.

show(snaked_image, normalise=False)
^^^^^^^^^
 Takes a snaked image, and plots it as a square black and white image using matplotlib.

 *snaked_image: numpy array of size (any integer) squared*
  The 1D array that will be converted into a 2D square numpy array that will represent the image.
 *normalise: boolean, default False*
  Represents whether the image is already normalised. If this parameter is True, regardless of the values in the image, a value of 1.0 will represent black, and a value of 0.0 will represent white. If this is False, then the image will be plotted with the largest value in the image being black and the smallest value in the image being white.

