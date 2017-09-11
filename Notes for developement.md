# Notes for further developement of MPS:
* A "Start node" and "end node" are created, which are not trained, which are vectors that exist to ensure that the rest of the MPS can be of a consistent shape. (Else the first and last "nodes", i.e. tensors would be of rank 2 whereas all the non-special nodes in the middle would be rank 3). This also has the benefit of allowing the special node to be anywhere, including those at the start and the end of the trained chain without special provisions for this.
* When the C1 and C2 matrices are calculated and put into matrices, the indices of each individual C1 and C2 in the C1s and C2s matrices are as follows: C1s\[i+1\] = C1\[i\] * (the contracted node\[i\]) , and C2s\[i-1\] = C2s\[i\] * (the contracted node\[i\]), which introduces interesting indexing for the two-site DMRG. (And quite beautiful indexing for one-site DMRG)

## Shapes of tensors used in MPS:
The words feature dimension and local dimension may be used interchangably, and refer to the length of the individual inputs. The dimensions of the tensors to the left and the right will be referred to as left dimension and right dimension respectively. The output dimension refers to the number of classes in a classification problem. The length of the MPS, which also is the number of inputs, is referred to as the input size. Batch size refers to the number of samples fed into the MPS at once, either for training or for inference.

### Inputs:
* (input size, batch size, feature dimension)

### Predictions/Labels:
* (batch size, output dimension)

### Nodes/Weights:
* Regular node: (feature dimension, left dimension, right dimension)
* Special node: (output dimension, feature dimension, left dimension, right dimension)

### C1 (All the nodes and inputs to the left of the current node contracted):
* (batch size, right dimension)

### C2 (All the nodes and inputs to the right of the current node contracted):
* (batch size, left dimension)

### Bond tensor (and gradient)
* (output dimension, feature dimension 1, feature dimension 2, left dimension, right dimension)

### C matrix (the projected input) for two-site
* (batch size, feature dimension 1, feature dimension 2, left dimension, right dimension)

### Outputs of \_bond_decomposition for two-site
* a_prime_j: (feature dimension, left dimension, right dimension), where the right dimension is now expanded/ condensed according to the results of SVD
* a_prime_j1: (left dimension, output dimension, feature dimension, right dimension), where the left dimension is now expanded/ condensed according to the results of SVD

### C matrix(the projected input) for one-site
* (batch size, feature dimension, left dimension, right dimension)

### "Bond" tensor and gradient for one-site
* (output dimension, feature dimension, left dimension, right dimension)

### Outputs of \_bond_decomposition for one-site
* a_prime_j: (feature dimension, left dimension, right dimension), where the right dimension is now expanded/ condensed according to the results of SVD
* a_prime_j1: (left dimension, output dimension, right dimension), where the left dimension is now expanded/ condensed according to the results of SVD

## File Format
Previous to this file format, the weights were saved as a pickled list. The disadvantages of this were that
* The list didn't explicitly contain higher level data about the MPS, e.g. the input_size
    - This meant that to initialise an MPS, even if we had the weights, we had to initialise MPSs by feeding in all of the parameters manually
* Pickle files are Python-specific
This file format was created to address these issues. An added benefit of the file format is that it is much [more space efficient than a Pickle](https://stackoverflow.com/questions/9619199/best-way-to-preserve-numpy-arrays-on-disk). By using this file format, one can also save the configuration of an MPS without saving the weights.
It is a binary file currently with the following format:

|Stored Content     |Type   |Bytes  |Total Bytes|
|-------------------|-------|-------|-----------|
|contains_weights   |bool   |1      |1          |
|input_size         |int16  |2      |3          |
|special_node_loc   |int16  |2      |5          |
|d_feature          |int16  |2      |7          |
|d_output           |int16  |2      |9          |

If contains_weights is True, then the dimensions of the weights and the weights themselves are read in as well. Dimension m refers to the right hand side dimension of the tensor at position m. The total bytes for weight m is a little complicated, but can be inferred from the contents of the table below/the implementation of from_file in mps.py can be used as a reference. 

|Stored Content     |Type               |Bytes      |Total Bytes|
|-------------------|-------------------|-----------|-----------|
|dim0               |int16              |2          |11         |
|dim1               |int16              |2          |13         |
|...                |...                |...        |...        |
|dim(input_size-1)  |int16              |2          |9 + (input_size-1)\*2|
|weight 0           |buffer of float32s |2\*input_size\*dim0\*d_feature\*4|...|
|weight 1           |buffer of float32s |dim0\*dim1\*d_feature\*4|...|
|...                |...                |...        |...        |
|weight m (m=special_node_loc)|buffer of float32s|dim(m-1)\*dim(m)\*d_feature\*4|...|
|...                |...                |...        |...        |
|final weight       |buffer of float32s |dim(-1)\*2\*input_size\*d_feature\*4|...|

