sqMPS
========

sqMPS is a subclass of MPS that uses the square error instead of the cross entropy. It should be used in the exact same way as the MPS class, including in optimization. Take note that when using sqMPS, one cannot use the Hessian during optimization. The only difference in API between sqMPS and its parent class is that the cost(f, labels) function for sqMPS returns the mean squared error as opposed to the softmax cross entropy with logits.
