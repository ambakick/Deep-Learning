#This can be used as a helper function for a bigger project
__author__ = "Neeraj Menon"
__email__ = "neerajmenons@gmail.com"

#Sigmoid function produces similar results to step function in that the output is between 0 and 1.
#Thus mainly uved in logistic regression

import numpy as np
def sigmoid(z):
    '''
    Arguments:
    z -- A scalar or numpy array of any size.
    Return:
    s -- sigmoid(z)
    '''
    s = None
    s = 1/(1+np.exp(-z))
    return s