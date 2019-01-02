#import libraries
import numpy as np
import math
#defining sigmoid function

def sigmoid(A, W1, W2, b1, b2):
    z1 = np.dot(W1,A) + b1
    print(z1)
    a1 = 1/(1+(math.exp(-z1)))
    z2 = np.dot(W12,a1)+b2
    a2 = 1/(1+(math.exp(-z2)))
    #print(a2)

def tanh(A, W1, W2, b1, b2):
    z1 = W1*A + b1
    a1 = max(0,z1)
    z2 = W2*a1+b2
    a2 = max(0,z2)
    print(a2)

def tanh(A, W1, W2, b1, b2):
    z1 = W1*A + b1
    a1 = 2/(1+(math.exp(-2*z1)))
    z2 = W2*a1+b2
    a2 = 2/(1+(math.exp(-2*z2)))
    print(a2)
'''
initialize the size of
1. input matrix (n_x)
2. hidden layer (n_h)
3. output layer (n_y)
'''
n_x = 3
n_h = 2
n_y = 1
A = np.array([3, 2, 1])

#There are many options in initializing the Weight matrix
#The best one would be to make a random initialization
W1 = np.random.randn(n_h, n_x)*0.01
b1 = np.zeros((n_h, 1))
W2 = np.random.randn(n_y, n_h)*0.01
b2 = np.zeros((n_y, 1))
#print(W1, W2, b1, b2)
# Z = W1*A  
sigmoid(A, W1, W2, b1, b2)
#relu(A, W1, W2, b1, b2)
# The size of matrix A is n_x and W1 is n_h*n_x