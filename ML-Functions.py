'''
In order to understand all of the functions of Machine Learning, I will create my own ML functions from scratch using this file
'''

import numpy as np

# ACTIVATION FUNCTIONS:
def relu(z):
    return np.maximum(0, z)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def softmax(z): # Converting raw scores to probabilities
    exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)

def tanh(z):
    return np.tanh(z)

def relu_derivative(z):
    return(z > 0).astype(float)
#----------------------------------------

#FORWARD PROPAGATION:
def forward_pass(X, w1, b1, w2, b2):
    z1 = w1 @ X + b1 # Layer 1 raw output: matrix multiply weights by the input and add bias
    a1 = ReLU(z1) # Layer 1 Activation: squash the negative values to zero, and keep the positive values
    z2 = w2 @ a1 + b2 #Layer 2 raw output: matrix multiply the weights by a1 and add bias
    a2 = softmax(z2) # Layer 2 Activation: convert the raw scores into probabilities (sums to 1)
    return z1, a1, z2, a2
#------------------------------------------

#BACKWARD PROPAGATION
def backward_pass(X, Y, w2, z1, a1, a2, m):
    dz2 = a2 - Y                            # Error at output layer, (predicted probability minus the true labels)
    dw2 = (1/m) * dz2 @ a1.T                # The gradient for w2: how much each weight caused the error (averaged over m samples)
    db2 = (1/m) * np.sum(dz2, axis=1, keepdims=True)    # The gradient for b2: The average error across all samples

    dz1 = w2.T @ dz2 * relu_derivative(z1)  # Propogate error back through layer 2 and kill the gradients where the ReLU was 0
    dw1 = (1/m) * dz1 @ X.T                 # Gradient for w1: how much each weight caused the error (averaged over m samples)
    db1 = (1/m) * np.sum(dz1, axis=1, keepdims=True) # Gradient for b1: The average error across all samples
    
    return dw1, db1, dw2, db2
#------------------------------------------------------------

# LOSS FUNCTIONS
def cross_entropy_loss(a2, Y):
    m = Y.shape[1]                                  # Get the number of training examples
    return -(1/m) * np.sum(Y * np.log(a2 + 1e-8))   # Average the log loss across all of the samples
                                                    # "Y * log(a2)" rewards confident and correct predictions
                                                    # 1e-8 prevents log(0) from happening which would otherwise result in -infinity
                                                    # negative sign flips the function so that the lower the loss the better

def mse_loss(predictions, Y):
    return np.mean((predictions - Y) ** 2)          # Average squared difference between predicted and actual
                                                    # Squaring makes all the errors positive and punishes bigger mistakes
#------------------------------------------------------------

# PARAMETER UPDATES
def update_params(w1, b1, w2, b2, dw1, db1, dw2, db2, lr):
    w1 -= lr * dw1                                  # Nudge w1 opposite to its gradient (step down the slope)
    b1 -= lr * db1                                  # Same for b1 bias
    w2 -= lr * dw2                                  # Same for w2
    b2 -= lr * db2                                  # Same for b2
    return w1, b1, w2, b2                           # Return the updated weights
#------------------------------------------------------------

#EVALUATION
def get_predictions(a2):
    return np.argmax(a2, axis=0)                    # Pick the index with highest probability - that is the predicted digit

def get_accuracy(predictions, Y):
    return np.mean(predictions == Y)                # Compare the predicted digits to the true labels, and return the fraction that match