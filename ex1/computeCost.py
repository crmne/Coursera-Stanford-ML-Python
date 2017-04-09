# -*- coding: utf-8 -*-
import numpy as np


def computeCost(X, y, theta):
    """
       computes the cost of using theta as the parameter for linear
       regression to fit the data points in X and y
    """
    m = y.size
    h_theta = np.sum(theta.T * X, axis=1)
    J = (1 / (2 * m)) * np.sum((h_theta - y)**2)

# ====================== YOUR CODE HERE ======================
# Instructions: Compute the cost of a particular choice of theta
#               You should set J to the cost.


# =========================================================================

    return J
