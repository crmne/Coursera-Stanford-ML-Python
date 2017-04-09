# -*- coding: utf-8 -*-
import numpy as np


def normalEqn(X, y):
    """ Computes the closed-form solution to linear regression
       normalEqn(X,y) computes the closed-form solution to linear
       regression using the normal equations.
    """
    theta = 0
# ====================== YOUR CODE HERE ======================
# Instructions: Complete the code to compute the closed form solution
#               to linear regression and put the result in theta.
#

# ---------------------- Sample Solution ----------------------

    # theta = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(y)
    theta = np.linalg.solve(X.T.dot(X), X.T.dot(y))
# -------------------------------------------------------------

    return theta

# ============================================================
