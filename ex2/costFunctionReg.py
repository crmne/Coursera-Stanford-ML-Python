# -*- coding: utf-8 -*-
from numpy import sum, copy
from costFunction import costFunction


def costFunctionReg(theta, X, y, Lambda):
    """
    Compute cost and gradient for logistic regression with regularization

    computes the cost of using theta as the parameter for regularized logistic
    regression and the gradient of the cost w.r.t. to the parameters.
    """
    # Initialize some useful values
    m = len(y)   # number of training examples

    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the cost of a particular choice of theta.
    #               You should set J to the cost.
    #               Compute the partial derivatives and set grad to the partial
    #               derivatives of the cost w.r.t. each parameter in theta
    theta_reg = copy(theta)
    theta_reg[0] = 0
    J = costFunction(theta, X, y) + ((Lambda / (2 * m)) * sum(theta_reg**2))
    # =============================================================

    return J
