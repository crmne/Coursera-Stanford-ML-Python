# -*- coding: utf-8 -*-
import numpy as np
from computeCost import computeCost


def gradientDescent(X, y, theta, alpha, num_iters):
    """
     Performs gradient descent to learn theta
       theta = gradientDescent(x, y, theta, alpha, num_iters) updates theta by
       taking num_iters gradient steps with learning rate alpha
    """

    # Initialize some useful values
    J_history = []
    m = y.size  # number of training examples

    for i in range(num_iters):
        #   ====================== YOUR CODE HERE ======================
        # Instructions: Perform a single gradient step on the parameter vector
        #               theta.
        #
        # Hint: While debugging, it can be useful to print out the values
        #       of the cost function (computeCost) and gradient here.
        #
        h_theta = np.sum(theta.T * X, axis=1)
        new_theta = np.empty_like(theta)
        for j, theta_j in np.ndenumerate(theta):
            j = j[0]
            new_theta[j] = theta_j - alpha * \
                (1 / m) * np.sum((h_theta - y) * X[:, j])
        theta = new_theta
        J = computeCost(X, y, theta)
        # print('cost: {:0.4f}'.format(J))

        # ============================================================

        # Save the cost J in every iteration
        J_history.append(J)

    return theta, J_history
