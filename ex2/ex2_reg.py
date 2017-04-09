# -*- coding: utf-8 -*-
# Logistic Regression
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from matplotlib import use
use('TkAgg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from costFunctionReg import costFunctionReg
from gradientFunctionReg import gradientFunctionReg
from ml import mapFeature, plotData, plotDecisionBoundary
from scipy.optimize import minimize
from show import show
from sigmoid import sigmoid


def optimize(Lambda):

    result = minimize(costFunctionReg, initial_theta, method='L-BFGS-B',
                      jac=gradientFunctionReg, args=(X.as_matrix(), y, Lambda),
                      options={'gtol': 1e-4, 'disp': False, 'maxiter': 1000})

    return result


# Plot Boundary
def plotBoundary(theta, X, y):
    plotDecisionBoundary(theta, X.values, y.values)
    plt.title(r'$\lambda$ = ' + str(Lambda))

    # Labels and Legend
    plt.xlabel('Microchip Test 1')
    plt.ylabel('Microchip Test 2')
    show()


# Initialization

# Load Data
#  The first two columns contains the X values and the third column
#  contains the label (y).

data = pd.read_csv('ex2data2.txt', header=None, names=[1, 2, 3])
X = data[[1, 2]]
y = data[[3]]

plotData(X.values, y.values)

# Labels and Legend
plt.xlabel('Microchip Test 1')
plt.ylabel('Microchip Test 2')
show()
input("Program paused. Press Enter to continue...")


# =========== Part 1: Regularized Logistic Regression ============

# Add Polynomial Features

# Note that mapFeature also adds a column of ones for us, so the intercept
# term is handled
X = X.apply(mapFeature, axis=1)

# Initialize fitting parameters
initial_theta = np.zeros(X.shape[1])

# Set regularization parameter lambda to 1
Lambda = 0.0

# Compute and display initial cost and gradient for regularized logistic
# regression
cost = costFunctionReg(initial_theta, X, y, Lambda)

print('Cost at initial theta (zeros): {}'.format(cost))

grad = gradientFunctionReg(initial_theta, X, y, Lambda)
print('Gradient at initial theta (zeros): {}'.format(grad))

# ============= Part 2: Regularization and Accuracies =============

# Optimize and plot boundary

Lambda = 1.0
result = optimize(Lambda)
theta = result.x
cost = result.fun

# print(to screen)
print('lambda = ' + str(Lambda))
print('Cost at theta found by scipy: {}'.format(cost))
print('theta:', ["{:0.4f}".format(i) for i in theta])

input("Program paused. Press Enter to continue...")

plotBoundary(theta, X, y)

# Compute accuracy on our training set
p = np.round(sigmoid(X.dot(theta)))
acc = np.mean(np.where(p == y.T, 1, 0)) * 100
print('Train Accuracy: {}'.format(acc))

input("Program paused. Press Enter to continue...")

# ============= Part 3: Optional Exercises =============


for Lambda in np.arange(0.0, 10.1, 1.0):
    result = optimize(Lambda)
    theta = result.x
    print('lambda = ' + str(Lambda))
    print('theta:', ["{:0.4f}".format(i) for i in theta])
    plotBoundary(theta, X, y)
input("Program paused. Press Enter to continue...")
