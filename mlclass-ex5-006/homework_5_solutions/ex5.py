# -*- coding: utf-8 -*-
"""
Created on Thu Sep 18 17:22:11 2014

@author: Kiran
"""

from __future__ import division

import numpy as np
import matplotlib.pylab as plt
import scipy.io as scipyio
import scipy.optimize as scipyopt

import mathutil

def reshape_to_np_1darray(x):
  assert(mathutil.is_np_2d_array(x))
  return x.reshape((-1))

def load_data():
  data = scipyio.loadmat('ex5data1.mat')
  x, y = data['X'], data['y']
  x_test, y_test = data['Xtest'], data['ytest']
  x_cv, y_cv = data['Xval'], data['yval']
  x = reshape_to_np_1darray(x)
  y = reshape_to_np_1darray(y)
  x_test = reshape_to_np_1darray(x_test)
  y_test = reshape_to_np_1darray(y_test)
  x_cv = reshape_to_np_1darray(x_cv)
  y_cv = reshape_to_np_1darray(y_cv)
  return x, y, x_test, y_test, x_cv, y_cv

def plot_data(x, y):
  plt.plot(x, y, 'rx', markersize=10.0, linewidth=1.5)
  plt.xlabel('change in water level')
  plt.ylabel('water flowing out of the dam')
  plt.grid()
  plt.show()

def get_aug_x(x):
  num_samples = x.size
  aug_x = np.concatenate((np.ones((num_samples, 1), dtype=np.float64), 
                          x[:, np.newaxis]), 
                         axis = 1)  
  return aug_x

def compute_error_per_sample(theta, x, y):
  assert(mathutil.is_np_1d_array(theta))
  assert(theta.size == 2)
  num_samples = x.size
  aug_x = get_aug_x(x)
  h_theta_x = np.sum(theta*aug_x, axis = 1)
  error_per_sample = h_theta_x - y
  assert(mathutil.is_np_1d_array(error_per_sample))
  assert(error_per_sample.size == num_samples)
  return error_per_sample

def compute_linear_regression_cost(theta, x, y, lagrange_lambda):
  """
  Compute the linear regression cost with the regularization penalty

  .. math::
     J(\\theta) = \\frac{1}{2m} (\\sum_{i=0}^{m-1} (h_\\theta(x^{(i)}) - y^{(i)})^2 + \\lambda \sum_{j=1}^{n-1} \\theta_j^2 )
  
  :param theta: the hyper-plane params
  :type theta: list
  """
  error_per_sample = compute_error_per_sample(theta, x, y)
  num_samples = x.size
  mean_error = (0.5/num_samples)*np.sum(error_per_sample**2)
  regularization_cost = (lagrange_lambda/num_samples)*theta[1]**2
  return (mean_error + regularization_cost)
  
def compute_linear_regression_gradient(theta, x, y, lagrange_lambda):
  """
  Compute the linear regression gradient with the regularization pull
  
  .. math::
     \\frac{\\partial J(\\theta)}{\\partial \\theta_0} = \\frac{1}{m} \\sum_{i=1}^{m-1} (h_\\theta(x^{(i)}) - y^{(i)}) x_j^{(i)} \,for \,j=0
     
     \\frac{\\partial J(\\theta)}{\\partial \\theta_j} = \\frac{1}{m} \\sum_{i=1}^{m-1} (h_\\theta(x^{(i)}) - y^{(i)}) x_j^{(i)} + \\frac{\\lambda}{m} \\theta_j \,for \,j \\ge 1     
  
  :param theta: the hyper-plane params
  :type theta: list
  :param x: design matrix
  :type x: np array
  :param y: output
  :type y: np array
  """  
  error_per_sample = compute_error_per_sample(theta, x, y)
  num_samples = x.size
  aug_x = get_aug_x(x)
  inner_term = error_per_sample[:, np.newaxis] * aug_x
  summed_inner_term = np.sum(inner_term, axis=0)
  assert(mathutil.is_np_1d_array(summed_inner_term))
  assert(summed_inner_term.size == 2)
  jacobian = (1/num_samples)*summed_inner_term
  jacobian_with_reg_pull = jacobian + np.array([0.0, lagrange_lambda*theta[1]/num_samples])
  return jacobian_with_reg_pull

def train_linear_regression(initial_theta, x, y, lagrange_lambda):
  opt_res = scipyopt.minimize(compute_linear_regression_cost, 
                              initial_theta, 
                              args = (x, y, lagrange_lambda),
                              method = 'CG',
                              jac = compute_linear_regression_gradient)
  optimal_theta = opt_res.x
  return optimal_theta
                      
def test_compute_cost_and_gradient(x, y):
  theta = np.array([1.0, 1.0])
  lagrange_lambda = 1.0
  cost = compute_linear_regression_cost(theta, x, y, lagrange_lambda)
  print(cost)
  gradient = compute_linear_regression_gradient(theta, x, y, lagrange_lambda)
  print(gradient)
  
def plot_fitted_data(trained_theta, x, y):
  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.plot(x, y, 'rx', markersize=10.0, linewidth=1.5)
  ax.set_xlabel('Change in water level (x)')
  ax.set_ylabel('Water flowing out of the dam (y)')
  ax.plot(x, np.dot(get_aug_x(x), trained_theta), '--', linewidth=2.0)
  ax.grid()
  plt.show()

def plt_learning_curve(training_set_size, training_error, cv_error):
  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.plot(training_set_size, training_error, 'g')
  ax.plot(training_set_size, cv_error, 'b')
  ax.set_title('learning curves')
  ax.set_xlabel('training set size (num samples)')
  ax.set_ylabel('error')
  ax.legend(['training error','cross-valid error'])
  ax.grid()
  plt.show()

def generate_learning_curve(x, y, x_cv, y_cv):
  """
  Generates learning curves by sweeping acrosss the size of the training set
  """
  assert(mathutil.is_np_1d_array(x))
  num_samples = x.size
  assert(num_samples>0)
  training_set_size = np.arange(1, num_samples)
  training_error = np.empty_like(training_set_size)
  cv_error = np.empty_like(training_set_size)
  for idx, curr_training_set_size in enumerate(training_set_size):
    x_train = x[:curr_training_set_size]
    y_train = y[:curr_training_set_size]
    initial_theta = np.ones((2), dtype=np.float64)
    trained_theta = train_linear_regression(initial_theta, x_train, y_train, 1.0)
    training_error[idx] = compute_linear_regression_cost(trained_theta, x_train, y_train, 0.0)
    cv_error[idx] = compute_linear_regression_cost(trained_theta, x_cv, y_cv, 0.0)
  plt_learning_curve(training_set_size, training_error, cv_error)
  
def process_main():
  x, y, x_test, y_test, x_cv, y_cv = load_data()
  plot_data(x, y)
  test_compute_cost_and_gradient(x, y)
  trained_theta = train_linear_regression(np.array([1., 1.]), x ,y, 0.0)
  plot_fitted_data(trained_theta, x, y)
  generate_learning_curve(x, y, x_cv, y_cv)
  #TODO(Kiran): the polynomial regression part of ex5 is incomplete
  
  
if __name__ == '__main__':
  process_main()