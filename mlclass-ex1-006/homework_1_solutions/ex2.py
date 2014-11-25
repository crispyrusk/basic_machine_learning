# -*- coding: utf-8 -*-
"""
Created on Mon Aug 04 14:50:10 2014

@author: Kiran
"""

from __future__ import division
import numpy as np
import matplotlib.pylab as plt

def load_data():
  data =  np.loadtxt('ex1data2.txt', delimiter=',')
  x = data[:, 0:2]
  y = data[:, 2]
  return x, y

# normalize just the features to have zero mean and unit variance
def feature_normalize(x):
  m, num_features = x.shape
  x_mu = np.mean(x, axis=0)
  x_sigma = np.std(x, axis=0)
  x_norm = (x - x_mu)/x_sigma
  return x_norm, x_mu, x_sigma

# adds a -1 to the last dimension of theta
def get_extended_theta(theta):
  theta_extended_shape = list(theta.shape)
  theta_extended_shape[-1] += 1
  theta_extended_shape = tuple(theta_extended_shape)
  theta_extended = np.zeros(theta_extended_shape, dtype=np.float64)
  theta_extended[...,:-1] = theta
  theta_extended[...,-1] = -1.0
  return theta_extended  

# computes the cost for different thetas
def compute_cost(aug_x, y, theta):
  num_samples, num_augfeatures = aug_x.shape
  theta_extended = get_extended_theta(theta)
  if (np.ndim(theta_extended)==1):
    theta_extended = theta_extended[np.newaxis, np.newaxis, ...]
  if (np.ndim(theta_extended)==2):
    numrows, numcols = theta_extended.shape
    theta_extended = np.reshape(theta_extended, (numrows, 1, numcols))
  assert(np.ndim(theta_extended)==3)
  num_thetas, bcastdim_size, num_thetavars = theta_extended.shape
  assert(bcastdim_size == 1)
  assert(num_thetavars == num_augfeatures+1)
  aug_xy = np.concatenate((aug_x, y[:,np.newaxis]), axis=1)
  aug_xy = aug_xy[np.newaxis, ...]
  elementwise_prod = theta_extended * aug_xy
  error = np.sum(elementwise_prod, axis = -1)
  error_square = error*error
  sum_error_square = np.sum(error_square, axis = -1)
  mean_error_square = (1/(2*num_samples))*sum_error_square
  return mean_error_square


def do_gradient_descent(aug_x, y, theta_init, alpha, num_iters):
  num_samples, num_augfeatures = aug_x.shape
  theta_extended = get_extended_theta(theta_init)
  assert(np.ndim(theta_extended)==1)
  num_theta_vars, = theta_extended.shape
  aug_xy = np.concatenate((aug_x, y[:, np.newaxis]), axis=1)
  assert(num_theta_vars == (num_augfeatures+1))
  cost_history = np.zeros((num_iters+1), dtype=np.float64)
  for iter_ctr in range(num_iters):
    curr_theta = theta_extended[:-1]
    cost_history[iter_ctr] = compute_cost(aug_x, y, curr_theta)
    # for each sample x^(i), y^(i)  (theta^T x - y) = e^(i); sample_error = [e^(i)]
    sample_error = np.sum(theta_extended[np.newaxis, :]*aug_xy, axis=1)
    # jacobian \dho(J)/\dho(\theta_j) = \sum_i e^(i) \times x_j^(i)
    # in matrix form:
    sample_error_times_x = np.dot(sample_error, aug_x)        
    curr_theta = curr_theta - (alpha/num_samples)*(sample_error_times_x)
    theta_extended[:-1] = curr_theta
  cost_history[iter_ctr+1] = compute_cost(aug_x, y, curr_theta)
  return curr_theta, cost_history

def viz_gradient_descent(aug_x, y):
  num_iters = 50                                                    
  for alpha in [0.01, 0.1]:
    theta, cost_history = do_gradient_descent(aug_x, y, 
                                              np.array([0.0, 0.0, 0.0]), 
                                              alpha, num_iters)
    plt.plot(cost_history)
  plt.grid()
        
def solve_normal_equation(aug_x, y):
  psuedo_inv = np.dot(np.linalg.inv(np.dot(aug_x.transpose(), aug_x)), aug_x.transpose())
  theta = np.dot(psuedo_inv, y)
  return theta

def solve_using_gradient_descent(input_x, x_mu, x_sigma, theta_gradient_descent):
  normalized_input_x = (input_x - x_mu)/x_sigma
  output_y =  np.dot(np.concatenate((np.array([1.0]), normalized_input_x)), theta_gradient_descent)
  print(output_y)

def solve_using_normal_equation(input_x, theta_from_normal_eq):
  output_y = np.dot(theta_from_normal_eq, np.concatenate((np.array([1.0]), input_x)))
  print(output_y)
  
if __name__ == '__main__':
  
  #setup input
  x,y = load_data()
  assert(np.ndim(y)==1)
  num_samples = np.size(y)
  x_norm, x_mu, x_sigma = feature_normalize(x)
  aug_x_normalized = np.concatenate((np.ones((num_samples,1), dtype=np.float64), x_norm), axis = 1)
  aug_x = np.concatenate((np.ones((num_samples,1), dtype=np.float64), x), axis = 1)
  
  # experiement  
  mean_error_square = compute_cost(aug_x_normalized, y, np.array([[1.0, 2.0, 3.0],[1.0, 3.0, 4.0]]))
  viz_gradient_descent(aug_x_normalized, y)
               
  # gradient descent                                       
  num_iters = 50                                                    
  alpha = 0.8
  theta_gradient_descent, cost_history = do_gradient_descent(aug_x_normalized, y, 
                                            np.array([0.0, 0.0, 0.0]), 
                                            alpha, num_iters)

  # normal equation or pseudo inverse                                              
  theta_from_normal_eq = solve_normal_equation(aug_x, y)
  
  # test
  input_x = np.array([1650.0, 3.0])
  solve_using_gradient_descent(input_x, x_mu, x_sigma, theta_gradient_descent)
  solve_using_normal_equation(input_x, theta_from_normal_eq)
  
  
  
  