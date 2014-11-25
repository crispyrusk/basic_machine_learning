# -*- coding: utf-8 -*-
"""
Created on Mon Aug 11 13:10:58 2014

@author: Kiran
"""

from __future__ import division
import numpy as np
import matplotlib.pylab as plt
import scipy.optimize as opt


def load_data():
  data = np.loadtxt('ex2data2.txt', delimiter = ',')
  x = data[:,0:-1]
  y = data[:,-1]
  num_samples, num_real_features = x.shape
  assert(np.size(y)==num_samples)                          
  aug_x = np.concatenate((np.ones((num_samples, 1), dtype=np.float64), 
                          x), axis=1)
  return aug_x, y                            


#data expansion
def map_features(x):
  num_samples, num_features = x.shape
  assert(num_features == 2)
  x_1 = x[:, 0]
  x_2 = x[:, 1]
  # higher order polynomial representation
  higher_dim_x = np.ones((num_samples, 28))
  for poly_degree in range(1,7):
    start_index = poly_degree*(poly_degree+1)/2
    for x_2_degree in range(0, poly_degree+1):
      x_1_degree = poly_degree - x_2_degree
      higher_dim_x[:, start_index+x_2_degree] = \
        x_1**(x_1_degree)*x_2**(x_2_degree)
  return higher_dim_x


def sigmoid_fn(x):
  return 1/(1+np.exp(-x))


def compute_h_theta(theta, x):
  num_samples, num_features = x.shape
  assert(np.ndim(theta) == 1)
  assert(np.size(theta) == num_features)
  return sigmoid_fn(np.dot(x, theta))


def compute_cost(theta, lagrange_lambda, x, y):
  num_samples, num_features = x.shape
  h_theta = compute_h_theta(theta, x)
  sample_error = -y*np.log(h_theta) - (1-y)*np.log(1-h_theta)
  mean_error = (1/num_samples)*np.sum(sample_error)
  regularization = \
    (lagrange_lambda/(2*num_samples))*np.inner(theta[1:], theta[1:])
  return mean_error + regularization
  
  
def compute_gradient(theta, lagrange_lambda, x, y):
  num_samples, num_features = x.shape
  h_theta = compute_h_theta(theta, x)
  sample_error = h_theta - y
  jacobian = np.dot(sample_error, x)
  regularization_pull = (lagrange_lambda/num_samples)*theta
  regularization_pull[0] = 0.0
  return jacobian + regularization_pull
  

def compute_optimal_theta(theta_init, lagrange_lambda, x, y):
  optima_result = opt.minimize(fun = compute_cost, x0 = theta_init,
                               args = (lagrange_lambda, x, y), 
                               method = 'Newton-CG',
                               jac = compute_gradient)
  return optima_result.x                               


def plt_data_with_decisionboundary(aug_x, y, theta):
  num_samples, num_features = aug_x.shape
  assert(num_features == 3)
  positive_samples = (y==1)
  negative_samples = (y==0)
  assert(np.all(np.logical_or(positive_samples, negative_samples)))
  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.plot(aug_x[positive_samples, 1], aug_x[positive_samples, 2], 'k+', 
          label = '+ive samples')
  ax.plot(aug_x[negative_samples, 1], aug_x[negative_samples, 2], 'ko', 
          label = '-ive samples')
  grid_x, grid_y = np.meshgrid(np.arange(-1.5, 1.5, 0.1),
                               np.arange(-1.5, 1.5, 0.1))
  grid_points_xy_matrix = \
    np.concatenate((
                    grid_x.reshape((np.size(grid_x),1)),
                    grid_y.reshape((np.size(grid_y),1))
                    ), axis=1)
  data_expand_grid_points = map_features(grid_points_xy_matrix)                    
  value_fn = np.dot(data_expand_grid_points, theta)
  value_fn_2D = value_fn.reshape(grid_x.shape)
  ax.contour(grid_x, grid_y, value_fn_2D, linewidths=2, levels=[0])
  ax.grid()
  ax.legend()
  plt.show()

def process_main():
  aug_x, y = load_data();  
  higher_dim_x = map_features(aug_x[:,1:])
  num_sample, num_features = higher_dim_x.shape
  theta_init = np.zeros((num_features), dtype=np.float64)
  lagrange_lambda = 10 # underfit #0.000001 # overfit
  cost = compute_cost(theta_init, lagrange_lambda, higher_dim_x, y)
  gradient = compute_gradient(theta_init, lagrange_lambda, higher_dim_x, y)
  optimal_theta = compute_optimal_theta(theta_init, lagrange_lambda, 
                                        higher_dim_x, y)
  plt_data_with_decisionboundary(aug_x, y, optimal_theta)                                          
    
 
if __name__=='__main__':
  process_main()