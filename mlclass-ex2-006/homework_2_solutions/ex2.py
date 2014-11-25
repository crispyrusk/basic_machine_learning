# -*- coding: utf-8 -*-
"""
Created on Fri Aug 08 15:03:58 2014

@author: Kiran
"""

from __future__ import division
import numpy as np
import matplotlib.pylab as plt
import scipy.optimize as opt

def load_data():
  data = np.loadtxt('ex2data1.txt', delimiter = ',')
  x = data[:, 0:2]
  y = data[:, 2]
  return x, y


def plot_data(x, y):
  assert(np.ndim(x)==2)
  r, c = x.shape
  assert(c==2)
  positive_class = (y==1)
  negative_class = (y==0)
  assert(np.all(np.logical_or(positive_class, negative_class)))
  plt.figure()
  plt.plot(x[positive_class, 0], x[positive_class, 1], 'k+')
  plt.plot(x[negative_class, 0], x[negative_class, 1], 'ko')
  plt.grid()
  plt.show()


def sigmoid_fn(x):
  return 1/(1+np.exp(-x))


def compute_hypothesis_logisticregression(aug_x, theta):
  num_samples, num_features = aug_x.shape
  hypothesis =  sigmoid_fn(np.dot(aug_x, theta))
  assert(np.ndim(hypothesis)==1)
  assert(np.size(hypothesis)==num_samples)
  return hypothesis


def plt_sigmoid_fn():
  x = np.arange(-20.0, 20.0, dtype=np.float64)
  y = sigmoid_fn(x)
  plt.figure()
  plt.plot(x, y)
  plt.grid()
  plt.show()  

  
def get_aug_x(x):
  num_samples, num_features = x.shape
  aug_x = np.concatenate((np.ones((num_samples, 1), dtype=np.float64), 
                          x), axis = 1)
  return aug_x


def compute_cost(theta, aug_x, y):
  num_samples, num_features = aug_x.shape
  assert(np.ndim(theta)==1)
  assert(np.size(theta)==num_features)
  h_theta = compute_hypothesis_logisticregression(aug_x, theta)
  return compute_cost_with_htheta(h_theta, y, num_samples)    


def compute_gradient(theta, aug_x, y):
  num_samples, num_features = aug_x.shape
  assert(np.ndim(theta)==1)
  assert(np.size(theta)==num_features)  
  h_theta = compute_hypothesis_logisticregression(aug_x, theta)
  return compute_gradient_with_htheta(h_theta, aug_x, y, num_samples, num_features)
    

def compute_cost_with_htheta(h_theta, y, num_samples):
  entropy_cost_per_sample = -y*np.log(h_theta) - (1-y)*np.log(1-h_theta)
  return (1/num_samples)*np.sum(entropy_cost_per_sample)
    
  
def compute_gradient_with_htheta(h_theta, aug_x, y, num_samples, num_features):
  error_per_sample = (h_theta - y)
  error_times_x = np.dot(error_per_sample, aug_x)
  assert(np.size(error_times_x)==num_features)
  mean_error_times_x =  (1/num_samples)*error_times_x
  return mean_error_times_x
    
  
def compute_cost_and_gradient(aug_x, y, theta):
  num_samples, num_features = aug_x.shape
  assert(np.ndim(y)==1)
  assert(num_samples == np.size(y))
  assert(np.ndim(theta)==1)
  assert(np.size(theta)==num_features)
  h_theta = compute_hypothesis_logisticregression(aug_x, theta)
  average_entropy_cost = compute_cost_with_htheta(h_theta, y, num_samples)
  gradient = compute_gradient_with_htheta(h_theta, aug_x, y, num_samples, num_features)
  return average_entropy_cost, gradient


def check_consistency_of_data(aug_x, y):
  num_samples, num_features = aug_x.shape
  assert(np.ndim(y)==1)
  assert(num_samples == np.size(y))


def compute_optimal_theta(aug_x, y, theta_init):
  opt_result = opt.minimize(fun = compute_cost, x0 = theta_init,
                               args=(aug_x, y), method='Newton-CG',
                               jac = compute_gradient)
  return opt_result.x

def plot_data_with_decision_boundary(x, y, theta):
  fig = plt.figure()
  ax = fig.add_subplot(111)
  r, c = x.shape
  assert(c==2)
  positive_class = (y==1)
  negative_class = (y==0)
  assert(np.all(np.logical_or(positive_class, negative_class)))
  ax.plot(x[positive_class, 0], x[positive_class, 1], 'k+', label = 'positive class')
  ax.plot(x[negative_class, 0], x[negative_class, 1], 'ko', label = 'negative class')
  decision_boundary_x = np.array([np.min(x[:,0]), np.max(x[:,0])])
  assert(np.size(theta)==3)
  decision_boundary_y = (-1/theta[2])*(theta[0] + theta[1]*decision_boundary_x)
  ax.plot(decision_boundary_x, decision_boundary_y, 'b', label = 'decision boundary')
  ax.legend()
  ax.grid()
  plt.show()
  
def process_main():
  x, y = load_data()
  plot_data(x, y)
  plt_sigmoid_fn()
  aug_x = get_aug_x(x)
  check_consistency_of_data(aug_x, y)
  num_samples, num_features = aug_x.shape
  theta_init = np.zeros((num_features), dtype=np.float64)
  cost, gradient = compute_cost_and_gradient(aug_x, y, theta_init)
  optimal_theta = compute_optimal_theta(aug_x, y, theta_init)
  optimal_cost, gradient_at_optima = compute_cost_and_gradient(aug_x, y, optimal_theta)
  plot_data_with_decision_boundary(x, y, optimal_theta)  
  
if __name__ == '__main__':
  process_main()
