# -*- coding: utf-8 -*-
"""
Created on Mon Oct 13 09:02:35 2014

@author: Kiran
"""

from __future__ import division

import numpy as np
import matplotlib.pylab as plt
import scipy.io as scipyio

import mathutil

def load_data(data_set_num):
  data = scipyio.loadmat('ex8data'+str(data_set_num)+'.mat')
  return data['X'], data['Xval'], data['yval'].flatten()

def estimate_gaussian(X):
  mu = np.mean(X, axis=0)
  sigma_sq = np.var(X, axis=0)
  return mu, sigma_sq

def viz_data(X, mu, sigma_sq):
  X1, X2 = np.meshgrid(np.arange(0,35,0.5), np.arange(0,35,0.5))
  X1_flattened = X1.flatten()
  X2_flattened = X2.flatten()
  new_X = np.empty((X1_flattened.size,2))
  new_X[:, 0] = X1_flattened
  new_X[:, 1] = X2_flattened
  Z = multivariate_gaussian(new_X, mu, sigma_sq)
  Z = Z.reshape(X1.shape)
  plt.hold()
  plt.plot(X[:,0], X[:,1], 'bx')
  plt.contour(X1, X2, Z, levels=10.0**(np.arange(-20.0,0,3)))
  plt.axis([0, 30, 0, 30])
  plt.xlabel('Latency ms')
  plt.ylabel('Tput Mbps') 
  plt.hold()

def multivariate_gaussian(X, mu, sigma_sq):
  if (mathutil.is_np_1d_array(sigma_sq)):
    sigma_sq = np.diag(sigma_sq)
  r, c = sigma_sq.shape
  assert(r == c)    
  inv_sigma_sq = np.linalg.inv(sigma_sq)
  X_minus_mu = X-mu
  exp_term_0 = np.dot(X_minus_mu, inv_sigma_sq)
  exp_term = np.sum(exp_term_0 * X_minus_mu, axis=1)
  dist = (2*np.pi)**(-r/2)*np.linalg.det(sigma_sq)**(-0.5)*np.exp(-exp_term)
  return dist
  
def select_threshold(y_val, p_val):
  best_eps, best_f1 = 0.0, 0.0
  true_anamoly = (y_val == 0)
  for epsilon in np.linspace(np.min(p_val), np.max(p_val), 1000):
    pred_anomaly = p_val < epsilon
    true_pos = np.count_nonzero(np.logical_and(pred_anomaly == True, true_anamoly == True))
    false_pos = np.count_nonzero(np.logical_and(pred_anomaly == True, true_anamoly == False))
    false_neg = np.count_nonzero(np.logical_and(pred_anomaly == False, true_anamoly == True))
    try:
      precision = true_pos/(true_pos + false_pos) # row-wise of the precision-recall table
    except ZeroDivisionError:
      precision = np.nan
    try:
      recall = true_pos/(true_pos + false_neg) # column-wise of the precision-recall table
    except ZeroDivisionError:
      recall = np.nan
    f1_score = 2*precision*recall/(precision+recall)
    if (f1_score > best_f1):
      best_eps, best_f1 = epsilon, f1_score
  return best_eps, best_f1

def exercise_1():
  X, x_val, y_val = load_data(1)
  mu, sigma_sq = estimate_gaussian(X)
  viz_data(X, mu, sigma_sq)
  p_val = multivariate_gaussian(x_val, mu, sigma_sq)
  eps_threshold, f1 = select_threshold(y_val, p_val)
  print(eps_threshold)
  print(f1)
  print('num of outliers: ' + str(np.count_nonzero(p_val<eps_threshold)))

def exercise_2():
  X, x_val, y_val = load_data(2)
  mu, sigma_sq = estimate_gaussian(X)
  p_val = multivariate_gaussian(x_val, mu, sigma_sq)
  eps_threshold, f1 = select_threshold(y_val, p_val)
  print(eps_threshold)
  print(f1)  
  print('num of outliers: ' + str(np.count_nonzero(p_val<eps_threshold)))
  

def process_main():
  exercise_1()
  exercise_2()
  
  
if __name__ == '__main__':
  process_main()
  