# -*- coding: utf-8 -*-
"""
Created on 

Sun Aug 17 11:22:25 2014

@author: Kiran
"""

from __future__ import division

import mathutil

import matplotlib.pylab as plt
import math
import numpy as np
import scipy.optimize as opt
import scipy.io as scipyio

def load_data():
  mat_data = scipyio.loadmat('ex3data1.mat')
  x = mat_data['X']
  y = mat_data['y']
  return x, y


def display_data(x, **kwargs):
  plt.set_cmap('gray')
  nrows, ncols = x.shape
  example_width = int(kwargs.get('example_width', round(math.sqrt(ncols))))
  example_height = int(ncols / example_width)
  display_rows = int(math.floor(math.sqrt(nrows)))
  display_cols = int(math.ceil(nrows/display_rows))
  pad = 1
  display_array = -np.ones( (pad + display_rows *(example_height + pad),
                            pad + display_cols * (example_width + pad)) );
  curr_ex = 0;
  for j in range(0, display_rows):
    for i in range(0, display_cols):
      if (curr_ex >= nrows):
        break;
      max_val = np.max(np.abs(x[curr_ex, :]))
      x_splice_start = pad + j*(example_height + pad)
      y_splice_start = pad + i*(example_width + pad)
      display_array[x_splice_start:(x_splice_start+example_height),
                    y_splice_start:(y_splice_start+example_width)] = \
                    np.reshape(x[curr_ex,:], (example_height, example_width)) / max_val
      curr_ex += 1
    if (curr_ex >= nrows):
      break
  plt.imshow(display_array)
  plt.show()


def train_one_vs_all_classifier(aug_x, y, num_labels, lagrange_lambda):
  n_samples, n_features = aug_x.shape
  all_theta = np.empty((num_labels, n_features), dtype=np.float64)
  assert(mathutil.is_np_col_vector(y))
  y_arr = np.reshape(y, -1)
  for label in range(0, num_labels):
    y_label_indicator = np.zeros_like(y_arr)
    assert(np.any(y==label))
    y_label_indicator[y_arr==label] = 1
    all_theta[label, :] = train_logistic_regression(aug_x, 
                                                    y_label_indicator.astype(np.float64),
                                                    lagrange_lambda)
  return all_theta


def train_logistic_regression(aug_x, y, lagrange_lambda):
  assert(aug_x.dtype == np.float64)
  assert(y.dtype == np.float64)
  n_samples, n_features = aug_x.shape
  assert(np.all(np.logical_or((y==0), (y==1))))
  initial_theta = np.zeros((n_features), dtype=np.float64)  
  opts = {'maxiter' : None,    # default value.
          'disp' : True,    # non-default value.
          'gtol' : 1e-5,    # default value.
          'norm' : np.inf,  # default value.
          'eps' : 1.4901161193847656e-08}  # default value.  
  optimal_theta_res = opt.minimize(compute_cost, initial_theta, 
                                   args=(aug_x, y, lagrange_lambda),
                                   method = 'cg',
                                   options = opts,
                                   jac = compute_gradient)    
  print(optimal_theta_res.success)
  print(optimal_theta_res.message)
  return optimal_theta_res.x                                     
      

def compute_cost(theta, aug_x, y, lagrange_lambda):
  n_samples, n_features = aug_x.shape
  assert(np.ndim(theta) == 1)
  assert(np.size(theta) == n_features)
  with np.errstate(over = 'raise'):
    inner_term = np.dot(aug_x, theta)
  hyp_h_theta = mathutil.sigmoid_fn(inner_term) # h_Theta(x)
  assert(mathutil.is_np_1d_array(y))
  assert(mathutil.is_np_1d_array(hyp_h_theta))
  
  cost_per_sample = -y*np.log(hyp_h_theta) \
                    -(1-y)*np.log(1-hyp_h_theta)
  assert(not np.any(np.isnan(cost_per_sample)))
  average_cost = (1/n_samples)*np.sum(cost_per_sample)
  cost_plus_regularization = average_cost + \
                             (lagrange_lambda/(2*n_samples))*np.sum(theta**2)
  assert(not np.isnan(cost_plus_regularization))
  return cost_plus_regularization


def compute_gradient(theta, aug_x, y, lagrange_lambda):
  n_samples, n_features = aug_x.shape
  assert(np.ndim(theta)==1)
  assert(np.size(theta) == n_features)
  inner_term = np.dot(aug_x, theta)
  hyp_h_theta = mathutil.sigmoid_fn(inner_term)
  assert(mathutil.is_np_1d_array(y))
  assert(mathutil.is_np_1d_array(hyp_h_theta))
  jac = (1/n_samples) * np.dot(aug_x.transpose(),(hyp_h_theta - y))
  assert(np.ndim(jac)==1)
  assert(np.size(jac)==n_features)
  regularization_pull = (lagrange_lambda/n_samples)*theta
  jac_with_regularization = jac + regularization_pull
  assert(not np.any(np.isnan(jac_with_regularization)))
  return jac_with_regularization


#TODO(Kiran): Move to mathutil if needed  
def get_normalized_aug_x(aug_x):
  assert(np.ndim(aug_x)==2)
  n_samples, n_features = aug_x.shape
  sample_mean = np.mean(aug_x, axis=0)
  sample_std_dev = np.std(aug_x, axis=0)
  assert(sample_mean[0]==1.0)
  assert(sample_std_dev[0]==0.0)
  sample_mean[0]=0.0
  sample_std_dev[0]=1.0
  cols_with_zero_std = (sample_std_dev==0)
  cols_with_nonzero_std = np.logical_not(cols_with_zero_std)
  norm_aug_x = (aug_x[:, cols_with_nonzero_std] - sample_mean[cols_with_nonzero_std])
  with np.errstate(divide = 'raise'):
    norm_aug_x /= sample_std_dev[cols_with_nonzero_std]
  norm_sample_mean = sample_mean[cols_with_nonzero_std]
  norm_sample_std_dev = sample_std_dev[cols_with_nonzero_std]
  return norm_aug_x, norm_sample_mean, norm_sample_std_dev

def get_num_labels(y):
  unq_labels = np.unique(y)
  assert(unq_labels[0] == 0)
  assert(unq_labels[-1] == (np.size(unq_labels)-1))
  return np.size(unq_labels)


'''
note that this still a linear separation classifier 
(so do theta dot x and find the best match (highest value among the classes for theta dot x))
'''
def test_predict_one_vs_all_test(aug_x, y, num_labels):
  lagrange_lambda = 0.1
  all_theta = train_one_vs_all_classifier(aug_x, y, num_labels, lagrange_lambda)
  n_classes, n_features_1 = all_theta.shape
  n_samples, n_features_2 = aug_x.shape
  assert(n_features_1 == n_features_2)
  three_d_product = all_theta[...,np.newaxis]*aug_x.transpose() 
  a, b, c = three_d_product.shape
  assert(a == n_classes)
  assert(b == n_features_1)
  assert(c == n_samples)
  predict_theta_x = np.sum(three_d_product, axis=1) # note that this still a linear separation classifier
  predicted_class = np.argmax(predict_theta_x, axis=0)
  y_arr = np.reshape(y, -1)
  classification_accuracy = np.count_nonzero(predicted_class==y_arr)/np.size(y_arr)
  print('classification accuracy on training set: '+str(classification_accuracy))
  
def process_main():
  x, y = load_data()
  y[y[:,:]==10]=0
  num_labels = get_num_labels(y)
  nrows, ncols = x.shape
  rand_indices = np.random.permutation(nrows)
  sel = x[rand_indices[0:100], :]
  display_data(sel)
  aug_x = mathutil.get_aug_x(x)
  test_predict_one_vs_all_test(aug_x, y, num_labels)

  
  
if __name__ == '__main__':
  process_main()