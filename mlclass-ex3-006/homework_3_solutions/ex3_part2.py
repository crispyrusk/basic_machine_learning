# -*- coding: utf-8 -*-
"""
Created on Wed Aug 27 09:51:45 2014

@author: Kiran
"""

from __future__ import division
import numpy as np
import scipy.io as scipyio
import mathutil

def load_data():
  mat_data = scipyio.loadmat('ex3data1.mat')
  x = mat_data['X']
  y = mat_data['y']
  return x, y

def get_nn_weights_from_mat():
  loaded_mat = scipyio.loadmat('ex3weights.mat')  
  theta_matrix_1 = loaded_mat['Theta1']
  theta_matrix_2 = loaded_mat['Theta2']
  return theta_matrix_1, theta_matrix_2
  
def run_feedforward_nn_for_sample(theta_matrix_1, theta_matrix_2, sample):
  assert(mathutil.is_np_1d_array(sample))
  n_features = np.size(sample)
  output_dim_1, input_dim_1 = theta_matrix_1.shape
  output_dim_2, input_dim_2 = theta_matrix_2.shape
  assert(n_features+1 == input_dim_1)
  a_1 = np.concatenate(([1.0], sample))
  z_2= np.dot(theta_matrix_1, a_1)
  a_withoutbias_2 = mathutil.sigmoid_fn(z_2)
  a_2 = np.concatenate(([1.0], a_withoutbias_2))
  assert( np.size(a_2) == input_dim_2 )
  z_3 = np.dot(theta_matrix_2, a_2)
  predicted_output_vector = mathutil.sigmoid_fn(z_3)
  return np.argmax(predicted_output_vector) + 1
  
def run_feedforward_nn_for_samples(theta_matrix_1, theta_matrix_2, samples, true_output):
  assert(mathutil.is_np_2d_array(samples))
  num_samples, num_features = samples.shape
  predicted_y = np.empty((num_samples,))
  for row_idx, sample in enumerate(samples):
    predicted_y[row_idx] = run_feedforward_nn_for_sample(theta_matrix_1, theta_matrix_2, sample)
  num_correctly_classified = np.count_nonzero(true_output[:,0] == predicted_y)
  print(num_correctly_classified/num_samples)
  
if __name__ == '__main__':
  x, y = load_data()
  theta_matrix_1, theta_matrix_2 = get_nn_weights_from_mat()
  run_feedforward_nn_for_samples(theta_matrix_1, theta_matrix_2, x, y)
  
  