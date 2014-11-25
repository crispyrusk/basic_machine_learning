# -*- coding: utf-8 -*-
"""
Created on Sat Aug 30 06:07:52 2014

@author: Kiran
"""

from __future__ import division

import mathutil

import numpy as np
import copy
import scipy.io as scipyio
import scipy.optimize as scipyopt
import matplotlib.pylab as plt
import math

RANDOM_INIT_EPSILON = 0.12
GRADIENT_CHECKING_EPSILON = 1e-4

def load_data():
  mat_data = scipyio.loadmat('ex4data1.mat')
  x = mat_data['X']
  y = mat_data['y']
  return x, y

def get_nn_weights_from_mat():
  loaded_mat = scipyio.loadmat('ex4weights.mat')  
  theta_matrix_1 = loaded_mat['Theta1']
  theta_matrix_2 = loaded_mat['Theta2']
  return (theta_matrix_1, theta_matrix_2)

def indicatorize_class(y):
  assert(mathutil.is_np_col_vector(y))
  y -= 1
  num_samples = np.size(y)
  y_classes = np.unique(y).reshape((-1))
  min_class_num = np.min(y_classes)
  max_class_num = np.max(y_classes)
  assert(min_class_num>=0)
  class_indicator_y = np.zeros((num_samples, max_class_num+1), dtype=np.float64)
  class_indicator_y[np.arange(0, num_samples), y[:,-1]] = 1.0
  return class_indicator_y

def compute_htheta_by_feedforward(theta_transfers, x):
  activations = x.transpose()
  num_features, num_samples = activations.shape
  activations_all_layers = [activations]
  z_all_layers = []
  for theta_transfer in theta_transfers:
    activations_withbias = \
      np.concatenate((np.ones((1,num_samples), dtype=np.float64),
                      activations), axis = 0)
    z = np.dot(theta_transfer, activations_withbias) 
    activations = mathutil.sigmoid_fn(z)
    z_all_layers.append(z)
    activations_all_layers.append(activations)
  h_theta = activations.transpose()
  return (h_theta, z_all_layers, activations_all_layers)

def compute_entropy_cost(h_theta, class_indicator_y):
  num_samples, num_classes = class_indicator_y.shape
  assert(h_theta.shape == class_indicator_y.shape)
  entropy_cost_per_sample_per_class = \
                              -class_indicator_y*np.log(h_theta) \
                              -(1.0-class_indicator_y)*np.log(1.0-h_theta)
  entropy_cost = (1/num_samples)*np.sum(entropy_cost_per_sample_per_class)
  return entropy_cost

# NOTE(Kiran): The regularization cost should not include the parameters
#              of the bias, unit, i.e., the 0th column of theta_transfer
def compute_regularization_cost(theta_transfers, num_samples, lagrange_lambda):
  theta_squared_sum = 0.0
  for theta_transfer in theta_transfers:
    theta_squared_sum += np.sum(theta_transfer[:,1:]**2)
  regularization_cost = (lagrange_lambda/(2*num_samples))*theta_squared_sum
  return regularization_cost
  
  
def compute_cost_via_feedforward(theta_transfers, x, class_indicator_y, 
                                    lagrange_lambda):
  h_theta, __, __ = compute_htheta_by_feedforward(theta_transfers, x)
  entropy_cost = compute_entropy_cost(h_theta, class_indicator_y)
  num_samples, num_classes = class_indicator_y.shape
  regularization_cost = compute_regularization_cost(theta_transfers, num_samples, lagrange_lambda)
  cost = entropy_cost + regularization_cost
  return cost               

def compute_errors_via_backpropagation(theta_transfers, z_all_layers, h_theta, class_indicator_y):
  num_samples_1, num_classes_1 = h_theta.shape
  num_samples_2, num_classes_2 = class_indicator_y.shape
  assert(num_samples_1 == num_samples_2)
  assert(num_classes_1 == num_classes_2)
  h_theta_T = h_theta.transpose()
  class_indicator_y_T = class_indicator_y.transpose()
  error_in_layer = h_theta_T - class_indicator_y_T
  z_trunc = z_all_layers[:-1]
  theta_transfers_trunc = theta_transfers[1:]
  assert(len(theta_transfers_trunc)==len(z_trunc))
  error_in_layers = [error_in_layer]
  for theta_transfer, z_current_layer in \
  reversed(zip(theta_transfers_trunc, z_trunc)):
    error_from_next_layer = error_in_layer  # \delta^(l+1)
    error_withbias_intermediate = np.dot(theta_transfer.transpose(), error_from_next_layer)
    assert(error_withbias_intermediate.shape[1] == num_samples_1)
    assert(z_current_layer.shape[1] == num_samples_1)
    error_in_layer = error_withbias_intermediate[1:,:] * mathutil.gradient_sigmoid_fn(z_current_layer)
    error_in_layers.insert(0, error_in_layer)
  return error_in_layers

#NOTE (Kiran): While adding the regularization pull to the gradient, make
#              sure that you dont add the theta_ij corres. to the bias units
def compute_gradient_via_backpropagation(theta_transfers, x, class_indicator_y, lagrange_lambda):
  h_theta, z_all_layers, activations_all_layers =  \
    compute_htheta_by_feedforward(theta_transfers, x)
  num_samples, num_classes = h_theta.shape
  error_in_layers = compute_errors_via_backpropagation(theta_transfers, z_all_layers, h_theta, class_indicator_y)
  jacob_thetas = []
  assert(len(error_in_layers)==len(activations_all_layers)-1)
  for error_in_next_layer, activation_in_layer, theta_transfer in zip(error_in_layers, activations_all_layers, theta_transfers):
    activation_withbias_in_layer = np.concatenate((np.ones((1,num_samples), dtype=np.float64), activation_in_layer))
    all_samples_delta_in_layer = error_in_next_layer[:, np.newaxis, :] * activation_withbias_in_layer
    averaged_delta_in_layer = (1/num_samples)*np.sum(all_samples_delta_in_layer, axis=-1)
    regularization_pull = np.zeros_like(theta_transfer, dtype=np.float64)
    regularization_pull[:,1:] = (lagrange_lambda/num_samples)*theta_transfer[:,1:]
    jacob_thetas.append(averaged_delta_in_layer+regularization_pull)
  return jacob_thetas

def compute_gradient_via_epsilon_approx(theta_transfers, x, class_indicator_y, lagrange_lambda):
  jacob_approx_thetas = []
  for theta_idx in range(0,len(theta_transfers)):
    n_r, n_c = theta_transfers[theta_idx].shape
    jacob_approx = np.empty_like(theta_transfers[theta_idx], dtype=np.float64)
    for r in range(0, n_r):
      for c in range(0, n_c):
        cost_perturbed = []
        for epsilon_direction in [-1.0, 1.0]:
          temp_theta_transfers = copy.deepcopy(theta_transfers)
          temp_theta_transfer = temp_theta_transfers[theta_idx]
          temp_theta_transfer[r,c] += epsilon_direction*GRADIENT_CHECKING_EPSILON
          cost_perturbed.append( compute_cost_via_feedforward(temp_theta_transfers, 
                                                                 x, 
                                                                 class_indicator_y, 
                                                                 lagrange_lambda) )
        jacob_approx[r,c] = (cost_perturbed[1] - cost_perturbed[0])/(2*GRADIENT_CHECKING_EPSILON)

    jacob_approx_thetas.append(jacob_approx)
  return jacob_approx_thetas                                                                       
        
def random_initialize_thetas(theta_transfers):
  rand_init_theta_transfers = []
  for theta_transfer in theta_transfers:
    rand_theta = (np.random.rand(*theta_transfer.shape)*2*RANDOM_INIT_EPSILON) \
                 - RANDOM_INIT_EPSILON
    rand_init_theta_transfers.append(rand_theta)
  return rand_init_theta_transfers

def check_using_numerical_gradient(jacob_thetas, theta_transfers, x, class_indicator_y, lagrange_lambda):
  jacob_approx_thetas = compute_gradient_via_epsilon_approx(theta_transfers, x, class_indicator_y, lagrange_lambda)
  theta_1 = jacob_thetas[0]
  theta_1_numerical = jacob_approx_thetas[0]
  print(np.linalg.norm(theta_1_numerical-theta_1)/np.linalg.norm(theta_1_numerical+theta_1))
  theta_2 = jacob_thetas[1]  
  theta_2_numerical = jacob_approx_thetas[1]
  print(np.linalg.norm(theta_2_numerical-theta_2)/np.linalg.norm(theta_2_numerical+theta_2))

def get_theta_transfers_from_flattened_version(theta_transfers_flattened, theta_transfer_shapes):
  offset = 0
  assert(mathutil.is_np_1d_array(theta_transfers_flattened))
  theta_transfers = []
  for theta_transfer_shape in theta_transfer_shapes:
    r, c = theta_transfer_shape
    theta_transfer = theta_transfers_flattened[offset:(offset+r*c)].reshape(r,c)
    theta_transfers.append(theta_transfer)
    offset += r*c
  assert(offset==theta_transfers_flattened.size)
  return theta_transfers    

def get_flattened_theta_transfers(theta_transfers, theta_transfer_shapes):
  for theta_transfer, theta_transfer_shape in zip(theta_transfers, theta_transfer_shapes):
    assert(theta_transfer.shape==theta_transfer_shape)
  num_elements = sum(r*c for r,c in theta_transfer_shapes)
  theta_transfers_flattened = np.zeros((num_elements), dtype=np.float64)
  offset = 0
  for theta_transfer in theta_transfers:
    r,c = theta_transfer.shape
    theta_transfers_flattened[offset:(offset+r*c)] = theta_transfer.reshape((-1))
    offset += r*c
  return theta_transfers_flattened    
  
def compute_cost_for_optimizer(theta_transfers_flattened, 
                               theta_transfer_shapes,
                               x, class_indicator_y, lagrange_lambda):
  theta_transfers =  get_theta_transfers_from_flattened_version(theta_transfers_flattened, theta_transfer_shapes)                                
  return compute_cost_via_feedforward(theta_transfers, x, class_indicator_y, lagrange_lambda)

def compute_gradient_for_optimizer(theta_transfers_flattened, 
                                   theta_transfer_shapes,
                                   x, class_indicator_y, lagrange_lambda):
  theta_transfers = get_theta_transfers_from_flattened_version(theta_transfers_flattened, theta_transfer_shapes)                                     
  jacob_thetas = compute_gradient_via_backpropagation(theta_transfers, x, class_indicator_y, lagrange_lambda)
  theta_transfers_flattened = get_flattened_theta_transfers(jacob_thetas, theta_transfer_shapes)
  return theta_transfers_flattened
  

def train_nn(x, class_indicator_y, lagrange_lambda, init_theta_transfers):
  assert(x.dtype == np.float64)
  assert(class_indicator_y.dtype == np.float64)
  theta_transfer_shapes = [theta_transfer.shape for theta_transfer in init_theta_transfers]
  init_theta_transfer_flattened = get_flattened_theta_transfers(init_theta_transfers, theta_transfer_shapes)
  optimal_theta_res = scipyopt.minimize(compute_cost_for_optimizer, 
                                        init_theta_transfer_flattened, 
                                        args=(theta_transfer_shapes, x, class_indicator_y, lagrange_lambda),
                                        method='cg',
                                        jac = compute_gradient_for_optimizer
                                        )
  print(optimal_theta_res)
  optimal_theta = optimal_theta_res.x
  optimal_theta_transfers = get_theta_transfers_from_flattened_version(optimal_theta, theta_transfer_shapes)
  return optimal_theta_transfers

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

def check_consistency(theta_transfers, x, class_indicator_y):
  num_samples_x, num_features_x = x.shape
  num_samples_y, num_classes_y = class_indicator_y.shape
  assert(x.dtype==np.float64)
  assert(class_indicator_y.dtype==np.float64)
  assert(num_samples_x == num_samples_y)
  assert(isinstance(theta_transfers, list))
  for theta_transfer in theta_transfers:
    assert(theta_transfer.dtype==np.float64)
  input_layer_transfer_matrix = theta_transfers[0]
  m_input_layer, n_input_layer = input_layer_transfer_matrix.shape
  assert(n_input_layer == (num_features_x+1))
  output_layer_transfer_matrix= theta_transfers[-1]
  m_output_layer, n_output_layer = output_layer_transfer_matrix.shape
  assert(m_output_layer == num_classes_y)

def process_main():
  x, y = load_data()
  class_indicator_y = indicatorize_class(y)
  theta_matrix_1, theta_matrix_2 = get_nn_weights_from_mat()
  theta_transfers = [theta_matrix_1, theta_matrix_2]
  check_consistency(theta_transfers, x, class_indicator_y)
  nrows, ncols = x.shape
  lagrange_lambda = 1.0
  cost = compute_cost_via_feedforward(theta_transfers, x, 
                                         class_indicator_y, lagrange_lambda)
  print(cost)

  rand_init_theta_transfers = random_initialize_thetas(theta_transfers)  
  optimal_theta_transfers = train_nn(x, class_indicator_y, lagrange_lambda, rand_init_theta_transfers)
  #jacob_thetas = compute_gradient_via_backpropagation(theta_transfers, x, class_indicator_y, lagrange_lambda)
  #check_using_numerical_gradient(jacob_thetas, theta_transfers, x, class_indicator_y, lagrange_lambda)
  h_theta, __, __ = compute_htheta_by_feedforward(optimal_theta_transfers, x)
  predicted_class = np.argmax(h_theta, axis=1)
  true_class= np.argmax(class_indicator_y, axis=1)
  print(np.count_nonzero(predicted_class==true_class)/np.size(true_class))
    

if __name__ == '__main__':
  process_main()
  
  