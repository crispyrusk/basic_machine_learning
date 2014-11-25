# -*- coding: utf-8 -*-
"""
Created on Wed Oct 15 09:08:50 2014

@author: Kiran
"""

from __future__ import division

import numpy as np
import matplotlib.pylab as plt
import scipy.io as scipyio
import scipy.optimize as scipyopt

import mathutil

def load_data(data_filename):
  data = scipyio.loadmat(data_filename)
  return data

def check_data(Y, R):
  print('average rating for movie 0 (Toy Story):' + str(Y[0,R[0,:]==1].mean()))
  plt.imshow(Y, extent=[0, 1, 0, 1], aspect='auto')
  plt.title('ratings')
  plt.xlabel('users')
  plt.ylabel('movies')
  plt.imshow(R, extent=[0, 1, 0, 1], aspect='auto')
  plt.title('rating indicator')
  plt.xlabel('users')
  plt.ylabel('movies')  
  plt.show()

def flatten_X_Theta(X, Theta):
  X_Theta_vertical_stacked = np.concatenate((X, Theta), axis=0)
  return X_Theta_vertical_stacked.flatten()

def unflatten_X_Theta(X_Theta_flattened, num_users, num_movies, num_features):
  assert(mathutil.is_np_1d_array(X_Theta_flattened))
  assert(X_Theta_flattened.size == (num_users+num_movies)*num_features)
  X = X_Theta_flattened[0:(num_movies*num_features)].reshape((num_movies, num_features))
  Theta = X_Theta_flattened[(num_movies*num_features):].reshape((num_users, num_features))
  return X, Theta
  
def compute_cofi_cost(X_Theta_flattened, Y, R, 
                      num_users, num_movies, num_features, lagrange_lambda):
  X, Theta = unflatten_X_Theta(X_Theta_flattened, num_users, num_movies, num_features)
  X_Theta_outer_prod = np.dot(X, Theta.transpose())
  error = X_Theta_outer_prod - Y
  error_for_valid_rating = error[R==1]
  error_sq_for_valid_rating = error_for_valid_rating**2
  cost_J = 0.5*error_sq_for_valid_rating.sum()
  regularization_penalty_X = 0.5*lagrange_lambda*(X**2).sum()
  regularization_penalty_Theta = 0.5*lagrange_lambda*(Theta**2).sum()
  total_cost = cost_J + regularization_penalty_X + regularization_penalty_Theta
  return total_cost

def compute_cofi_gradient(X_Theta_flattened, Y, R, 
                      num_users, num_movies, num_features, lagrange_lambda):
  X, Theta = unflatten_X_Theta(X_Theta_flattened, num_users, num_movies, num_features)
  X_Theta_outer_prod = np.dot(X, Theta.transpose())
  error = X_Theta_outer_prod - Y
  error[R!=1] = 0.0
  # TODO (Kiran): Need to double check if this computation is correct
  grad_X = np.dot(error, Theta)
  grad_X_with_regularization_pull = grad_X + lagrange_lambda*X
  grad_Theta = np.dot(error.transpose(), X)
  grad_Theta_with_regularization_pull = grad_Theta + lagrange_lambda*Theta
  return flatten_X_Theta(grad_X_with_regularization_pull, grad_Theta_with_regularization_pull)
  
  
def do_part2():
  data = load_data('ex8_movies.mat')
  Y, R = data['Y'], data['R']
  data_trained = load_data('ex8_movieParams.mat')
  X = data_trained['X']
  Theta = data_trained['Theta']
  #r1, c1 = X.shape # shape == num_movies \times num_features
  #r2, c2 = Theta.shape # shape == num_users \times num_features
  # the outer product of X and Theta, X.Theta.transpose() 
  # should have the shape of Y 
  num_users, num_movies, num_features = 4, 5, 3
  X = X[0:num_movies, 0:num_features]
  Y = Y[0:num_movies, 0:num_users]
  R = R[0:num_movies, 0:num_users]
  Theta = Theta[0:num_users, 0:num_features]
  print('error on reduced data set with not regularization')
  print(
    compute_cofi_cost(flatten_X_Theta(X, Theta), Y, R, num_users, num_movies, num_features, 0)    
    )
  print('error on reduced data set with regularization lambda = 1.5')
  print(
    compute_cofi_cost(flatten_X_Theta(X, Theta), Y, R, num_users, num_movies, num_features, 1.5)    
    )

def normalize_ratings(Y, R):
  normalized_Y = np.copy(Y)
  mean_Y = np.empty(Y.shape[0], dtype=np.float64)
  for row_idx in range(0, normalized_Y.shape[0]):
    mean_Y[row_idx] = normalized_Y[row_idx,R[row_idx,:]==1].mean()
    normalized_Y[row_idx, R[row_idx,:]==1] = normalized_Y [row_idx, R[row_idx,:]==1] - mean_Y[row_idx]
  return normalized_Y, mean_Y

def compute_optimal_x_theta(initial_X, initial_Theta, 
                            Y, R, 
                            num_users, num_movies, num_features, 
                            lagrange_lambda):
  init_X_Theta_flattened = flatten_X_Theta(initial_X, initial_Theta)
  opt_re = scipyopt.minimize(compute_cofi_cost, init_X_Theta_flattened, 
                             args=(Y, R, num_users, num_movies, num_features, lagrange_lambda),
                             method='cg',
                             jac = compute_cofi_gradient)
  opt_X_Theta_flattened = opt_re.x
  opt_X, opt_Theta = unflatten_X_Theta(opt_X_Theta_flattened, 
                                       num_users, num_movies, num_features)
  return opt_X, opt_Theta
                      

#TODO(Kiran): This part is incomplete because I havent added user preference part
def do_part7():
  data = load_data('ex8_movies.mat')
  Y, R = data['Y'].astype(np.float64), data['R'].astype(np.float64)
  normalized_Y, mean_Y = normalize_ratings(Y, R)
  num_movies, num_users = Y.shape
  num_features = 10
  X = np.random.randn(num_movies, num_features)
  Theta = np.random.randn(num_users, num_features)
  opt_X, opt_Theta = \
    compute_optimal_x_theta(X, Theta, Y, R, num_users, num_movies, num_features, 10.0)  
  p = np.dot(X, Theta.transpose())
  my_predictions = p + mean_Y[:, np.newaxis]
  print(Y)
  print(my_predictions)
  
  
def process_main():
  data = load_data('ex8_movies.mat')
  Y, R = data['Y'], data['R']
  check_data(Y, R)
  do_part2()
  do_part7()

if __name__ == '__main__':
  process_main()