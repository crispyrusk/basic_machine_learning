# -*- coding: utf-8 -*-
"""
Created on Thu Oct 09 14:02:33 2014

@author: Kiran
"""

from __future__ import division

import numpy as np
import math
import matplotlib.pylab as plt
import scipy.io as scipyio
import scipy.linalg as linalg

def load_data():
  data = scipyio.loadmat('ex7data1.mat')
  return data['X']

def feature_normalize(X):
  mu = np.mean(X, axis=0)
  sigma = np.std(X, axis=0)
  X_norm = (X - mu)/sigma
  return X_norm, mu, sigma

def pca(X):
  num_samples, num_features = X.shape
  sigma = (1/num_samples)*np.dot(X.transpose(), X)
  U, S, __ = linalg.svd(sigma)
  return U, S

def plot_eig_basis(X, mu, U, S):
  fig = plt.figure()
  ax = fig.add_subplot(111)
  r, c = U.shape
  assert(r==2 and c==2)
  ax.plot(X[:,0], X[:,1], 'o')
  eig_vector_0 = np.empty((r,2))
  eig_vector_0[:,0] = mu
  eig_vector_0[:,1] = mu + 1.5*S[0]*U[:,0]
  eig_vector_1 = np.empty((r,2))
  eig_vector_1[:,0] = mu
  eig_vector_1[:,1] = mu + 1.5*S[1]*U[:,1]
  ax.plot(eig_vector_0[0,:], eig_vector_0[1,:])
  ax.plot(eig_vector_1[0,:], eig_vector_1[1,:])
  ax.set_title('eig space')
  ax.grid()
  plt.show()

def plot_normalized_data(X_norm):
  plt.figure()
  plt.plot(X_norm[:,0], X_norm[:,1], 'bo')
  plt.grid()
  plt.title('normalized data')
  plt.show()

def project_data(X, U, K):
  __, c = U.shape
  assert(K <= c)
  U_reduce = U[:,0:K]
  Z = np.dot(X, U_reduce)
  return Z

def recover_data(Z, U, K):
  X_rec_transpose = np.dot(U[:,0:K], Z.transpose())
  return X_rec_transpose.transpose()

def exercise_1():
  X = load_data()
  X_norm, mu, sigma = feature_normalize(X)
  U, S = pca(X_norm)
  print('U = '+str(U))
  print('S = '+str(S))
  plot_eig_basis(X, mu, U, S)
  plot_normalized_data(X_norm)
  K = 1
  Z = project_data(X_norm, U, K)
  print(Z[0])
  X_rec = recover_data(Z, U, K)
  print(X_rec[0,:])

def load_face_data():
  data = scipyio.loadmat('ex7faces.mat')
  return data['X']  
  
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
  
def exercise_2():
  X = load_face_data()
  display_data(X[0:100,:])
  X_norm, mu, sigma = feature_normalize(X)
  U, S = pca(X_norm)
  display_data(U[:,0:36].transpose())
  K = 100
  Z = project_data(X_norm, U, K)
  print(Z.shape)
  X_rec = recover_data(Z, U, K)
  display_data(X_norm[0:100,:])
  display_data(X_rec[0:100,:])
  

def process_main():
  exercise_1()
  exercise_2()  
  
if __name__ == '__main__':
  process_main()