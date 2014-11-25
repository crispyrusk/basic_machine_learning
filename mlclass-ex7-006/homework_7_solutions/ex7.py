# -*- coding: utf-8 -*-
"""
Created on Thu Oct 09 09:54:54 2014

@author: Kiran
"""

from __future__ import division

import scipy.io as scipyio
import numpy as np
import matplotlib.pylab as plt
import matplotlib.image as mpimg

import mathutil

def load_data():
  data = scipyio.loadmat('ex7data2.mat')
  X = data['X']
  return X

def find_closest_centroids(X, centroids):
  assert(mathutil.is_np_2d_array(centroids))
  r, c = centroids.shape
  centroids_reshaped = centroids.reshape((r,1,c))
  outter_product_distance_sq = np.sum((centroids_reshaped - X)**2, axis=-1)
  cluster_idx = np.argmin(outter_product_distance_sq, axis=0)
  return cluster_idx
  
def compute_centroids(X, cluster_idx, num_clusters):
  assert(mathutil.is_np_2d_array(X))
  __, c = X.shape  
  centroids = np.empty((num_clusters, c), dtype=np.float64)
  centroids[...] = np.inf
  for cluster_id in range(num_clusters):
    lidx_this_cluster = cluster_idx == cluster_id
    centroids[cluster_id, :] = np.mean(X[lidx_this_cluster, :], axis=0)
  return centroids    

def get_fig_ax():
  fig = plt.figure()
  ax = fig.add_subplot(111)
  return ax

def plot_k_means_progress(X, cluster_idx, hist_centroids):
  fig = plt.figure()
  num_itr, num_clusters, c = hist_centroids.shape
  ax = fig.add_subplot(111)
  for cluster_id in range(num_clusters):
    lidx_cluster = (cluster_idx==cluster_id)
    ax.plot(X[lidx_cluster,0], X[lidx_cluster,1], '*')
    ax.plot(hist_centroids[:,cluster_id,0], 
            hist_centroids[:,cluster_id,1], 'k-*')
  plt.show()
  
def run_k_means(X, initial_centroids, max_iter, plot_progress):
  assert(mathutil.is_np_2d_array(X))
  num_clusters, c = initial_centroids.shape
  hist_centroids = np.empty((max_iter+1, num_clusters, c), dtype=np.float64)
  hist_centroids[0, ...] = initial_centroids
  for itr in range(max_iter):
    cluster_idx = find_closest_centroids(X, hist_centroids[itr, ...])
    hist_centroids[itr+1, ...] = compute_centroids(X, cluster_idx, num_clusters)
  if plot_progress:
    plot_k_means_progress(X, cluster_idx, hist_centroids)
  final_centroids = hist_centroids[-1,...]
  return final_centroids, cluster_idx

def exercise_1():
  X = load_data()
  num_clusters = 3
  initial_centroids = np.array([[3, 3], [6, 2], [8, 5]], dtype=np.float64)
  cluster_idx = find_closest_centroids(X, initial_centroids)
  centroids = compute_centroids(X, cluster_idx, num_clusters)
  print(centroids)
  run_k_means(X, initial_centroids, 10, True)
  
def k_means_init_centroids(X, num_clusters):
  rand_idx = np.random.permutation(num_clusters)
  return X[rand_idx, :]

def exercise_2():
  A = mpimg.imread('bird_small.png')
  plt.imshow(A)
  A_scaled = A/255
  X = A_scaled.reshape((A.shape[0]*A.shape[1],3))
  num_clusters = 16
  max_iters = 10
  initial_centroids = k_means_init_centroids(X, num_clusters)
  centroids, __ = run_k_means(X, initial_centroids, max_iters, False)
  closest_idx = find_closest_centroids(X, centroids)
  X_recovered = centroids[closest_idx, :]
  A_recovered = (X_recovered.reshape((A.shape[0], A.shape[1], A.shape[2])))*255
  fig = plt.figure()
  ax0 = fig.add_subplot(1,2,1)
  ax1 = fig.add_subplot(1,2,2)
  ax0.imshow(A)
  ax1.imshow(A_recovered)
  
def process_main():
  exercise_1()
  exercise_2()
  

if __name__ == '__main__':
  process_main()