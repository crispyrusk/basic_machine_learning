# -*- coding: utf-8 -*-
"""
Created on Mon Oct 06 09:21:53 2014

@author: Kiran
"""

from __future__ import division

import numpy as np
import scipy.io as scipyio
import matplotlib.pylab as plt
import sklearn.svm as svm

def load_data(num):
  data = scipyio.loadmat('ex6data'+str(num)+'.mat')
  x = data['X']
  y = data['y']
  assert(np.ndim(x) == 2)
  assert(x.shape[1]==2)
  y = np.reshape(y, (-1))
  try:
    x_val = data['Xval']
    y_val = data['yval'].reshape((-1))
    return x, y, x_val, y_val
  except KeyError:
    return x,y

def plot_data(x, y):
  lidx_positive_samples = (y==1)
  lidx_negative_samples = (y==0)
  assert(np.all(np.logical_or(lidx_negative_samples, lidx_positive_samples)))
  assert(not np.any(np.logical_and(lidx_negative_samples, lidx_positive_samples)))
  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.plot(x[lidx_negative_samples,0], x[lidx_negative_samples, 1], 'ko', 
          markerfacecolor='y', markersize=7)
  ax.plot(x[lidx_positive_samples,0], x[lidx_positive_samples, 1], 'k+', 
          linewidth=1, markersize=7)          
  ax.grid()
  ax.set_xlabel('x_0')
  ax.set_ylabel('y_0')
  return fig

def visualize_decision_boundaray(clf, x, y):
  resolution = 1000
  x0, x1 = x[:,0], x[:,1]
  x0_mesh, x1_mesh = np.meshgrid(np.linspace(np.min(x0)-0.1, np.max(x0), resolution),
                                 np.linspace(np.min(x1)-0.1, np.max(x1), resolution))
  r, c = x0_mesh.shape                                 
  predicted_vals = np.empty_like(x0_mesh)
  for i in range(0, x0_mesh.shape[1]):
    this_x = np.empty((r,2))
    this_x[:,0] = x0_mesh[:,i]
    this_x[:,1] = x1_mesh[:,i]
    predicted_vals[:,i] = clf.predict(this_x)
  fig = plot_data(x,y)
  plt.figure(fig.number)
  ax=fig.get_axes()
  assert(len(ax)==1)
  ax[0].contour(x0_mesh, x1_mesh, predicted_vals, color='b', N=0)
  plt.show()

def train_linear_svm(x, y, C):
  clf = svm.LinearSVC(C=C, tol=1e-3)
  clf.fit(x, y)
  visualize_decision_boundaray(clf, x, y)
  return clf

def train_rbf_svm(x, y, C ,sigma, viz=True):
  gamma = 1/(2*sigma**2)
  clf = svm.SVC(C=C, tol=1e-3, gamma=gamma)
  clf.fit(x, y)
  assert(clf.kernel=='rbf')
  if viz:
    visualize_decision_boundaray(clf, x, y)
  return clf

def experiment_1st_data_set():
  x_1, y_1 = load_data(1)
  plot_data(x_1, y_1)
  train_linear_svm(x_1, y_1, C=1.0)
  train_linear_svm(x_1, y_1, C=100.0)

def experiment_2nd_data_set():
  x_2, y_2 = load_data(2)
  plot_data(x_2, y_2)
  train_rbf_svm(x_2, y_2, C = 1.0,sigma=0.1)

def experiment_3rd_data_set():
  x_3, y_3, x_val_3, y_val_3 = load_data(3)
  plot_data(x_3, y_3)
  best_C, best_sigma, best_clf_acc = -1.0, -1.0, -1.0
  search_set = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30]
  for this_C in search_set:
    for this_sigma in search_set:
      clf = train_rbf_svm(x_3, y_3, C=this_C, sigma=this_sigma, viz=False)
      clf_acc = np.count_nonzero(y_val_3 == clf.predict(x_val_3))/np.size(y_val_3)
      if (clf_acc > best_clf_acc):
        best_C, best_sigma, best_clf_acc = this_C, this_sigma, clf_acc
  print( 'best: C=' + str(best_C) + 
        ', sigma=' + str(best_sigma) + 
        ', clf_acc' + str(best_clf_acc))
  train_rbf_svm(x_3, y_3, C=best_C, sigma=best_sigma)  
  
def process_main():
  experiment_1st_data_set()
  experiment_2nd_data_set()
  experiment_3rd_data_set()  

  
if __name__ == '__main__':
  process_main()