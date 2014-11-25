# -*- coding: utf-8 -*-
"""

"""

from __future__ import division

import numpy as np
import matplotlib.pylab as plt
import matplotlib.animation as animation


def return_eye_matrix(ndim):
  return np.eye(ndim)

def read_data():
  data = np.loadtxt('ex1data1.txt', delimiter=',')
  return data

def plot_data(x, y):
  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.plot(x,y, '.')
  ax.grid()
  plt.show()

def compute_cost(theta_0, theta_1, x, y):
  m = np.size(x)
  return (1/(2*m))*np.sum((theta_0+theta_1*x-y)**2)

# multiparameter sweep based computation of function at several points
def compute_costs_thetas(theta_0, theta_1, x, y):
  m = np.size(x)
  theta_0_extended = theta_0[..., np.newaxis]
  theta_1_extended = theta_1[..., np.newaxis]
  error = (theta_0_extended + theta_1_extended*x-y)
  squarred_error = error**2
  sum_squarred_error = np.sum(squarred_error, axis=-1)
  return (1/(2*m))*sum_squarred_error

def compute_newtheta_gradient_of_cost(theta, alpha, m, x, y):
  theta_0_temp = (theta[0] - (alpha/m)*np.sum(theta[0]+theta[1]*x-y))
  theta_1_temp = (theta[1] - (alpha/m)*np.sum((theta[0]+theta[1]*x-y)*x))
  return theta_0_temp, theta_1_temp

def update_line(num, theta_hist, x, y, line):
  x_min = np.min(x)
  x_max = np.max(x)
  error_j = compute_cost(theta_hist[num][0], theta_hist[num][1], x, y)
  print('iter: ' + str(num) + ", error:" + str(error_j))
  x = np.array([x_min, x_max])
  y = theta_hist[num][0] + theta_hist[num][1]*x
  line.set_data(x, y)
  return line

def run_gradient_descent(data, theta, alpha, num_iters, show_animation = False):
  x = data[:, 0]
  y = data[:, 1]
  assert(x.ndim==1)
  assert(y.ndim==1)
  m = np.size(x)
  assert(np.size(y)==m)
  theta_hist = np.zeros((num_iters+1, 2))

  theta_hist[0][0] = theta[0]
  theta_hist[0][1] = theta[1]
  for iter_ctr in range(0, num_iters):
    theta_hist[iter_ctr+1, 0], theta_hist[iter_ctr+1, 1] = \
      compute_newtheta_gradient_of_cost(theta_hist[iter_ctr], alpha, m, x, y)
  
  if show_animation:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x,y,'.')
    ax.grid()
    fitted_line, = ax.plot([], [], 'r') 
    fitted_line_animation = \
      animation.FuncAnimation(fig, update_line, frames = num_iters, 
                              fargs=(theta_hist, x, y, fitted_line))
    plt.show()                                                    
    
  assert(num_iters>0)
  return theta_hist[-2,:]

def plot_cost_function(data):
  x = data[:, 0]
  y = data[:, 1]
  theta_0_mesh, theta_1_mesh = np.meshgrid(np.arange(-10, 10, 0.01), np.arange(-1, 4, 0.01))
  cost_all_points = np.zeros(theta_0_mesh.shape)
  cost_all_points = compute_costs_thetas(theta_0_mesh, theta_1_mesh, x, y)
  plt.contour(theta_0_mesh, theta_1_mesh, cost_all_points, 100)    
  plt.show()
  
def process_main():
  data = read_data()
  theta = run_gradient_descent(data, np.array([0.0, 0.0]), 0.01, 1500, show_animation = False)
  test1 = np.dot(theta, [1, 3.5])
  assert(np.allclose(test1, 0.45228764580657632))
  test2 = np.dot(theta, [1,7])
  assert(np.allclose(test2, 4.5343872966379282))
  plot_cost_function(data)
  

if __name__=='__main__':
  process_main()
  