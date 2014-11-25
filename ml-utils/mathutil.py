# -*- coding: utf-8 -*-
"""
Created on Mon Aug 18 16:05:27 2014

@author: Kiran
"""

from __future__ import division
import numpy as np

EXP_INP_MAX = 10.0
EXP_IN_MIN = -10.0

def get_aug_x(x):
  assert(np.ndim(x)==2)
  nrows, ncols = x.shape
  aug_x = np.concatenate((np.ones((nrows,1)), x), axis=1)
  return aug_x
  
def sigmoid_fn(x, min_x=EXP_IN_MIN, max_x=EXP_INP_MAX):
  with np.errstate(over='raise', under='raise'):
    capped_x = x
    capped_x[capped_x<min_x] = min_x
    capped_x[capped_x>max_x] = max_x
    result = 1.0/(1.0+np.exp(-capped_x))
  assert(np.all(result>0.0))
  assert(np.all(result<1.0))
  return result

def gradient_sigmoid_fn(x, min_x=EXP_IN_MIN, max_x=EXP_INP_MAX):
  return sigmoid_fn(x)*(1.0-sigmoid_fn(x))    

def is_np_col_vector(x):
  if (np.ndim(x)==2):
    n_rows, n_cols = x.shape
    return (n_cols == 1)
  return False
  
def is_np_row_vector(x):
  if (np.ndim(x)==2):
    n_rows, n_cols = x.shape
    return (n_rows == 1)
  return False

def is_np_2d_array(x):
  return np.ndim(x)==2

def is_np_1d_array(x):
  return np.ndim(x)==1

if __name__ == '__main__':
  print('local tests')
  x=np.arange(-4, 4, 0.5)
  print(sigmoid_fn(x))
  test = 0.0 
  print(is_np_1d_array(np.array(test)))  
  print(is_np_1d_array(np.array([test])))  
  print(is_np_1d_array(np.array([[test]])))
  print(gradient_sigmoid_fn(np.array([-100.0, -100.0, 0.0])))