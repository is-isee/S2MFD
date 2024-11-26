'''
Tools for S2MFD calculations

'''
import sys
import numpy as np
from numba import njit, float64
from numba.types import Array, Tuple, unicode_type

#TODO: completed
@njit(Array(float64, 2, 'C', False, aligned=True)
      (
      Array(float64, 2, 'C', False, aligned=True), 
      Array(float64, 0, 'C', False, aligned=True), 
      unicode_type  
      ), 
      cache=True)
def drr1(qq,drr,dir):
    '''
    To calculate 1st order accuracy derivative in r direction
   
    Input:
        qq: quantity to be differentiated (numpy 2D array)
        drr: grid spacing in r direction (float)
        dir: string, 'up' or 'dw'
            'up': qq[i] = qq[i+1] - qq[i]
            'dw': qq[i] = qq[i] - qq[i-1]
    Output:
        dqq: differentiated quantity (numpy 2D array)
    '''
   
    if dir == 'up':
        i0, i1 = 1, qq.shape[0]
    elif dir == 'dw':
        i0, i1 = 0, qq.shape[0]-1
    else:
        print('Error: dir must be up or dw')

    # vectorization
    # dqq = np.zeros_like(qq)
    # dqq[i0:i1,:] = (qq[1:,:] - qq[:-1,:])/drr

    # for loop
    inum = qq.shape[0]
    jnum = qq.shape[1]
    dqq = np.zeros_like(qq)
    for i in range(i1-i0):
        for j in range(jnum):
            dqq[i+i0,j] = (qq[i+1,j] - qq[i,j])/drr
   
    return dqq

#TODO: completed
@njit(Array(float64, 2, 'C', False, aligned=True)
      (
      Array(float64, 2, 'C', False, aligned=True), 
      Array(float64, 0, 'C', False, aligned=True), 
      ), 
      cache=True)
def drr2(qq,drr):
    '''
    To calculate 2nd order accuracy derivative in r direction
   
    Input:
        qq: quantity to be differentiated (numpy 2D array)
        drr: grid spacing in r direction (float)
    Output:
        dqq: differentiated quantity (numpy 2D array)
    '''   

    # vectorization
    # dqq = np.zeros_like(qq)
    # dqq[1:-1,:] = (qq[2:qq.shape[0],:] - qq[0:-2,:])/drr*0.5

    # for loop
    inum = qq.shape[0]
    jnum = qq.shape[1]
    dqq = np.zeros_like(qq)
    for i in range(inum-2):
        for j in range(jnum):
            dqq[i+1,j] = (qq[i+2,j] - qq[i,j])/drr*0.5
   
    return dqq

#TODO: completed
@njit(Array(float64, 2, 'C', False, aligned=True)
      (
      Array(float64, 2, 'C', False, aligned=True), 
      Array(float64, 0, 'C', False, aligned=True), 
      unicode_type  
      ), 
      cache=True) 
def dth1(qq,dth,dir):
    '''
    To calculate 1st order accuracy derivative in theta direction
   
    Input:
        qq: quantity to be differentiated (numpy 2D array)
        drr: grid spacing in theta direction (float)
        dir: string, 'up' or 'dw'
            'up': qq[j] = qq[j+1] - qq[j]
            'dw': qq[j] = qq[j] - qq[j-1]
    Output:
        dqq: differentiated quantity (numpy 2D array)
    '''
   
    if dir == 'up':
        j0, j1= 1,qq.shape[1]
    elif dir == 'dw':
        j0, j1 = 0, qq.shape[1]-1
    else:
        print('Error: dir must be up or dw')
   
    # vectorization
    # dqq = np.zeros_like(qq)
    # dqq[:,j0:j1] = (qq[:,1:] - qq[:,:-1])/dth

    # for loop
    inum = qq.shape[0]
    dqq = np.zeros_like(qq)
    for i in range(inum):
        for j in range(j1-j0):
            dqq[i,j+j0] = (qq[i,j+1] - qq[i,j])/dth
        
    return dqq

#TODO: completed
@njit(Array(float64, 2, 'C', False, aligned=True)
      (
      Array(float64, 2, 'C', False, aligned=True),  
      Array(float64, 0, 'C', False, aligned=True)    
      ), 
      cache=True)
def dth2(qq,dth):
    '''
    To calculate 2nd order accuracy derivative in theta direction
   
    Input:
        qq: quantity to be differentiated (numpy 2D array)
        drr: grid spacing in theta direction (float)
    Output:
        dqq: differentiated quantity (numpy 2D array)
    '''
    
    # vectorization
    # dqq = np.zeros_like(qq)
    # dqq[:,1:-1] = (qq[:,2:qq.shape[1]] - qq[:,0:-2])/dth*0.5

    # for loop
    inum = qq.shape[0]
    jnum = qq.shape[1]
    dqq = np.zeros_like(qq)
    for i in range(inum):
        for j in range(jnum-2):
            dqq[i,j+1] = (qq[i,j+2] - qq[i,j])/dth*0.5

    return dqq

#########