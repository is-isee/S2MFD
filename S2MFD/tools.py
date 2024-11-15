'''
Tools for FLD_ISEE calculations

'''
import sys
import numpy as np
from numba import njit

@njit
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
   
    dqq = np.zeros_like(qq)
    dqq[i0:i1,:] = (qq[1:,:] - qq[:-1,:])/drr
   
    return dqq

@njit
def drr2(qq,drr):
    '''
    To calculate 2nd order accuracy derivative in r direction
   
    Input:
        qq: quantity to be differentiated (numpy 2D array)
        drr: grid spacing in r direction (float)
    Output:
        dqq: differentiated quantity (numpy 2D array)
    '''   
    dqq = np.zeros_like(qq)
    dqq[1:-1,:] = (qq[2:qq.shape[0],:] - qq[0:-2,:])/drr*0.5
   
    return dqq
   
@njit   
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
   
    dqq = np.zeros_like(qq)
    dqq[:,j0:j1] = (qq[:,1:] - qq[:,:-1])/dth
   
    return dqq

@njit
def dth2(qq,dth):
    '''
    To calculate 2nd order accuracy derivative in theta direction
   
    Input:
        qq: quantity to be differentiated (numpy 2D array)
        drr: grid spacing in theta direction (float)
    Output:
        dqq: differentiated quantity (numpy 2D array)
    '''
   
    dqq = np.zeros_like(qq)
    dqq[:,1:-1] = (qq[:,2:qq.shape[1]] - qq[:,0:-2])/dth*0.5
   
    return dqq

#########