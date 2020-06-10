# Hadis Anahideh, Abolfazl Asudeh 2020
import numpy as np
import pandas as pd

Sz = None; Sx=None; Sy=None
m=0 # number of features (columns)
n=0 # number of samples (rows)
covXS = None # This shows the underlying covariance of features and sensitive attribute
covXY = None # The cov between the features of labeled set and true label y
theta = None # The theta of the latest model
covhS = None # The cov between the current model and S

def init(X,y,_covXS,_theta):
    global Sz,Sx,Sy,n,m,covXY,covXS,theta,covhS
    n,m = X.shape
    covXS = _covXS
    Sz = np.zeros(m); Sx = np.zeros(m)
    Sy = np.sum(y)
    for i in range(n):
        Sx = np.add(Sx,X[i])
        Sz = np.add(Sz, X[i]*y[i])
    Ey = 1./n*Sy
    covXY = 1./n*np.subtract(Sz,Sx*Ey)
    theta=_theta
    covhS = np.dot(covXS.transpose(),theta)[0,0]

def updateAggs(newX, newy,newtheta):
    global Sz,Sx,Sy,n,m,covXY,theta,covhS
    n += 1
    Sy += newy; Ey = 1./n*Sy
    Sx = np.add(Sx,newX)
    Sz = np.add(Sz, newX*newy)
    covXY = 1./n*np.subtract(Sz,Sx*Ey)
    theta = newtheta
    covhS = np.dot(covXS.transpose(),theta)[0,0]

# Expected Fairness Improvement <-- this should get maximized
def efi(tX, p):
    _efc = 0
    for i in range(len(p)): _efc += p[i]*fi(tX,i)
    return _efc

# Fairness Improvement
def fi(tX, ty):
    global Sz,Sx,Sy,n,m,covXY,covXS,theta,covhS
    nprime = n+1.
    Ey = (Sy + ty)/nprime
    _fc = 0
    sign = 1. if covhS>0 else -1.
    Ez = np.add(Sz , tX*ty)/nprime # vectorized
    Ex = np.add(Sx , tX)/nprime # vectorized
    cov = np.subtract(Ez, (Ex*Ey)) # cov of x and y if adding (tX, ty)
    _fc = np.dot((abs(theta*covXS)).transpose(),(abs(covXY)-abs(cov)).reshape(cov.shape[0],1))[0,0] # 

    return _fc