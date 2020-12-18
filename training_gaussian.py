from matplotlib import pyplot as plt
from glob import glob
import cv2
import numpy as np
import os


X = np.load("X_data.npy")
Y = np.load("Y_data.npy")

number_of_1 = np.count_nonzero(Y==1)
indices = np.where(Y==1)
n = Y.shape[0]
theta_1 = number_of_1/n
X_1 = X[indices[0],:]
mu_1 = np.sum(X_1,axis=0)/number_of_1
data_1 = X_1 - mu_1
cov_1 = (data_1.T @ data_1)/number_of_1

number_of_0 = np.count_nonzero(Y==0)
indices = np.where(Y==0)
theta_0 = number_of_0/n
X_0 = X[indices[0],:]
mu_0 = np.sum(X_0,axis=0)/number_of_0
data_0 = X_0 - mu_0
cov_0 = (data_0.T @ data_0)/number_of_0

np.savetxt('cov0.txt',cov_0)
np.savetxt('cov1.txt',cov_1)
np.savetxt('mu0.txt',mu_0)
np.savetxt('mu1.txt',mu_1)
#np.savetxt('theta1.txt',theta_1)
#np.savetxt('theta0.txt',theta_0)

print(theta_1)
print(theta_0)
print(mu_1)
print(mu_0)
print(cov_1)
print(cov_0)