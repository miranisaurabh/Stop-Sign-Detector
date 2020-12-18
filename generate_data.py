from matplotlib import pyplot as plt
from glob import glob
import cv2
import numpy as np
import os

path1 = r'C:/Users/mirani/Documents/ECE 276A/Project 1/ECE276A_PR1 (1)/ECE276A_PR1/hw1_starter_code/merge_data2/'
path2 = r'C:/Users/mirani/Documents/ECE 276A/Project 1/ECE276A_PR1 (1)/ECE276A_PR1/hw1_starter_code/labelled_imgs/'
#f = glob(path)
X = np.zeros([1,3])
Y_temp = np.zeros([1,1])

for file_name in os.listdir(path1):

    img = cv2.imread(os.path.join(path1,file_name))
    new_img = img.reshape((img.shape[0]*img.shape[1]), img.shape[2])
    X = np.append(X, new_img, axis=0)
    print(X.shape)
    
    data = np.load(os.path.join(path2,file_name[:-3]+"npy"))
    print(file_name)
    if len(data.shape) > 2 :
        data = data[:,:,0]
    new_data = data.reshape(data.shape[0]*data.shape[1],1)
    Y_temp = np.append(Y_temp, new_data, axis=0)
    print(Y_temp.shape)
    Y = Y_temp

print("Done!")
np.save('X_data.npy',X)
np.save('Y_data.npy',Y)
print("Saved!")
