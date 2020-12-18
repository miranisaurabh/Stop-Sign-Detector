from roipoly import RoiPoly, MultiRoi
from matplotlib import pyplot as plt
from glob import glob
import cv2
import numpy as np
import os

path = r'C:\Users\mirani\Documents\ECE 276A\Project 1\ECE276A_PR1 (1)\ECE276A_PR1\hw1_starter_code\trainset_mini2/*'
f = glob(path)
path = r'C:/Users/mirani/Documents/ECE 276A/Project 1/ECE276A_PR1 (1)/ECE276A_PR1/hw1_starter_code/trainlables/'
out_f = path

for fname in f:
	image = cv2.imread(fname)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	plt.imshow(image)
	roi_result = MultiRoi(roi_names=roi_list)
	
	print(len(roi_result.rois))
	mask = np.zeros([image.shape[0],image.shape[1]])
	for roi_name in roi_list:
		try:
			mask += roi_result.rois[roi_name].get_mask(gray)
		except:
			pass

	ind = (mask > 0)
	mask[ind] = 1

	print(np.unique(mask))
	mask_img = cv2.resize(mask, (640, 480))
	cv2.imshow('mask', mask_img*255) # to check the binary image mask
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	head, tail = os.path.split(fname)
	np.save(out_f + tail[:-3] + "npy", mask)