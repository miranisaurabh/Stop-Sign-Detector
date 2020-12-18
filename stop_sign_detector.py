import os, cv2
from skimage.measure import label, regionprops
import numpy as np

class StopSignDetector():
	def __init__(self):
		'''
			Initilize your stop sign detector with the attributes you need,
			e.g., parameters of your classifier
		'''

		# Initializing parameters obtained from training GDM

		self.mu0 = np.array([1.346783574588043564e+02,1.278213595792159509e+02,1.177959088511585435e+02])
		#self.mu1 = np.array([53.85255293,46.50559933,160.38144411])
		self.mu1 = self.mu0

		self.cov0 = np.array([[5152.30905618, 3929.18525226, 3032.51053292],
		[3929.18525226, 3596.06457886, 3231.52661721],
		[3032.51053292, 3231.52661721, 3489.2491608 ]])
		self.cov1 = np.array([[3635.21794126, 3850.27150293, 1267.46298175],
		[3850.27150293, 4325.5668949,  1268.41744937],
		[1267.46298175, 1268.41744937, 3123.23525558]])
		
		self.theta0 = 0.9879016202055569
		self.theta1 = 1 - self.theta0

	def multivariate_gaussian(self,X):
		
		#Calculate probability value using GDM and MLE parameters

		#Class '0' - not red
		x_vect = X - self.mu0
		dim = self.cov0.shape[0]
		sig_inv = np.linalg.inv(self.cov0)
		data1 = np.matmul(x_vect, sig_inv)
		data2 = np.multiply(data1,x_vect)
		exp_pow = -0.5*np.sum(data2,axis=1)
		det = np.linalg.det(self.cov0)
		denom = np.sqrt(((2*np.pi)**dim) * det)
		p_0 = np.exp(exp_pow)/denom

		#Class '1' - red
		x_vect = X - self.mu1
		dim = self.cov1.shape[0]
		sig_inv = np.linalg.inv(self.cov1)
		data1 = np.matmul(x_vect, sig_inv)
		data2 = np.multiply(data1,x_vect)
		exp_pow = -0.5*np.sum(data2,axis=1)
		det = np.linalg.det(self.cov1)
		denom = np.sqrt(((2*np.pi)**dim) * det)
		p_1 = np.exp(exp_pow)/denom

		return p_0,p_1

	def segment_image(self, img):
		'''
			Obtain a segmented image using a color classifier,
			e.g., Logistic Regression, Single Gaussian Generative Model, Gaussian Mixture, 
			call other functions in this class if needed
			
			Inputs:
				img - original image
			Outputs:
				mask_img - a binary image with 1 if the pixel in the original image is red and 0 otherwise
		'''

		#Use Gamma correction
		gamma = 2.0
		invGamma = 1.0 / gamma
		table = np.array([((i / 255.0) ** invGamma) * 255
			for i in np.arange(0, 256)]).astype("uint8")
		img = cv2.LUT(img,table)
		
		#Reshaping 3D array to 2D array
		X = img.reshape((img.shape[0]*img.shape[1]), img.shape[2])

		#Get the probability values from Gaussian distribution
		p_0,p_1 = self.multivariate_gaussian(X)
		
		#Use Bayes' decision rule
		result = p_1*self.theta1 > p_0*self.theta0

		#Convert result to compatible shape and datatype
		mask_img = result.astype(np.uint8)
		mask_img = mask_img.reshape(img.shape[0],img.shape[1])
		return mask_img

	def get_bounding_box(self, img):
		'''
			Find the bounding box of the stop sign
			call other functions in this class if needed
			
			Inputs:
				img - original image
			Outputs:
				boxes - a list of lists of bounding boxes. Each nested list is a bounding box in the form of [x1, y1, x2, y2] 
				where (x1, y1) and (x2, y2) are the top left and bottom right coordinate respectively. The order of bounding boxes in the list
				is from left to right in the image.
				
			Our solution uses xy-coordinate instead of rc-coordinate. More information: http://scikit-image.org/docs/dev/user_guide/numpy_images.html#coordinate-conventions
		'''

		#Get the size of image
		x_max = np.shape(img)[1]
		y_max = np.shape(img)[0]
		
		#Create an empty list
		boxes = []
		similarity = []
		 
		#Get the binary mask
		img_r = self.segment_image(img)
		
		#Convert the mask to 0 and 255 type for cv2 functions
		img_r = img_r*255



		#If the image is of high resolution, apply Gaussian smoothing and threshold else leave
		if x_max*y_max > 200000:
			blurred = cv2.GaussianBlur(img_r, (5, 5),0)
			ret,thresh = cv2.threshold(blurred ,127,255,0)
		else:
			thresh = img_r

		#Get the contours from the binary mask
		contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

		#For each obtained contour, check if it satisfies our conditions
		for cnt in contours:

			#Fit an approximate polygon
			approx = cv2.approxPolyDP(cnt,0.04*cv2.arcLength(cnt,True),True)

			#If the edges meet our condition then proceed
			if (len(approx) >=4) & (len(approx) <=12 ):
				
				#similarty value calculation; ideal edges=8
				similarity1 = 100 - (np.absolute(8-len(approx))/8.0*100)

				#Get the features of bounding rectangle
				x,y,w,h = cv2.boundingRect(cnt)
				area_ratio = cv2.contourArea(cnt)/(x_max*y_max)
				
				#If it is a high resolution image then proceed
				if (x_max*y_max > 200000):

					#If this condition is met, then it is most likely a STOP sign	
					if (h/w >=0.9) & (h/w <=1.3) & (area_ratio > 0.002):											
						boxes.append([x,y_max-(y+h),x+w,y_max-y])

						#similarty value calculation; ideal h/w=1
						similarity2 = 100 - np.absolute((h/w)-1)*100
						#Take average of both the similarity values
						similarity_val = (similarity1+similarity2)/2
						similarity.append([similarity_val])
				
				#The conditions are relaxed a bit for low resolution images
				elif (h/w >=0.6) & (h/w <=1.7) & (area_ratio>=0.0007):
					boxes.append([x,y_max-(y+h),x+w,y_max-y])

					#similarty value calculation; ideal h/w=1
					similarity2 = 100 - np.absolute((h/w)-1)*100
					#Take average of both the similarity values
					similarity_val = (similarity1+similarity2)/2
					similarity.append([similarity_val])
		
		#Sorting the list w.r.t x indices for autograder compatibility
		print(boxes)
		boxes.sort()
		return boxes


if __name__ == '__main__':
	folder = "trainset"
	my_detector = StopSignDetector()
	for filename in os.listdir(folder):
		# read one test image
		img = cv2.imread(os.path.join(folder,filename))
		# cv2.imshow('image', img)
		# cv2.waitKey(0)
		# cv2.destroyAllWindows()

		#Display results:
		#(1) Segmented images
		mask_img = my_detector.segment_image(img)
		#(2) Stop sign bounding box
		boxes = my_detector.get_bounding_box(img)

