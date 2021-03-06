## About
* A classifier that will segement the image into two parts - red and not red using Gaussian Discriminative Model
* A detector which detects if the red part has any STOP signs based on contour, and if a STOP sign is detected, it creates a bounding box around the STOP sign and return its co-ordinates in pixels


## File Details

### data_labelling.py: 
This script shows the image and asks the user to plot a region of interest. Multiple roi's can be given. It used the roipoly library. It generates an output numpy array of binary values 1 and 0, where 1 represents roi and 0 otherwise

### generate_data.py:
This script loads all the training images and their respective labels y=0 or 1 as numpy arrays and vectorizes them for faster computations, i.e. converts BGR (mxnx3) image to (m*nx3) 2D array where each row is a pixel and columns represent B,G and R channels respectively. It saves 2 npy files, X containing data of training images and Y containing data of image labels

### training_gaussian.py:
This script takes the numpy arrays X and Y generated by generate_data.py file as input. It uses MLE method to compute all the parameters required for the model, theta,mu and sigma for both the classes and stores them in a text file.

### stop_sign_detector:
This file has a class StopSignDetector(), with important memeber functions segement_iamge() and get_bounding_box()  

## Technical Report
* [Saurabh Himmatlal Mirani. "Color Segementation" Jan. 2020](report/ColorSegementation.pdf)

## Results

### Case 1:
<p float="left">
  <img src="images/4_mask.png" width="49%" />
  <img src="images/4_rect.png" width="49%" /> 
</p>

### Case 2:
<p float="left">
  <img src="images/8_mask.png" width="49%" />
  <img src="images/8_rect.png" width="49%" /> 
</p>

### Case 3:
<p float="left">
  <img src="images/18_mask.png" width="49%" />
  <img src="images/18_rect.png" width="49%" /> 
</p>

### Case 4:
<p float="left">
  <img src="images/19_mask.png" width="49%" />
  <img src="images/19_rect.png" width="49%" /> 
</p>

### Case 5:
<p float="left">
  <img src="images/31_mask.png" width="49%" />
  <img src="images/31_rect.png" width="49%" /> 
</p>

### Case 6:
<p float="left">
  <img src="images/41_mask.png" width="49%" />
  <img src="images/41_rect.png" width="49%" /> 
</p>

### Case 7:
<p float="left">
  <img src="images/49_mask.png" width="49%" />
  <img src="images/49_rect.png" width="49%" /> 
</p>

### Case 8:
<p float="left">
  <img src="images/50_mask.png" width="49%" />
  <img src="images/50_rect.png" width="49%" /> 
</p>

### Case 9:
<p float="left">
  <img src="images/4_mask.png" width="49%" />
  <img src="images/4_rect.png" width="49%" /> 
</p>

### Case 10:
<p float="left">
  <img src="images/62_mask.png" width="49%" />
  <img src="images/62_rect.png" width="49%" /> 
</p>

### Case 11:
<p float="left">
  <img src="images/72_mask.png" width="49%" />
  <img src="images/72_rect.png" width="49%" /> 
</p>

### Case 12:
<p float="left">
  <img src="images/99_mask.png" width="49%" />
  <img src="images/99_rect.png" width="49%" /> 
</p>

### Case 13:
<p float="left">
  <img src="images/100_mask.png" width="49%" />
  <img src="images/100_rect.png" width="49%" /> 
</p>
