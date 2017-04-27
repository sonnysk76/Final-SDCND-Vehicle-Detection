[//]: # (image reference)
[hog1]: proj/hog_white.JPG 
[hog2]: proj/hog_black.JPG
[img1]: proj/white.JPG
[img2]: proj/black.JPG
[hist1]: proj/hist_white.JPG
[hist2]: proj/hist_black.JPG
[false]: proj/FalsePositives.JPG
[test1]: proj/test1_Pipeline.JPG
[test4]: proj/test4_Pipeline.JPG
[split]: proj/split_window.JPG
[lag]: proj/lagging_window.JPG
[delay]: proj/delay.JPG

# Vehicle Detection and Tracking
Vehicle Detection Project

The goals / steps of this project are the following:

- Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
- Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
- Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
- Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
- Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
- Estimate a bounding box for vehicles detected.

## Writeup / Read Me

### CRITERIA
#### Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. Here is a template writeup for this project you can use as a guide and a starting point.
_This document aims to meet this specification._

## Histogram of Oriented Gradients (HOG)
Original Image ![img1] Hog Image ![hog1] 

Color Histogram for white car image
![hist1]  

Original Image ![img2] Hog Image ![hog2] 

Color Histogram for black car image 
![hist2]
### CRITERIA
#### Explain how (and identify where in your code) you extracted HOG features from the training images. Explain how you settled on your final choice of HOG parameters.
In the VehicleDetection.py From Line 71 to 88 the function get_hog_features, uses the function skimage.feature.hog in order to extract the histogram of gradient features of each training image.

```
from skimage.feature import hog
def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                     vis=True, feature_vec=False):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  transform_sqrt=True,
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:
        features = hog(img, orientations=orient,
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),
                       transform_sqrt=True,
                       visualise=vis, feature_vector=feature_vec)
        return features
```
#### Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).
The starting point of the VehicleDetection.py program is in line 289, then the program has two options Train a model or Make a prediction base on a previous trained model.
In the first option if train variable is set to True it do the followin steps.
```
if __name__ == "__main__":

    # Modify Train to True when want to train a model
    train = False

    if train == True:
        # Read in cars and notcars for trainig
```
- Read the images set, cars and no cars and create an array for each.
- After there a space to set the training parameters.
Parameters used to train the model.

```        ### Training Parameters
        color_space = 'YUV'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
        orient = 12  # HOG orientations
        pix_per_cell = 8  # HOG pixels per cell
        cell_per_block = 2  # HOG cells per block
        hog_channel = "ALL"  # Can be 0, 1, 2, or "ALL"
        spatial_size = (32, 32)  # Spatial binning dimensions
        hist_bins = 32 # Number of histogram bins
        spatial_feat = True  # Spatial features on or off
        hist_feat = True  # Histogram features on or off
        hog_feat = True  # HOG features on or off
        y_start_stop = [400, 550]  # Min and max in y to search in slide_window()```
```
- Extract the features using the extract_features function.
- Consolidate the extracted features (X), and normalize using sklearn.preprocessing.StardardScaler().
- Generate the labels array (y)
- Random Split the data for training and test (80%/20%)
- Fit a LinearSVC model
- Save the result to a pickle file, for later use.


## Sliding Window Search

### CRITERIA
#### Describe how (and identify where in your code) you implemented a sliding window search. How did you decide what scales to search and how much to overlap windows?
The function get_boxes() uses the model saved pickle file, and with the diferent size of windows defined in scales[] array, get the hog, color histogram, and binned position vector, using find_cars() function.
The process is repeated for each different size scale.
I try with ranges from 0.5 to 2.5 with 0.1 increments.
Found useful scales of 1 and 1.5, I decide to use only this 2 scales due the high time required to extract the features for the different scales.

Line 201
```
def get_boxes(img):
    with open('modelHSV.pickle', 'rb') as g:
        modelp = pickle.load(g)

    scales = [1,1.5] #np.arange(0.6, 2.0, 0.7, np.float64)
    boxes = []
    for scale in scales:
        # print(scale)
        bbox = find_cars(np.copy(img),
                                  modelp['y_start_stop'][0],
                                  modelp['y_start_stop'][1],
                                  scale,
                                  modelp['svc'],
                                  modelp['X_scaler'],
                                  modelp['orient'],
                                  modelp['pix_per_cell'],
                                  modelp['cell_per_block'],
                                  modelp['spatial_size'],
                                  modelp['hist_bins'])
        boxes.extend(bbox)
    return boxes
 ``` 
 

#### Show some examples of test images to demonstrate how your pipeline is working. How did you optimize the performance of your classifier?
The process to optimize the classifier was in first place using the test images, and adjusting the parameters of hog threshold, and search windows size, more search windows give more data but also more false positives, to the focus was to get the less false positives possible by adjusting the search area in Y and X. Also different color space were tryed, HSV, YUV, RGB, YCrCb
The final parameters were tuned using video as input, selecting small clips with VideoFileClip().subclip()  selecting 5 seconds intervals and adjusting as needed, here another parameter was important the windows class get_windows() function determine the historic windows retrived, this can affect the detection, no detection and false positives. 

First attempts show high number of false positives.
![false]
#### Test images result Test1
![test1]
#### Test images result Test4
![test4]

## Video Implementation

### CRITERIA
#### Provide a link to your final video output. Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
https://youtu.be/2tEKpNtJ6lc

### The video don't show false positives.
#### Split window on second 12
![split]

#### Lagging window on second 23
![lag]

#### Delay 1 second to find white car again, second 37
![delay]

#### Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.
The separate file windows.py in the class windows what do is to combine the last n detected windows in order to carry information from one frame to the next, this make more stable the windows, avoid false positives, in the not to good side, depending the quantity of windows considered it can prevent to detect vehicles when various vehicles are in a frame, and also can appear lagged if there is few windows detected in every frame.
To filter false positives also apply_threshold() function was used in the heatmap part. 
```
def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap
```

## Discussion
### CRITERIA

#### Briefly discuss any problems / issues you faced in your implementation of this project. Where will your pipeline likely fail? What could you do to make it more robust?
In my case the may problem was to match color spaces, you have to take care that the color space parameters used for training are used for the vehicle traking part, is difficult to know I don't find a way to know the color space used on a stage of the program, have to map, if I read an image with cv2, it is BGR, and track forward all the process, and do the same for the video. This can lead to poor performance and in my case to unnecesary move parameters when the problem was a color space mismatch.
The process of apply to the video take half hour for the project video, I try to lower the time reducing the different windows sizes, but still take to much time to allow a better parameters optimization.
The window class return the last n windows, it works, but can be improved by considering number of previous frames instead of number of windows.
With this modification could avoid the delay and no detected car shown on the video
