import cv2
import glob
import numpy as np
from sklearn.svm import LinearSVC
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import time
from scipy.ndimage.measurements import label
from moviepy.editor import VideoFileClip
import pickle
from pprint import pprint
from windows import windows

######### MODEL TRAINING #################
# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, color_space='RGB', spatial_size=(32, 32),
                     hist_bins=32, orient=9,
                     pix_per_cell=8, cell_per_block=2, hog_channel=0,
                     spatial_feat=True, hist_feat=True, hog_feat=True):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        file_features = []
        # Read in each one by one
        image = cv2.imread(file)
        # apply color conversion if other than 'RGB'
        if color_space != 'RGB':
            if color_space == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif color_space == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif color_space == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif color_space == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif color_space == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else:
            feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if spatial_feat == True:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            file_features.append(spatial_features)
        if hist_feat == True:
            # Apply color_hist()
            hist_features = color_hist(feature_image, nbins=hist_bins)
            file_features.append(hist_features)
        if hog_feat == True:
            # Call get_hog_features() with vis=False, feature_vec=True
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image[:, :, channel],
                                                         orient, pix_per_cell, cell_per_block,
                                                         vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)
            else:
                hog_features = get_hog_features(feature_image[:, :, hog_channel], orient,
                                                pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            # Append the new feature vector to the features list
            file_features.append(hog_features)
        features.append(np.concatenate(file_features))
    # Return list of feature vectors
    return features

# Define a function to return HOG features and visualization
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


# Define a function to compute binned color features
def bin_spatial(img, size=(32,32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel()
    # Return the feature vector
    return features


# Define a function to compute color histogram features
# NEED TO CHANGE bins_range if reading .png files with mpimg!
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:, :, 0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:, :, 1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:, :, 2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features



###############################################################
########

def convert_color(img, conv='RGB2YCrCb'):
    if conv == 'RGB2HSV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    if conv == 'RGB2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == 'BGR2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if conv == 'RGB2LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)

# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, color_space):
    bbox = []
    #draw_img = np.copy(img)

    img_tosearch = img[ystart:ystop, 640:, :]
    ctrans_tosearch = convert_color(img_tosearch, conv='RGB2HSV')
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

    ch1 = ctrans_tosearch[:, :, 0]
    ch2 = ctrans_tosearch[:, :, 1]
    ch3 = ctrans_tosearch[:, :, 2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - 1
    #nfeat_per_block = orient * cell_per_block ** 2
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    # Compute individual channel HOG features for the entire image
    hog1, img_hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2, img_hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3, img_hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb * cells_per_step
            xpos = xb * cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos * pix_per_cell
            ytop = ypos * pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], (64, 64))

            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            test_features = X_scaler.transform(
                np.hstack((hog_features)).reshape(1, -1))
                #np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
            # test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))
            test_prediction = svc.predict(test_features)

            if test_prediction == 1:
                xbox_left = np.int(xleft * scale)+640
                ytop_draw = np.int(ytop * scale)
                win_draw = np.int(window * scale)
                bbox.append(((xbox_left, ytop_draw + ystart),(xbox_left + win_draw, ytop_draw + win_draw + ystart)))
                #cv2.rectangle(draw_img, (xbox_left, ytop_draw + ystart),
                #              (xbox_left + win_draw, ytop_draw + win_draw + ystart), (0, 0, 255), 6)

    return bbox

def get_boxes(img):
    with open('modelYUV.pickle', 'rb') as g:
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
                                  modelp['hist_bins'],
                                  modelp['color_space'])
        boxes.extend(bbox)
    return boxes

###### HEAT MAP FUNCTIONS #####
def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap  # Iterate through list of bboxes


def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap


def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1] + 1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        x = np.max(nonzerox) - np.min(nonzerox)
        y = np.max(nonzeroy) - np.min(nonzeroy)
        if ((x > 50) | (y > 60 )):
            # Draw the box on the image
            text = "Car "#+str(car_number)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, text, (bbox[0][0],bbox[0][1]-10), font, 1, (255, 0, 55), 2)
            cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)
    # Return the image
    return img


def heat_step(image, boxlist):
    heat = np.zeros_like(image[:, :, 0]).astype(np.float)
    heat = add_heat(heat, boxlist)
    heat = apply_threshold(heat, 2)
    heatmap = np.clip(heat, 0, 255)
    labels = label(heatmap)
    result = draw_labeled_bboxes(np.copy(image), labels)
    #heatmap = np.dstack((heatmap,heatmap,heatmap))
    return result



def pipeline(img):
    box_list = get_boxes(img)
    win.set_windows(box_list)
    out_img = heat_step(img, win.get_windows())
    print(len(win.get_windows()))
    return out_img


if __name__ == "__main__":

    train = False

    if train == True:
        # Read in cars and notcars for trainig
        # list of images 5966 cars and 5966 notcars, png images
        images = glob.glob('vehicles/*.png')
        cars = []
        notcars = []
        for image in images:
            if 'image' in image or 'extra' in image:
                notcars.append(image)
            else:
                cars.append(image)

        ### Training Parameters
        color_space = 'YUV'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
        orient = 12  # HOG orientations
        pix_per_cell = 8  # HOG pixels per cell
        cell_per_block = 2  # HOG cells per block
        hog_channel = "ALL"  # Can be 0, 1, 2, or "ALL"
        spatial_size = (32, 32)  # Spatial binning dimensions
        hist_bins = 32 # Number of histogram bins
        spatial_feat = False  # Spatial features on or off
        hist_feat = False  # Histogram features on or off
        hog_feat = True  # HOG features on or off
        y_start_stop = [400, 550]  # Min and max in y to search in slide_window()

        car_features = extract_features(cars, color_space=color_space,
                                        spatial_size=spatial_size, hist_bins=hist_bins,
                                        orient=orient, pix_per_cell=pix_per_cell,
                                        cell_per_block=cell_per_block,
                                        hog_channel=hog_channel, spatial_feat=spatial_feat,
                                        hist_feat=hist_feat, hog_feat=hog_feat)
        notcar_features = extract_features(notcars, color_space=color_space,
                                           spatial_size=spatial_size, hist_bins=hist_bins,
                                           orient=orient, pix_per_cell=pix_per_cell,
                                           cell_per_block=cell_per_block,
                                           hog_channel=hog_channel, spatial_feat=spatial_feat,
                                           hist_feat=hist_feat, hog_feat=hog_feat)

        X = np.vstack((car_features, notcar_features)).astype(np.float64)
        # Fit a per-column scaler
        X_scaler = StandardScaler().fit(X)
        # Apply the scaler to X
        scaled_X = X_scaler.transform(X)

        # Define the labels vector
        y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

        # Split up data into randomized training and test sets
        rand_state = np.random.randint(0, 100)
        X_train, X_test, y_train, y_test = train_test_split(
            scaled_X, y, test_size=0.2, random_state=rand_state)

        print('Using:', orient, 'orientations', pix_per_cell,
              'pixels per cell and', cell_per_block, 'cells per block')
        print('Feature vector length:', len(X_train[0]))
        # Use a linear SVC
        svc = LinearSVC()
        # Check the training time for the SVC
        t = time.time()
        svc.fit(X_train, y_train)
        ## Save trained model to pickle
        #pickle.dump(svc, open('svc_pickle.p', 'wb'))
        t2 = time.time()
        print(round(t2 - t, 2), 'Seconds to train SVC...')
        # Check the score of the SVC
        print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
        # Check the prediction time for a single sample
        t = time.time()
        model = {}
        model['svc'] = svc
        model['y_start_stop'] = y_start_stop
        model['X_scaler'] = X_scaler
        model['orient'] = orient
        model['pix_per_cell'] = pix_per_cell
        model['cell_per_block'] = cell_per_block
        model['spatial_size'] = spatial_size
        model['hist_bins'] = hist_bins
        model['color_space'] = color_space

        with open('modelYUV.pickle', 'wb') as f:
            pickle.dump(model, f)


    else:
        win = windows()
        # Apply process image to video.
        #white_output = 'images/project_video_0_4_H1.mp4'
        #clip1 = VideoFileClip("images/project_video.mp4").subclip(27,37)
        #white_clip = clip1.fl_image(pipeline)  # NOTE: this function expects color images!!
        #white_clip.write_videofile(white_output, audio=False)

        img = mpimg.imread('images/test6.jpg')
        imagen_prueba = pipeline(img)
        #imagen_prueba = cv2.cvtColor(imagen_prueba, cv2.COLOR_BGR2RGB)
        plt.imshow(imagen_prueba)
        plt.show()



        '''  
        with open('model4.pickle', 'rb') as g:
            modelp = pickle.load(g)

        #print(modelp)

        img = cv2.imread('images/test6.jpg')  # modified from mpimg


        #ystart = 450
        #ystop = 720
        scales = np.arange(0.6,2.3,0.1,np.float64)
        boxes =[]
        for scale in scales:
            #print(scale)
            out_img, bbox = find_cars(np.copy(img),
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
            #print(len(bbox))
            #print(bbox)
            boxes.extend(bbox)
            #print('2', boxes)
        #plt.imshow(out_img)
        #plt.show()

    #############################################################
    ####### HEAT MAP
        # Read in a pickle file with bboxes saved
        # Each item in the "all_bboxes" list will contain a
        # list of boxes for one of the images shown above
        #box_list = boxes #pickle.load(open("bbox_pickle.p", "rb"))
        #print(bbox)
        # Read in image similar to one shown above
        #image3 = img #mpimg.imread('images/ejemplo1.jpg')  #changed from mpimg
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        heat = np.zeros_like(image3[:, :, 0]).astype(np.float)

        # Add heat to each box in box list
        heat = add_heat(heat, box_list)

        # Apply threshold to help remove false positives
        heat = apply_threshold(heat, 8)

        # Visualize the heatmap when displaying
        heatmap = np.clip(heat, 0, 255)

        # Find final boxes from heatmap using label function
        image3 = cv2.cvtColor(image3, cv2.COLOR_BGR2RGB)
        labels = label(heatmap)
        draw_img = draw_labeled_bboxes(np.copy(image3), labels)

        fig = plt.figure()
        plt.subplot(121)
        plt.imshow(draw_img)
        plt.title('Car Positions')
        plt.subplot(122)
        plt.imshow(heatmap, cmap='hot')
        plt.title('Heat Map')
        fig.tight_layout()
        plt.show()






        imagen_prueba = pipeline(img)
        plt.imshow(out_img)
        plt.show()
  
}
        # Apply process image to video.
        white_output = 'images/test_video_Processed.mp4'
        clip1 = VideoFileClip("images/test_video.mp4")
        white_clip = clip1.fl_image(pipeline)  # NOTE: this function expects color images!!
        white_clip.write_videofile(white_output, audio=False)
        '''
        ### pickle models
        # modelp.pickle, HSV (32,32), 32 bins
        # modelC.pickle, YCrCb (16,16), 16 bins
        # modelC32.pickle, YCrCb (32,32), 32 bins
        # modelR32.pickle, RGB  (32,32), 32 bins
        # modelH32.pickle, HSV (32,32), 32 bins
        # modelH320.pickle, HSV (32,32), 32 bins Hog channel 0