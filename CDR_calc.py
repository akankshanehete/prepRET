from scipy import signal
import cv2
import sys
import glob
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import os
from PIL import Image
import xlrd 
import math
from pylab import*
from scipy import signal
import numpy as np
from matplotlib import pyplot as plt
from skimage.feature import graycomatrix, graycoprops
from skimage import data
from skimage import io
import matplotlib.image as mpimg
import matplotlib.cm as cm
import scipy.ndimage as ndimage



def load_image(path):
    # returns an image of dtype int in range [0, 255]
    return np.asarray(Image.open(path))

#function to load image and their name
def load_set(folder, shuffle=False):
    img_list = sorted(glob.glob(os.path.join(folder, '*.png')) + \
                      glob.glob(os.path.join(folder, '*.jpg')) + \
                      glob.glob(os.path.join(folder, '*.jpeg')))
    if shuffle:
        np.random.shuffle(img_list)
    data = []
    filenames = []
    for img_fn in img_list:
        img = load_image(img_fn)
        data.append(img)
        filenames.append(img_fn)
    return data, filenames


#DATA EXTRACTION FUNCTION
#db_folder = drishti dataset folder
#cdr = set it true to get the cdr values of 4 experts
#train_data = setting it true gives training data and false gives testing data
def extract_DRISHTI_GS_train(db_folder,cdr,train_data = True):

    file_codes_all,exp1,exp2,exp3,exp4 = [], [], [], [], []
    if train_data:
        set_path = 'DRISTHI-DATASET/Drishti-GS1_files/Training'
    else:
        set_path = 'DRISTHI-DATASET/Drishti-GS1_files/Test'
    images_path = set_path + '/Images'
    
    X_all, file_names = load_set(images_path)
    rel_file_names = [os.path.split(fn)[-1] for fn in file_names]
    rel_file_names_wo_ext = [fn[:fn.rfind('.')] for fn in rel_file_names]
    if train_data:
        file_codes = [fn[fn.find('_'):] for fn in rel_file_names_wo_ext]
    else:
        file_codes = [fn[fn.find('_'):] for fn in rel_file_names_wo_ext]
    file_codes_all.extend(file_codes)
    
    for fn in rel_file_names_wo_ext:
        if cdr:
            if train_data:
                CDR = open(os.path.join(set_path, 'GT', fn,fn + '_cdrValues.txt'),'r')
            else:
                CDR = open(os.path.join(set_path, 'Test_GT', fn,fn + '_cdrValues.txt'),'r')
            CDR = list(CDR)
            CDR = CDR[0].split()
            exp1.append(CDR[0])
            exp2.append(CDR[1])
            exp3.append(CDR[2])
            exp4.append(CDR[3])
            
    return X_all, file_codes_all,exp1,exp2,exp3,exp4,file_names
    #This functions returns the data images,their names and the corresponding cdr values of each expert in order

def is_near_perimeter(coord, image_shape, threshold):
    x, y = coord
    height = image_shape[0]
    width = image_shape[1]
    return ((x < threshold) or (y < threshold) or (x > (width - threshold)) or (y > (height - threshold)))

def isolate_ROI(image):
    #io.imshow(image)
    b,g,r = cv2.split(image)
    g = cv2.GaussianBlur(g,(15,15),0)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))
    g = ndimage.grey_opening(g,structure=kernel)	
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(g)
    #print(maxLoc)

    if is_near_perimeter(maxLoc, image.shape, 143):
        return np.asarray(image[400:1400,500:1600,:])
    
    # width of image size to extract region of interest
    half_width = 550  
    half_height = 500 
    x0 = int(maxLoc[0])-half_width
    y0 = int(maxLoc[1])-half_height
    x1 = int(maxLoc[0])+half_width
    y1 = int(maxLoc[1])+half_height
    return np.asarray(image[y0:y1,x0:x1])

def imshow(image):
  image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  io.imshow(image_rgb)

def convert(img):
    imin = img.min()
    imax = img.max()

    a = (255 - 0) / (imax - imin)
    b = 255 - a * imax
    new_img = (a * img + b).astype(np.uint8)
    return new_img


# FUNCTION TO SEGMENT CUP AND DISK
def segment(image, show_segmented = False):
    # use diff method for extracting region of interest
    #image = image[400:1400,500:1600,:] 
    image = isolate_ROI(image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(10,10))
    Abo,Ago,Aro = cv2.split(image)  #splitting into 3 channels
    #Aro = clahe.apply(Aro)
    Ago = clahe.apply(Ago)
    M = 60    #filter size
    filter = signal.gaussian(M, std=6) #Gaussian Window
    filter=filter/sum(filter)
    STDf = filter.std()  #It'standard deviation
    

    Ar = Aro - Aro.mean() - Aro.std() #Preprocessing Red
    
    Mr = Ar.mean()                           #Mean of preprocessed red
    SDr = Ar.std()                           #SD of preprocessed red
    Thr = 0.5*M - STDf - Ar.std()            #Optic disc Threshold
    #print(Thr)

    Ag = Ago - Ago.mean() - Ago.std()		 #Preprocessing Green
    Mg = Ag.mean()                           #Mean of preprocessed green
    SDg = Ag.std()                           #SD of preprocessed green
    Thg = 0.5*Mg +2*STDf + 2*SDg + Mg        #Optic Cup Threshold

    
    r,c = Ag.shape
    disk_segmented = np.zeros(shape=(r,c)) #Segmented disc image initialization
    cup_segmented = np.zeros(shape=(r,c)) #Segmented cup image initialization

    #Using obtained threshold for thresholding of the fundus image
    for i in range(1,r):
        for j in range(1,c):
            if Ar[i,j]>Thr:
                disk_segmented[i,j]=255
            else:
                disk_segmented[i,j]=0

    for i in range(1,r):
        for j in range(1,c):
        
            if Ag[i,j]>Thg:
                cup_segmented[i,j]=1
            else:
                cup_segmented[i,j]=0

    # plots segmented images (without predefined borders)
    if show_segmented == True: 
        fig, axs = plt.subplots(1, 3, figsize=(10, 5))
        axs[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axs[0].set_title("ROI")
        axs[0].axis('off')
        axs[1].imshow(disk_segmented)
        axs[1].set_title("Segmented Optic Disc")
        axs[1].axis('off')
        axs[2].imshow(cup_segmented)
        axs[2].set_title("Segmented Optic Cup")
        axs[2].axis('off')
        plt.show()
    

    cup_segmented = convert(cup_segmented)
    disk_segmented = convert(disk_segmented)

    #print("Image Type:", disk_segmented.dtype, cup_segmented.dtype)
    return disk_segmented, cup_segmented
    


# FUNCTION TO CALCULATE CDR

def cdr(cup,disc,show_plot = False, roi = []):
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(10,10))

    # morphological closing and opening operations 
    R1 = cv2.morphologyEx(cup, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2)), iterations = 1)
    r1 = cv2.morphologyEx(R1, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)), iterations = 1)
    R2 = cv2.morphologyEx(r1, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(1,21)), iterations = 1)
    r2 = cv2.morphologyEx(R2, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(21,1)), iterations = 1)
    R3 = cv2.morphologyEx(r2, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(33,33)), iterations = 1)	
    r3 = cv2.morphologyEx(R3, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(43,43)), iterations = 1)

    img = clahe.apply(r3)
    
    
    ret,thresh = cv2.threshold(cup,127,255,0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) #Getting all possible contours in the segmented image
    img = thresh
    cup_diameter = 0
    largest_area = 0
    el_cup = contours[0]
    if len(contours) != 0:
        for i in range(len(contours)):
            if len(contours[i]) >= 5:
                area = cv2.contourArea(contours[i]) #Getting the contour with the largest area
                if (area>largest_area):
                    largest_area=area
                    index = i
                    el_cup = cv2.fitEllipse(contours[i])
                
    cv2.ellipse(img,el_cup,(140,60,150),3)  #fitting ellipse with the largest area
    x,y,w,h = cv2.boundingRect(contours[index]) #fitting a rectangle on the ellipse to get the length of major axis
    #print(w, h)
    # CHANGE THIS LATER IF IT INCREASES THE ACCURACY
    cup_diameter = max(w,h) #major axis is the diameter
    

    # morphological closing and opening operations
    R1 = cv2.morphologyEx(disc, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2)), iterations = 1)
    r1 = cv2.morphologyEx(R1, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)), iterations = 1)
    R2 = cv2.morphologyEx(r1, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(1,21)), iterations = 1)
    r2 = cv2.morphologyEx(R2, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(21,1)), iterations = 1)
    R3 = cv2.morphologyEx(r2, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(33,33)), iterations = 1)
    r3 = cv2.morphologyEx(R3, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(43,43)), iterations = 1)

    img2 = clahe.apply(r3)
    
    ret,thresh = cv2.threshold(disc,127,255,0)
    contours,hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) #Getting all possible contours in the segmented image
    img2 = thresh
    disk_diameter = 0
    largest_area = 0
    el_disc = el_cup
    if len(contours) != 0:
          for i in range(len(contours)):
            if len(contours[i]) >= 5:
                area = cv2.contourArea(contours[i]) # Getting the contour with the largest area
                if (area > largest_area):
                    largest_area=area
                    index = i
                    el_disc = cv2.fitEllipse(contours[i])
                    
    cv2.ellipse(img2,el_disc,(140,60,150),3) # fitting ellipse with the largest area
    x,y,w,h = cv2.boundingRect(contours[index]) #fitting a rectangle on the ellipse to get the length of major axis
    disk_diameter = max(w,h) #major axis is the diameter
                
    if show_plot == True and roi != []:
        fig, axs = plt.subplots(1, 3, figsize=(10, 5))
        axs[0].imshow( cv2.ellipse(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB),el_disc,(140,60,150),3))
        axs[0].set_title("Isolated ROI")
        axs[0].axis('off')
        axs[1].imshow(img2)
        axs[1].set_title("Estimated Disc Boundary")
        axs[1].axis('off')
        axs[2].imshow(img)
        axs[2].set_title("Estimated Cup Boundary")
        axs[2].axis('off')
        plt.show()

    elif show_plot == True and roi == []:
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(img2)
        axs[0].set_title("Estimated Disc Boundary")
        axs[0].axis('off')
        axs[1].imshow(img)
        axs[1].set_title("Estimated Cup Boundary")
        axs[1].axis('off')
        plt.show()

        
    if(disk_diameter == 0): return 1 # disc segmentation failure, results are not accurate enough and CDR may be infinity
    
    pred_cdr = cup_diameter/disk_diameter # ratio of major axis of cup and disc
    return pred_cdr

