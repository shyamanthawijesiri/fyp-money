import pandas as pd
import pickle

import mahotas as mt
import cv2 as cv
import numpy as np
import csv



def extract_feature(image):
    (mean, std) = cv.meanStdDev(image)
    color_feature = np.array(mean)
    color_feature = np.concatenate([color_feature, std]).flatten()
    ##Texture Feature
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    textures = mt.features.haralick(gray)
    ht_mean = textures.mean(axis=0)
    ## Shape Features
    ret, thresh = cv.threshold(gray, 127, 255, 0)
    contours, hierarchy = cv.findContours(thresh.copy(), 1, 2)
    cnt = contours[0]
    area = cv.contourArea(cnt)
    perimeter = cv.arcLength(cnt, True)
    shape = np.array([])
    shape = np.append(shape, area)
    shape = np.append(shape, perimeter)
    ht_mean = np.concatenate([ht_mean, color_feature]).flatten()
    ht_mean = np.concatenate([ht_mean, shape]).flatten()
    return (ht_mean)

def create_csv(img):
    mydata = [
        ['energy', 'contrast', 'correlation', 'variance', 'inverse difference moment', 'sum average', 'sum variance',
         'sum entropy', 'entropy', 'difference variance', 'difference entropy', 'info_corr',
         'maximal_corr_coeff', 'mean_B', 'mean_G', 'mean_R', 'std_B', 'std_G', 'std_R', 'area', 'perimeter']]


    feature = extract_feature(img)
    feature = feature.tolist()
    mydata.append(feature)
    myfile = open('inputImage.csv', 'w')
    with myfile:
        writer = csv.writer(myfile)
        writer.writerows(mydata)

def moneyClassification(img):

    create_csv(img)
    test = pd.read_csv('inputImage.csv', sep=',')
    fileName = 'RF'
    load_model = pickle.load(open(fileName, 'rb'))
    results = load_model.predict(test)
    return results






