from scipy.spatial import distance as dist
import os
import cv2
import math
import time
import numpy as np
import statistics
import pyrealsense2 as rs
import csv
from scipy.spatial import cKDTree
from datetime import datetime
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import time
from CONFIGURATION_VISION import *
from pointcloud_utility_library import *
import open3d as o3d
from sklearn.preprocessing import RobustScaler
import pandas as pd
import random
from scipy.spatial import Delaunay
from scipy.spatial import ConvexHull





def clean_measurements(diameter_list, length_list, threshold_multiplier=3):
    # Calculate the mean and standard deviation of the measurements
    mean_diameter = np.mean(diameter_list)
    std_dev_diameter = np.std(diameter_list)
    mean_length = np.mean(length_list)
    std_dev_length = np.std(length_list)

    # Initialize empty lists to store the indices of potential outliers
    diameter_outliers = []
    length_outliers = []

    # Iterate through the diameter_list to identify potential outliers
    for j, diameter in enumerate(diameter_list):
        # Calculate the absolute deviation from the mean diameter
        deviation = diameter - mean_diameter

        # Check if the deviation exceeds the threshold
        if deviation > threshold_multiplier * std_dev_diameter:
            # If it does, add the index to the list of diameter outliers
            diameter_outliers.append(j)
            # print("bad diam:",diameter_list[j],[j])

    # Iterate through the length_list to identify potential outliers
    for i, length in enumerate(length_list):
        # Calculate the absolute deviation from the mean length
        deviation = length - mean_length

        # Check if the deviation exceeds the threshold
        if deviation > threshold_multiplier * std_dev_length:
            # If it does, add the index to the list of length outliers
            length_outliers.append(i)
            # print("bad len:", length_list[i],[i])

    # Print the indices of potential outliers
    # print("OUTLIER, L: ", len(length_outliers), "/", len(length_list) , ", D:", len(diameter_outliers), "/", len(length_list))
    # print("mean diam", mean_diameter)




    # Replace outliers with the mean of nearby measurements
    for outlier_index_d in diameter_outliers:
        # Replace the diameter measurement with the mean of nearby measurements
        if outlier_index_d > 0 and outlier_index_d < len(diameter_list) - 1:
            print( "substituted diameter ",[outlier_index_d], diameter_list[outlier_index_d], "  with mean :",diameter_list[outlier_index_d - 1], diameter_list[outlier_index_d + 1])

            diameter_list[outlier_index_d] = np.mean([diameter_list[outlier_index_d - 1], diameter_list[outlier_index_d + 1]])
    # print("leght: ",length_list)
    # print("mean len", mean_length)
    for outlier_index_l in length_outliers:
        # Replace the length measurement with the mean of nearby measurements
        if outlier_index_l > 0 and outlier_index_l < len(length_list) - 1:
            print( "substituted length ",[outlier_index_l],length_list[outlier_index_l] ," with mean :",length_list[outlier_index_l - 1], length_list[outlier_index_l + 1])

            length_list[outlier_index_l] = np.mean([length_list[outlier_index_l - 1], length_list[outlier_index_l + 1]])

    # Return the cleaned lists of diameters and lengths
    return diameter_list, length_list



def delete_csv_file(filenam_path):
    # Delete the existing CSV file (if it exists)
    if os.path.exists(filenam_path):
        print("REMOVE:", filenam_path)
        os.remove(filenam_path)
    else:
        print(filenam_path, " no such file in memory")






def timeit(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        processing_time = end_time - start_time
        print(f"Processing time of {func.__name__}: {processing_time:.6f} seconds")
        return result

    return wrapper

def plot_histogram(values):
    plt.hist(values, bins=100, edgecolor='black')
    mean_vol = np.mean(values)
    plt.vlines(mean_vol, 0, 50, linestyles ="dashed", colors ="g")
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.title('Histogram of Values, mean:' +  str(int(mean_vol)) +  " STD:"  +  str(int(np.std(values))))
    plt.show()



def save_to_csv_recursive(value, file_path):
    with open(file_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(value)




def resize_image(img, percentage):
    # Display the frame
    # saved in the file
    scale_percent = 70  # percent of original size
    width = int(img.shape[1] * percentage / 100)
    height = int(img.shape[0] * percentage / 100)
    dim = (width, height)

    # resize image
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return resized

