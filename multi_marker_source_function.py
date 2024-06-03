import numpy as np
import cv2
import os
import sys
from scipy.stats import linregress, norm
from scipy.optimize import curve_fit
from itertools import groupby
from statistics import mean
from scipy.signal import savgol_filter, decimate
from scipy.interpolate import interp1d
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import seaborn as sns
import matplotlib.font_manager
import pandas as pd
import matplotlib.pyplot as plt

# Append the utility library path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utility_library'))

from decorator import *

# Save the DataFrame to an Excel file
output_excel = 'marker_data.xlsx'

### Model all the curves and extract a k (5 k), then plot them against vext, and therefore you can estimate k knowing the V.
# Continue with the optical model by looking for better solutions.
# Do not take the mean and std dev but all speed values for trial and distance are needed.

# Define the calibration folder path
folder_calibration = "CALIBRATION_CAMERA_FILE"
PLOT_SINGLE_PATH = 0

# Load camera calibration parameters
mtx = np.load(os.path.join(folder_calibration, "camera_matrix.npy"))
dist = np.load(os.path.join(folder_calibration, "dist_coeffs.npy"))
print("mtx", mtx)
print("dist", dist)

# Extract parameters from the calibration matrix
fx = mtx[0, 0]  # Focal length along the x-axis
fy = mtx[1, 1]  # Focal length along the y-axis
cx = mtx[0, 2]  # x-coordinate of the projection center
cy = mtx[1, 2]  # y-coordinate of the projection center
print(fx, fy, cx, cy)

# Define the video name and path
video_name = 'GX010118.MP4'
video_path = 'aquisition_raw/' + video_name

# Define the size of the ArUco marker
# MARKER_SIZE = 0.0978 # Small marker test 1 (in cm)
MARKER_SIZE = 0.1557 # Large marker test 2 (in cm)

# Initialize the ArUco marker detector
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
parameters = cv2.aruco.DetectorParameters_create()

# Create a folder for the data
folder_path = 'dati_marker_optical_flow'
os.makedirs(folder_path, exist_ok=True)

# Define the path for the CSV file
csv_path = os.path.join(folder_path, 'dati_marker_optical_flow.csv')

# Open the video
cap = cv2.VideoCapture(video_path)

# Initialize the optical flow detector
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Define the color for drawing optical flow vectors
color = (0, 255, 0)

# List of marker IDs
marker_ids = [7, 8, 9, 10, 11]
# Velocity values
v1 = 0.25
v2 = 0.5
v3 = 0.75
v4 = 0.94
v5 = 0.97
v_all = [v1, v2, v3, v4, v5]

# Define the name of the results Excel file
output_excel_res = 'results.xlsx'

# Check if the results file exists
if os.path.exists(output_excel_res):
    print("File exists, appending results")
else:
    # Define the header for the Excel file
    header = ['timestamp', '7_z', '7_vx', '8_z', '8_vx', '9_z', '9_vx', '10_z', '10_vx', '11_z', '11_vx']

    # Create an empty DataFrame with the header
    df_res = pd.DataFrame(columns=header)

    # Save the DataFrame with the header to the Excel file
    df_res.to_excel(output_excel_res, index=False)

    print(f"Created the file '{output_excel}' with empty headers.")
