import numpy as np

import time
import struct
from numpy.linalg import norm
import os
import statistics
import os
import sys
import math
import cv2
import csv
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2
from datetime import datetime
import pyrealsense2 as rs
from misc_utility_library import *


OFFSET_CM_COMPRESSION = 50

def optional_closing(frame):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    # frame = cv2.erode(frame, kernel, iterations=1)
    frame = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel)
    print("opened")

    return frame
def brightness(img):
    if len(img.shape) == 3:
        # Colored RGB or BGR (*Do Not* use HSV images with this function)
        # create brightness with euclidean norm
        return np.average(norm(img, axis=2)) / np.sqrt(3)
    else:
        # Grayscale
        return np.average(img)
def mask_generation(frame, bottom_color, top_color):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, bottom_color, top_color)

    imask = mask > 0
    imagem = (255 - mask)

    green = 255 * np.ones_like(frame, np.uint8)
    green[imask] = frame[imask]  # dentro i mask metto frame

    return green, imagem
def undesired_objects(image):
    image = image.astype('uint8')
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=4)
    sizes = stats[:, -1]

    max_label = 1
    max_size = sizes[1]
    for i in range(2, nb_components):
        if sizes[i] > max_size:
            max_label = i
            max_size = sizes[i]

    img2 = np.zeros(output.shape)
    img2[output == max_label] = 255
    return img2
def count_files_in_folder(folder_path):
    try:
        # Get the list of files in the folder
        files = os.listdir(folder_path)

        # Filter out directories, leaving only files
        files = [file for file in files if os.path.isfile(os.path.join(folder_path, file))]

        # Count the number of files
        file_count = len(files)
        print("n files:",file_count)

        return file_count

    except FileNotFoundError:
        print(f"The specified folder '{folder_path}' does not exist.")
    except Exception as e:
        print(f"An error occurred: {e}")
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
def crop_with_box_one_shoot(box, mask_bu, frame, depth):
    P1 = box[0]
    P2 = box[1]
    P3 = box[2]
    P4 = box[3]
    margine = 10
    xmax = max(P1[0], P2[0], P3[0], P4[0]) + margine
    xmin = min(P1[0], P2[0], P3[0], P4[0]) - margine
    ymax = max(P1[1], P2[1], P3[1], P4[1]) + margine
    ymin = min(P1[1], P2[1], P3[1], P4[1]) - margine
    # print(box)

    h, w, c = frame.shape

    if xmin > 0 and ymin > 0:
        if xmax <= w and ymax <= h:
            mask_bu = mask_bu[ymin:ymax, xmin:xmax]
            frame = frame[ymin:ymax, xmin:xmax]
            depth = depth[ymin:ymax, xmin:xmax]

            return mask_bu, frame, depth

    print("impossible crop entire image, width:", w, "x min e max:", xmin, xmax, "height: ", h, ymin, ymax)
    return mask_bu, frame, depth
def write_pointcloud(filename, xyz_points, rgb_points=None):
    '''
    Function that writes a .ply file of the point cloud according to the camera points
    and eventually the corresponding color points. Saves the .ply at the specified path.
    INPUTS:
    - filename: full path of the pointcloud to be saved, i.e. ./clouds/pointcloud.ply
    - xyz_points: 3D camera points passed as numpy array npoints x 3
    - rgb_points: corresponding color triplets that will be applied to each point
    OUTPUTS:
    - saves the cloud at the specified path.
    '''

    assert xyz_points.shape[1] == 3, 'ERROR: input XYZ points should be Nx3 float array!'
    # if no color points have been passed, put them at max intensity
    if rgb_points is None:
        rgb_points = np.ones(xyz_points.shape).astype(np.uint8) * 255

    assert xyz_points.shape == rgb_points.shape, 'ERROR: input RGB colors should be Nx3 float array and have same size as input XYZ points!'

    # write header of .ply file
    fid = open(filename, 'wb')
    fid.write(bytes('ply\n', 'utf-8'))
    fid.write(bytes('format binary_little_endian 1.0\n', 'utf-8'))
    fid.write(bytes('element vertex %d\n' % xyz_points.shape[0], 'utf-8'))
    fid.write(bytes('property float x\n', 'utf-8'))
    fid.write(bytes('property float y\n', 'utf-8'))
    fid.write(bytes('property float z\n', 'utf-8'))
    fid.write(bytes('property uchar red\n', 'utf-8'))
    fid.write(bytes('property uchar green\n', 'utf-8'))
    fid.write(bytes('property uchar blue\n', 'utf-8'))
    fid.write(bytes('end_header\n', 'utf-8'))

    # write 3D points to .ply file
    # WARNING: rgb points are assumed to be in BGR format, so saves them
    # in RGB format by inverting columns here
    for i in range(xyz_points.shape[0]):
        fid.write(bytearray(struct.pack("fffccc", xyz_points[i, 0], xyz_points[i, 1], xyz_points[i, 2],
                                        rgb_points[i, 2].tobytes(), rgb_points[i, 1].tobytes(),
                                        rgb_points[i, 0].tobytes())))
    fid.close()
def draw_and_identify_current_cnt(frame, i, box):
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color = (255, 0, 0)
    thickness = 2
    frame = cv2.putText(frame, str(i), (box[0][0], box[0][1] - 100), font,
                        fontScale, color, thickness, cv2.LINE_AA)
def draw_and_calculate_poligonal_max_diameter(cnt, frame):
    epsilon = 0.003 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)

    #cv2.drawContours(frame, [approx], -1, (255, 255, 0), 1)
def draw_and_calculate_rotated_box(cnt, frame):
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    #cv2.drawContours(frame, [box], 0, (0, 0, 255), 2)
    return box
def fit_and_draw_line_cnt(cnt, frame):
    rows, cols = frame.shape[:2]
    [vx, vy, x, y] = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
    lef = int((-x * vy / vx) + y)
    ri = int(((cols - x) * vy / vx) + y)
    cv2.line(frame, (cols - 1, ri), (0, lef), (0, 255, 0), 1)
def first_layer_detect_raw_shoots(im, frame):
    im = (255 - im)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (12, 12))

    im = cv2.dilate(im, kernel, iterations=1)
    im = cv2.morphologyEx(im, cv2.MORPH_CLOSE, kernel)
    edge = cv2.Canny(im, 175, 175)
    kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    edge = cv2.dilate(edge, kernel1, iterations=1)

    contours, hierarchy = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Find the index of the largest contour
    valid = []
    color = 0
    i = 0
    # print("|||||||||||||||||||||||||||||||||||||||||||")





    for cnt in contours:
        i += 1

        # area
        area = cv2.contourArea(cnt)
        # perimeter
        perimeter = cv2.arcLength(cnt, True)
        # fitting a line

        if perimeter != 0 and area != 0:
            circularity = (4 * math.pi * area) / (pow(perimeter, 2))

            # print("AVIABLE:  i A,P,C | ", i," | ", int(area)," | ", int(perimeter), " | ",circularity)
            '''
            if perimeter > 100 and perimeter < 100000:  # 1200  
                if circularity > 0.0005 and circularity < 0.9:  # 0.05 / 0.1, 0.02
                    if area > 300 and area < 120000:  # 2200
            '''
            if area > 1000:

                if circularity < 0.8:
                    if perimeter > 500:
                        # print("first layer CHOSEN! i A,P,C", i, int(area), int(perimeter), circularity)

                        return i
    print("no one aviable")
    return 0
def count_and_display_pixel(green, mask):
    height = green.shape[0]
    width = green.shape[1]
    # org = (int(height/2), int(width/2))
    org = (50, 50)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.6
    color = (255, 0, 0)
    thickness = 1
    number_of_black_pix = np.sum(mask == 0)
    green = cv2.putText(green, 'pix : ' + str(number_of_black_pix), org, font,
                        fontScale, color, thickness, cv2.LINE_AA)
    return number_of_black_pix
def writeCSVdata(time, data):
    """
    write data 2 CSV
    :param data: write to a csv file input data (append to the end)
    :return: nothing
    """
    # scrive su un file csv i dati estratti dalla rete Neurale
    name = ""

    file = open('./data/' + time + '.csv', 'a')
    writer = csv.writer(file)
    writer.writerow(data)
    file.close()
def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)
def two_points_euc_distance(P1, P2):
    #3d
    x1 = P1[0]
    y1 = P1[1]
    z1 = P1[2]
    x2 = P2[0]
    y2 = P2[1]
    z2 = P2[2]

    p1 = np.array([x1, y1, z1])
    p2 = np.array([x2, y2, z2])

    squared_dist = np.sum((p1 - p2) ** 2, axis=0)
    dist = np.sqrt(squared_dist)
    return dist
def create_flatten_array_for_ply_save(pointcloud):
    X, Y, Z = cv2.split(pointcloud)  # For BGR image
    X = X.flatten()
    Y = Y.flatten()
    Z = Z.flatten()

    array = np.array([X, Y, Z])
    # transpose
    array = array.T
    print(array.shape)
    return array
def convert_u8_img_to_u16_d435_depth_image(u8_image):
    # print(u8_image)
    u16_image = u8_image.astype('uint16')
    u16_image_off = u16_image + OFFSET_CM_COMPRESSION
    u16_image_off_mm = u16_image_off * 10
    # print(u16_image_off_mm)
    # print(u16_image_off_mm.shape, type(u16_image_off_mm),u16_image_off_mm.dtype)

    return u16_image_off_mm
def obtain_intrinsics():
    intrinsics = rs.intrinsics()
    with open("intrinsics.csv", "r") as f:
        reader = csv.reader(f)
        for i, line in enumerate(reader):
            if i == 0:
                intrinsics.width = int(line[0])
                intrinsics.height = int(line[1])
                intrinsics.ppx = float(line[2])
                intrinsics.ppy = float(line[3])
                intrinsics.fx = float(line[4])
                intrinsics.fy = float(line[5])

                if str(line[6]) == "distortion.inverse_brown_conrady":
                    intrinsics.model = rs.distortion.inverse_brown_conrady
                else:
                    print("not rec ognized this string for model: ", str(line[6]))
                    intrinsics.model = rs.distortion.inverse_brown_conrady

                listm = line[7].split(",")

                new_list = []
                for i in range(len(listm)):
                    element = listm[i]
                    element = element.replace("[", "")
                    element = element.replace(" ", "")
                    element = element.replace("]", "")
                    element = float(element)
                    new_list.append(element)

                intrinsics.coeffs = new_list

    return intrinsics
def distanceCalculate(p1, p2):
    """p1 and p2 in format (x1,y1) and (x2,y2) tuples"""
    dis = ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5
    return dis
def volume_from_lenght_and_diam_med(box, frame, mask_bu):
    # lenght of the box, aka minimum rectangular distance
    # |__->   L2
    # area : pixel
    # L1(diam_med) = A / L2
    lenght_shoot = distanceCalculate(box[3], box[1])

    pixel = count_and_display_pixel(frame, mask_bu)

    diam_med = pixel / lenght_shoot
    # cv2.line(frame, box[3], box[1], (255, 255, 0), 4) #int(diam_med)

    volume = math.pi * pow((diam_med / 2), 2) * lenght_shoot
    # print("volume: ", volume)
    return volume, frame

