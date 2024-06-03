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
import open3d as o3d
from sklearn.preprocessing import RobustScaler

import pandas as pd
import random
from scipy.spatial import Delaunay
from scipy.spatial import ConvexHull





def open3d_area_conv_hull(points_mm):
    # Create an Open3D point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_mm)

    # Fit a plane to the point cloud
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=100)
    normal_vector = plane_model[:3]  # Normal vector of the fitted plane

    # Project the points onto the plane defined by the normal vector
    projected_points = points_mm - np.outer(points_mm.dot(normal_vector), normal_vector)

    # Compute the convex hull of the projected points
    hull = ConvexHull(projected_points[:, :2])  # Consider only x and y coordinates

    # Calculate the total planar area (area of the convex hull)
    total_area = hull.volume
    return total_area



def remove_lonely_outliers(points_mm, radius=2.0, min_neighbors=11):
    """
    Remove lonely outliers from a point cloud based on neighbor count.

    Args:
        points_mm (numpy.ndarray): Input point cloud represented as a numpy array.
        radius (float): Radius within which neighbors will be searched.
        min_neighbors (int): Minimum number of neighbors a point should have to not be considered an outlier.

    Returns:
        numpy.ndarray: Inlier points without lonely outliers.
    """
    # Create a KD-Tree for efficient nearest neighbor search
    tree = cKDTree(points_mm)

    # Initialize an empty list to store indices of lonely outliers
    lonely_outliers_indices = []

    # Iterate through each point in the point cloud
    for i, point in enumerate(points_mm):
        # Find the indices of neighbors within the specified radius
        neighbor_indices = tree.query_ball_point(point, r=radius)

        # Exclude the point itself from the neighbor count
        num_neighbors = len(neighbor_indices) - 1
        #print(num_neighbors,",", end =" ")

        # Check if the point has fewer neighbors than the minimum threshold
        if num_neighbors < min_neighbors:
            lonely_outliers_indices.append(i)

    # Extract the inlier points (exclude lonely outliers)
    inlier_indices = [i for i in range(len(points_mm)) if i not in lonely_outliers_indices]
    inlier_points = points_mm[inlier_indices]

    #print("removed outlier : ",len(lonely_outliers_indices) ,"/" , len(points_mm) )

    return inlier_points



def dimension_evaluator(points_mm):
    # Create a point cloud from the input lists
    #points_mm = z_score_normalization(points_mm,5)

    points_mm = remove_lonely_outliers(points_mm)
    area_hull = open3d_area_conv_hull(points_mm)
    #area_delunay = delunay_area(points_mm)

    # Calculate the centroid
    centroid = np.mean(points_mm, axis=0)
    z_mean = centroid[2]



    # Create an Open3D point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_mm)

    # Set the centroid color
    centroid_color = [1.0, 0.0, 0.0]  # Red color
    # Add a new point at (centroid_x, centroid_y, centroid_z + 1)
    new_point = [centroid[0], centroid[1], centroid[2] + 1.0]
    pcd.points.append(new_point)  # Append the new point to the point cloud
    # Generate random RGB colors for each point
    # Define a fixed color (e.g., red)
    fixed_color = [1.0, 1.0, 0.0]  # Red color (R, G, B values)

    # Create a color array with the fixed color for all points
    colors = [fixed_color] * len(points_mm[:, 0])

    # Assign the color array to the point cloud
    pcd.colors = o3d.utility.Vector3dVector(colors)



    # Calculate the oriented bounding box
    obb = pcd.get_oriented_bounding_box()

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(obb.get_box_points())
    lines = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]
    line_set.lines = o3d.utility.Vector2iVector(lines)

    # Set the color (R, G, B) for the lines (e.g., red color)
    line_set.colors = o3d.utility.Vector3dVector(np.array([1.0, 0.0, 0.0]) * np.ones((len(lines), 3)))

    # Get the dimensions and transformation matrix of the oriented bounding box
    dimensions = obb.extent
    transformation_matrix = obb.R

    # Extract dimensions along x, y, and z
    length = dimensions[0]
    width = dimensions[1]
    height = dimensions[2]
    width_real_hull = area_hull / length
   # width_real_deluney = area_delunay / length
    #print(" hull d:",width_real_hull," fake d",width)




    if SHOW_PC_WITH_BOX:
        print("box Dimensions (length, width, height):", length, width, height)


        # Create a visualization object
        vis = o3d.visualization.Visualizer()

        # Add the point clouds and oriented bounding box
        vis.create_window()
        vis.add_geometry(pcd)
        # vis.add_geometry(pcd_or)
        vis.add_geometry(obb)

        # Add the grid point cloud
        # vis.add_geometry(grid_pcd)

        # Set background color (light gray)
        opt = vis.get_render_option()
        opt.background_color = np.asarray([0.6, 0.6, 0.6])

        # Visualize the scene
        vis.run()
        vis.destroy_window()

    return length,width_real_hull, z_mean, pcd , line_set





def split_PC(points_mm):


    # Create a point cloud from the input lists
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points_mm)

    # Compute the mean of the point cloud to find the centroid
    centroid = np.mean(points_mm, axis=0)
    # Compute the mean of the point cloud
    mean = np.mean(points_mm, axis=0)

    # Calculate the covariance matrix
    cov_matrix = np.cov((points_mm - centroid).T)

    # Compute the eigenvalues and eigenvectors of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Sort eigenvalues and eigenvectors in descending order
    sorted_indices = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    # Define the normal vector of the plane as the first principal component
    normal = eigenvectors[:, 0]  # First principal component

    # Define a point on the plane as the centroid
    point_on_plane = centroid

    ## Calculate the projection of each point onto the PCA normal vector
    distances = np.dot(points_mm - point_on_plane, normal)

    # Define a threshold to split the point cloud
    split_threshold = 0.0  # Adjust the threshold as needed

    # Classify points as belonging to one half or the other based on the projection
    half1_indices = np.where(distances >= split_threshold)
    half2_indices = np.where(distances < split_threshold)

    # Create two point clouds: one for each half with different colors
    half1_colors = np.array([[1.0, 0.0, 0.0] for _ in range(len(half1_indices[0]))])  # Red color for half1
    half1_cloud = point_cloud.select_by_index(half1_indices[0].tolist())
    half1_cloud.colors = o3d.utility.Vector3dVector(half1_colors)

    half2_colors = np.array([[0.0, 0.0, 1.0] for _ in range(len(half2_indices[0]))])  # Blue color for half2
    half2_cloud = point_cloud.select_by_index(half2_indices[0].tolist())
    half2_cloud.colors = o3d.utility.Vector3dVector(half2_colors)

    half1_points = np.asarray(half1_cloud.points)
    half2_points = np.asarray(half2_cloud.points)


    if VISUALIZE_SPLITTING_OP:


        # Define the size of the plane
        plane_size = 50.0  # Adjust the size as needed

        # Calculate two vectors orthogonal to the normal vector
        v1 = np.cross(normal, [0, 0, 1])  # Assuming [0, 0, 1] as an arbitrary reference vector
        v1 = v1 / np.linalg.norm(v1)  # Normalize v1
        v2 = np.cross(normal, v1)  # Calculate v2 orthogonal to normal and v1

        # Define points to represent the larger orthogonal plane
        plane_points = [
            centroid + plane_size * (v1 + v2),
            centroid + plane_size * (-v1 + v2),
            centroid + plane_size * (-v1 - v2),
            centroid + plane_size * (v1 - v2),
        ]

        # Create a line set to represent the orthogonal plane
        plane_line_set = o3d.geometry.LineSet()
        plane_line_set.points = o3d.utility.Vector3dVector(plane_points)
        lines = [[0, 1], [1, 2], [2, 3], [3, 0]]
        plane_line_set.lines = o3d.utility.Vector2iVector(lines)

        # Visualize the splits and the splitting plane
        o3d.visualization.draw_geometries([half1_cloud,half2_cloud, plane_line_set])





    return half1_points,half2_points



def visualize_3D_list_pointcloud(points_mm):


    # Stack the arrays horizontally to create the final matrix

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_mm)

    # Create a visualization object
    vis = o3d.visualization.Visualizer()

    # Add the point clouds and oriented bounding box
    vis.create_window()
    vis.add_geometry(pcd)
    #vis.add_geometry(pcd_or)

    # Add the grid point cloud
    #vis.add_geometry(grid_pcd)

    # Set background color (light gray)
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0.6, 0.6, 0.6])

    # Visualize the scene
    vis.run()
    vis.destroy_window()


def radius_outlier_removal(point_cloud, radius_threshold = 4, min_neighbors =4):
    # Calculate the distance to the k-nearest neighbors
    kdtree = cKDTree(point_cloud)
    distances, _ = kdtree.query(point_cloud, k=min_neighbors + 1)

    # Calculate mean distance to neighbors excluding itself
    mean_distances = np.mean(distances[:, 1:], axis=1)

    # Apply radius outlier removal
    outlier_indices = mean_distances > radius_threshold
    #print(outlier_indices)
    filtered_cloud = point_cloud[~outlier_indices]

    print("removed:", len(point_cloud) - len(filtered_cloud), "/",len(point_cloud), " outl. points")

    return filtered_cloud


def z_score_normalization(point_cloud, threshold_factor=0.05):
    # Extract z-values
    z_values = point_cloud[:, 2]



    # Check if all points are equiplanar in the z-axis
    if np.all(np.isclose(z_values - np.mean(z_values), 0)):
        print("EQUIPLANAR. ADDING POINT.")
        corrected_point_cloud = np.copy(point_cloud)
        # Calculate the centroid of the corrected point cloud
        centroid = np.mean(corrected_point_cloud, axis=0)

        # Create a new point at (centroid_x, centroid_y, centroid_z + 1)
        new_point = [centroid[0], centroid[1], centroid[2] + 1.0]


        # Insert the new point into the corrected point cloud
        corrected_point_cloud = np.vstack((corrected_point_cloud, new_point))
    else:
        # Calculate mean and standard deviation of z-values
        z_mean = np.mean(z_values)
        z_std = np.std(z_values)

        # Calculate z-scores
        z_scores = (z_values - z_mean) / z_std

        # Apply a correction to the z-values based on their z-scores
        corrected_z_values = z_mean + (z_scores * threshold_factor * z_std)

        # Update the z-values in the point cloud
        corrected_point_cloud = np.copy(point_cloud)
        corrected_point_cloud[:, 2] = corrected_z_values
        centroid = np.mean(corrected_point_cloud, axis=0)

        # Create a new point at (centroid_x, centroid_y, centroid_z + 1)
        new_point = [centroid[0], centroid[1], centroid[2] + 1.0]
        corrected_point_cloud = np.vstack((corrected_point_cloud, new_point))



    return corrected_point_cloud


