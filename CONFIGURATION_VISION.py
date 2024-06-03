#!/usr/bin/env python
import os

PATH_EXAMPLE = "/home/mmt-ben/Desktop"

#ACQUISITION CONFFIGURATION
TIME_WAITER_REALSENSE_FREEZER = 0.8
basler = True
realsense = False



ZOOM = 100
iteration = 5
#aumenta per includere piu roba
THRES_VALUE = 90
ALPHA_FILTER = 1
PATH_2_AQUIS = "/aquisition/"
PATH_HERE = os.getcwd()
OFFSET_CM_COMPRESSION = 50
KERNEL = 5
POINT_CLOUD_GRAPH = False
L_real = 315
D_real = 73
CONTINOUS_STREAM = 1
#processo di filtraggio dell immagine depth prima della conversione in pointcloud
SHOW_FILTERING_PROCESS = 0
MIN_DEPTH = 503
CONFIDENCE_INTERVAL = 0.5
SHOW_CYLINDRIFICATION = 0
SHOW_CYLINDRIFICATION_RGB = 0
SHOW_HIST_OF_DIAMETERS = 0
SHOW_PC_WITH_BOX = 0
VISUALIZE_SPLITTING_OP = 0
SHOW_MASK_ONLY = 0
POINTCLOUD_EVALUATION_VOLUME = 1
SHOW_DEPTH_ONLY = 0
KERNEL_VALUE_DILATION = 3
DILATION_ITERATION = 4

#tutto il legno con boxx
VISUALIZE_CYLINDRIFICATED_WOOD = 0
csv_file_path = 'data_volume_iteration_' +str(iteration) +  '.csv'

#visdualizza o no i volumi troppo grandi / usare solo se non si attiva il visualizzatore costante

SHOW_VOLUME_TOO_BIG = 0
#valore di trehold dei valori troppo grandi
OUTLIER_VOLUME = 26000


