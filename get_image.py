
import time
import torch
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from PIL import Image
import open3d as o3d

import pyzed.sl as sl


# Create a ZED camera
zed = sl.Camera()
# Create a InitParameters object and set configuration parameters
init_params = sl.InitParameters()
#TODO: change depth mode 
init_params.depth_mode = sl.DEPTH_MODE.NEURAL  # Use PERFORMANCE depth mode  
init_params.coordinate_units = sl.UNIT.METER  # Use meter units (for depth measurements)
init_params.camera_resolution = sl.RESOLUTION.HD720  # Use HD1080 video mode
init_params.camera_fps = 15  # Set fps at 30

#TODO: check extrinsic matrix
# rotation = calibration_params.get_rotation_matrix()
# translation = calibration_params.get_translation_vector()
# extrinsic_matrix = np.array([[rotation[0], rotation[1], rotation[2], translation[0]],
#                              [rotation[3], rotation[4], rotation[5], translation[1]],
#                              [rotation[6], rotation[7], rotation[8], translation[2]],
#                              [0, 0, 0, 1]])

# Open the camera
err = zed.open(init_params)
if err != sl.ERROR_CODE.SUCCESS:
    exit(1)
calibration_params=zed.get_camera_information().camera_configuration.calibration_parameters
fx=calibration_params.left_cam.fx
fy=calibration_params.left_cam.fy
cx=calibration_params.left_cam.cx
cy=calibration_params.left_cam.cy
#capture image and depth from zed camera
image = sl.Mat()
depth = sl.Mat()
runtime_parameters = sl.RuntimeParameters()
i = 0
file_path="zed_data"
while(True):
        # Grab an image, a RuntimeParameters object must be given to grab()
    if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
        # A new image is available if grab() returns SUCCESS
        zed.retrieve_image(image, sl.VIEW.LEFT)
        zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
        image_save = image.get_data()
        depth_save = depth.get_data()
    # Close the camera
    
        cv2.imwrite(os.path.join(file_path,"image_{}.png".format(i)),image_save)
        cv2.imwrite(os.path.join(file_path,"depth_{}.png".format(i)),depth_save)
        input()
    i +=1

#TODO: check the depth range

        

        