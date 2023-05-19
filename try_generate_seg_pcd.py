
import time
import torch
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from PIL import Image
import open3d as o3d

import pyzed.sl as sl

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from detectron2.engine.defaults import create_ddp_model
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer,GenericMask
from detectron2.structures.instances import Instances


def cv2_imshow(img_bgr):
    plt.rcParams['figure.figsize'] = (18, 36)
    plt.axis('off')
    print(img_bgr.shape)
    print(img_bgr[...,::-1].shape)
    plt.imshow(img_bgr[...,::-1])
    plt.show()


cfg = LazyConfig.load('projects/ViTDet/configs/LVIS_MIX/cascade_mask_rcnn_vitdet_h_100ep.py')
metadata = MetadataCatalog.get(cfg.dataloader.train.dataset.names) # to get labels from ids
classes = metadata.thing_classes

id=[]
for i in range(len(classes)):
    id.append(metadata.get('class_image_count')[i]['id'])
cfg = LazyConfig.load('projects/ViTDet/configs/LVIS/cascade_mask_rcnn_vitdet_h_100ep.py')
cfg.train.init_checkpoint = "checkpoints/ViTDet/model_final_11bbb7.pkl" # replace with the path were you have your model


model = instantiate(cfg.model)
model.to(cfg.train.device)
model = create_ddp_model(model)
DetectionCheckpointer(model).load(cfg.train.init_checkpoint)

model.eval()
print("model loaded",model)
file_path = 'ZED_camera_data'
file_list = os.listdir(file_path)
rgb_result_path='./rgb_result/VItdet_new'
save = True

# Create a ZED camera
zed = sl.Camera()
# Create a InitParameters object and set configuration parameters
init_params = sl.InitParameters()
#TODO: change depth mode 
init_params.depth_mode = sl.DEPTH_MODE.NEURAL  # Use PERFORMANCE depth mode  
init_params.coordinate_units = sl.UNIT.METER  # Use meter units (for depth measurements)
init_params.camera_resolution = sl.RESOLUTION.HD720  # Use HD1080 video mode
init_params.depth_minimum_distance = 0.2
init_params.camera_fps = 30  # Set fps at 30

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
# Grab an image, a RuntimeParameters object must be given to grab()
if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
    # A new image is available if grab() returns SUCCESS
    zed.retrieve_image(image, sl.VIEW.LEFT)
    zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
    image = image.get_data()
    depth = depth.get_data()
# Close the camera
zed.close()
cv2.imwrite("image.png",image)
#TODO: check the depth range

with torch.inference_mode():
    
    # pcd = o3d.io.read_point_cloud(os.path.join(file_path, pcd_name))
    # o3d.visualization.draw_geometries([pcd])
    
    H_o, W_o = image.shape[:2]
    #TODO: need to change the image shape to decide whether to resize or not
    #resize image to half size(solve the problem of detectron2 input size)
    # H_detect = 800
    # W_detect = 1280
    image = cv2.resize(image,(int(W_o / 2), int(H_o / 2)))
    #convert data type of image
    image = np.array(image, dtype=np.uint8)
    H, W = image.shape[:2]
    image_model = np.moveaxis(image, -1, 0)        
    time_start = time.time()
    output = model([{'image': torch.from_numpy(image_model)}])
    time_end = time.time()
    print('time cost', time_end - time_start, 's')
    time_start = time.time()
    result=[]
    #filter the result by id
    if "instances" in output[0]:
        instances = output[0]["instances"].to("cpu")
        for i in range(len(instances.pred_classes)):
            if instances.pred_classes[i]+1 in id:
                result.append(instances[i])
        result=Instances.cat(result)
    time_end = time.time()
    print('time cost', time_end - time_start, 's')
    v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(cfg.dataloader.train.dataset.names), scale=1.2)
    # v = Visualizer(image_show[:, :, ::-1], metadata, scale=1.2)
    v = v.draw_instance_predictions(result.to("cpu"))
    output = v.get_image()[:, :, ::-1]
    print("output",output.shape)
    #visualize the result
    cv2_imshow(output)
    #resize image back to original size
    image_result = cv2.resize(output, (W_o, H_o))

    #instance segmentation result to point cloud
    intrinsic_matrix = o3d.camera.PinholeCameraIntrinsic(width=W_o, height=H_o, fx=fx, fy=fy, cx=cx, cy=cy)

    rgbd_image=o3d.geometry.create_rgbd_image_from_color_and_depth(image_result,depth,depth_scale=1.0,depth_trunc=3.0,convert_rgb_to_intensity=False)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image, intrinsic_matrix)
    o3d.visualization.draw_geometries([pcd]) 

        

        