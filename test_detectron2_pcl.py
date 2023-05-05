'''
Author: Xiaolin Lin xlim1996@outlook.com
Date: 2023-03-24 11:55:06
LastEditors: Xiaolin Lin xlim1996@outlook.com
LastEditTime: 2023-05-05 10:52:40
FilePath: /hiwi/test_detectron2.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import time
import torch
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from PIL import Image
import open3d as o3d

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from detectron2.engine.defaults import create_ddp_model
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer
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
file_path = 'real_image_depth_maps/depth_maps_150622'
file_list = os.listdir(file_path)
rgb_result_path='rgb_result/VItdet_new'
save = True

def draw_bbox(image,bboxs):
    for bbox in bboxs:
        cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0,255,0), 2)
    return image

with torch.inference_mode():

    file_list = os.listdir(file_path)
    for file_name in file_list:
        # load data
        data = np.load(os.path.join(file_path, file_name),allow_pickle=True).item()
        #get image
        image = data['image'][..., :3]
        image = np.array(image, dtype=np.uint8)
        image_model = np.moveaxis(image, -1, 0)
        #get depth
        depth_map = data['depth']
        intrinsics_matrix=data['intrinsics_matrix']

        time_start = time.time()
        output = model([{'image': torch.from_numpy(image_model)}])
        time_end = time.time()
        print('time cost', time_end - time_start, 's')
        # print(output[0]["instances"].pred_classes, output[0]["instances"].pred_boxes)
        time_start = time.time()
        result=[]
        if "instances" in output[0]:
            instances = output[0]["instances"].to("cpu")
            for i in range(len(instances.pred_classes)):
                if instances.pred_classes[i]+1 in id:
                    result.append(instances[i])
            result=Instances.cat(result)
        print(result.pred_classes.shape)
        print(result.pred_masks.shape)
        input()
        time_end = time.time()
        print('time cost', time_end - time_start, 's')

        v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(cfg.dataloader.train.dataset.names), scale=1.2)
        # v = Visualizer(image_show[:, :, ::-1], metadata, scale=1.2)
        v = v.draw_instance_predictions(result.to("cpu"))
        cv2_imshow(v.get_image()[:, :, ::-1])
        print(result.pred_classes, result.pred_boxes)
        image = draw_bbox(image,result.pred_boxes.tensor.numpy().astype(np.int32))
        plt.imshow(image)
        plt.show()
        print(image.shape)
        print(depth_map.shape)
        # create depth map and rgb image
        h,w = depth_map.shape
        depth_map=o3d.geometry.Image(depth_map)
        rgb = o3d.geometry.Image(image)
        # create camera intrinsic
        intrinsic = o3d.camera.PinholeCameraIntrinsic(width=w, height=h, fx=525.97467041, fy=525.97467041, cx=648.17633057, cy=358.7739563)
     
        # create rgbd image from depth map and rgb image
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb, depth_map, convert_rgb_to_intensity=False)
            
        # create point cloud from depth rgbd image
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image, intrinsic)

        # visualize point cloud
        o3d.visualization.draw_geometries([pcd])

        #try add segmentation to pcl    

        break

