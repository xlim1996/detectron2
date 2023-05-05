'''
Author: Xiaolin Lin xlim1996@outlook.com
Date: 2023-03-24 11:55:06
LastEditors: Xiaolin Lin xlim1996@outlook.com
LastEditTime: 2023-05-05 10:52:59
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
print("id",id)
cfg = LazyConfig.load('projects/ViTDet/configs/LVIS/cascade_mask_rcnn_vitdet_h_100ep.py')
cfg.train.init_checkpoint = "checkpoints/ViTDet/model_final_11bbb7.pkl" # replace with the path were you have your model

model = instantiate(cfg.model)
model.to(cfg.train.device)
model = create_ddp_model(model)
DetectionCheckpointer(model).load(cfg.train.init_checkpoint)

model.eval()
print("model loaded",model)
rgb_path = 'rgb'
rgb_result_path='rgb_result/VItdet_new'
save = True


with torch.inference_mode():

    imagelist = os.listdir(rgb_path)
    for file_name in imagelist:
        # print(file_name)
        img_path = os.path.join(rgb_path, file_name)
        print(img_path)
        image = cv2.imread(img_path, 1)
        print(image.shape)
        input()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_show = np.array(image, dtype=np.uint8)
        image = np.moveaxis(image, -1, 0)
        time_start = time.time()
        output = model([{'image': torch.from_numpy(image)}])
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
        time_end = time.time()
        print('time cost', time_end - time_start, 's')
        v = Visualizer(image_show[:, :, ::-1], MetadataCatalog.get(cfg.dataloader.train.dataset.names), scale=1.2)
        # v = Visualizer(image_show[:, :, ::-1], metadata, scale=1.2)
        v = v.draw_instance_predictions(result.to("cpu"))
        if save:
            Image.fromarray(v.get_image()[:, :, ::-1]).save(os.path.join(rgb_result_path, file_name))
        else:
            cv2_imshow(v.get_image()[:, :, ::-1])
