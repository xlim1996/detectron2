'''
Author: Xiaolin Lin xlim1996@outlook.com
Date: 2023-03-24 11:55:06
LastEditors: Xiaolin Lin xlim1996@outlook.com
LastEditTime: 2023-05-05 10:51:47
FilePath: /hiwi/test_detectron2.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import time
import torch
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mplc
import matplotlib as mpl
from PIL import Image
import open3d as o3d

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from detectron2.engine.defaults import create_ddp_model
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer,GenericMask
from detectron2.structures.instances import Instances
# from detectron2.utils.visualizer import draw_polygon
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
print("id",id)
color_map = dict()
#create a color map
for i in range(len(id)):
    color = np.random.rand(3)
    color = mplc.to_rgb(color)
    vec = np.random.rand(3)
    # better to do it in another color space
    vec = vec / np.linalg.norm(vec) * 0.5
    res = np.clip(vec + color, 0, 1)
    color_map[id[i]]= res

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

def draw_polygon(img, segment, color, edge_color=None, alpha=0.5):
    """
    Args:
        img (numpy array): input image.
        segment (numpy array of shape Nx2): containing all the points in the polygon.
        color (str or tuple): color of the polygon. Refer to `matplotlib.colors` for a full list of
            formats that are accepted.
        edge_color (str or tuple): color of the polygon edges. Refer to `matplotlib.colors` for a
            full list of formats that are accepted. If not provided, a darker shade
            of the polygon color will be used instead.
        alpha (float): blending efficiency. Smaller values lead to more transparent masks.

    Returns:
        output (numpy array): output image with polygon drawn.
    """
    if edge_color is None:
        # make edge color darker than the polygon color
        if alpha > 0.8:
            edge_color = mplc.to_rgb(color) + (-0.7,)
        else:
            edge_color = mplc.to_rgb(color)
    edge_color = mplc.to_rgb(edge_color) + (1,)

    polygon = mpl.patches.Polygon(
        segment,
        fill=True,
        facecolor=mplc.to_rgb(color) + (alpha,),
        edgecolor=edge_color,
        linewidth=max(1, img.shape[0] // 100),
    )
    fig, ax = mpl.pyplot.subplots()
    ax.imshow(img)
    ax.add_patch(polygon)
    ax.axis('off')
    fig.canvas.draw()
    output = np.array(fig.canvas.renderer.buffer_rgba())

    mpl.pyplot.close(fig)
    return output

def draw_bbox(image,bboxs):
    for bbox in bboxs:
        cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0,255,0), 2)
    return image

# def draw_mask(image,mask,label):
#     for i in range(len(label)):
#         for segment in mask[i].polygons:
#             print("segment",segment.shape)
#             segment = np.array(segment).reshape(-1, 2)
#             image=draw_polygon(image, segment, color_map[label])
#     return image

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
        time_end = time.time()
        print('time cost', time_end - time_start, 's')
        labels = result.pred_classes.numpy()
        bboxes = result.pred_boxes.tensor.numpy()
        masks = result.pred_masks
        masks = masks.numpy().astype(np.uint8)
        H, W = image.shape[:2]
        masks = [GenericMask(x, H, W) for x in masks]
        # labels_np = labels
        # bboxes_np = bboxes
        # masks_np = masks
        v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(cfg.dataloader.train.dataset.names), scale=1.2)
        # v = Visualizer(image_show[:, :, ::-1], metadata, scale=1.2)
        v = v.draw_instance_predictions(result.to("cpu"))
        H, W = image.shape[:2]
        output = v.get_image()[:, :, ::-1]
        cv2_imshow(v.get_image()[:, :, ::-1])
        #resize image back to original size
        image = cv2.resize(output, (W, H))
        plt.imshow(image)
        plt.show()
        # image = draw_mask(image,result.pred_masks.numpy().astype(np.int32),
        #                   result.pred_classes.numpy().astype(np.int32),
        #                   colors=color_map, alpha=0.8)
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
        # Map each pixel in the mask to the corresponding point in the point cloud
        # # seg_pcd= np.asarray(pcd.points)[obj_mask == True, :]
        # for i in range(len(labels_np)):
        #     # 获取检测结果和分割结果
        #     obj_label = labels_np[i]
        #     obj_bbox = bboxes_np[i]
        #     obj_mask = masks_np[i]

           
            
        #     # Crop the point cloud based on the mask
        #     # obj_pcd = pcd.crop(np.logical_not(obj_mask))
        #     obj_pcd = pcd.crop(
        #         o3d.geometry.AxisAlignedBoundingBox(obj_bbox)
        #     )
        #     # 为对象的点云赋予颜色
        #     obj_color = np.zeros((len(obj_pcd.points), 3))
        #     obj_color[obj_mask, :] = color_map[obj_label]
        #     obj_pcd.colors = o3d.utility.Vector3dVector(obj_color)

        #     # 添加对象的点云和 label 到完整的点云中
        #     obj_pcd.paint_uniform_color(np.array([1, 1, 1]))
        #     obj_pcd.transform(pcd.get_rotation_matrix_from_xyz((np.pi / 2, 0, 0)))
        #     obj_pcd.translate((0, 0, i * 2))
        #     pcd += obj_pcd
        #     label_text = o3d.geometry.Text3D(str(obj_label), (0, 0, 0.5 + i * 2), 0.5)
        #     label_text.transform(pcd.get_rotation_matrix_from_xyz((np.pi / 2, 0, 0)))
        #     label_text.translate((0, 0, i * 2))
        #     pcd += label_text

        # visualize point cloud
        o3d.visualization.draw_geometries([pcd])

        #try add segmentation to pcl    

        break

