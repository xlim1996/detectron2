import os
import shutil
from tqdm import tqdm
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import numpy as np
import skimage.io as io
import cv2
import json

seg_image = Image.open('000000.seg.png')
img = cv2.imread('000000.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
json_file_path = '000000.json'

np.savetxt('seg_image.txt', np.array(seg_image))
seg_image = np.array(seg_image)
print(seg_image.shape)
json_file=json.load(open(json_file_path))
print(json_file)
objects = json_file['objects']
print(len(objects))
print(objects)
for i in range(len(objects)):
    bbox = objects[i]['bounding_box']
    x = bbox['top_left'][1]
    y = bbox['top_left'][0]
    w = bbox['bottom_right'][1] - x
    h = bbox['bottom_right'][0] - y
    print(x, y, w, h)
    class_name = objects[0]['class']

    draw_1=cv2.rectangle(img, (x,y),(x+w,y+h), (0,255,0), 2)
plt.imshow(img)
plt.show()
print(bbox)
