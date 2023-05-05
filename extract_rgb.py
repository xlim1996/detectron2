import numpy as np
import matplotlib.pyplot as plt
import os

data_path_1='real_image_depth_maps/depth_maps_150622'
data_path_2='real_image_depth_maps/depth_maps_200622'
rgb_save_path='rgb'
npy_listdir = os.listdir(data_path_1)
index = 0
for npy_idx in npy_listdir:
    npy_path = os.path.join(data_path_1, npy_idx)
    npy_file = np.load(npy_path, allow_pickle=True).item()

    curent_image = npy_file['image']
    
    plt.imsave(rgb_save_path+'{}.png'.format(index), curent_image)
    index += 1

    # plt.imshow(curent_image)
    # plt.show()




