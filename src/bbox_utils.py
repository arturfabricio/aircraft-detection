import colorsys
import random
import cv2
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
import skimage.io

class BBOX():
    def __init__(self, min_x: int, min_y: int, width: int, height: int):
        self.arr = np.array([min_x, min_y, min_x+width, min_y+height])

    def __str__(self):
        return "BBOX: % s" % (self.arr)

def generate(s, n, increment, im_shape) -> list[BBOX]:
    scaling = 1.0/s+2
    bbox_final = []
    x_list, y_list, _ = get_referencepoints(im_shape, s)
    for x in x_list:
        for y in y_list:
            for i in range(0, n, increment):
                bbox_final.append(BBOX(x - (i // 2),   y - (i // 2),   i,   i))
                bbox_final.append(BBOX(x - (i*2 // 2), y - (i // 2),   i*2, i))
                bbox_final.append(
                    BBOX(x - (i // 2),   y - (i*2 // 2), i,   i*2))
    return bbox_final


def display(BBOXs: list[BBOX], image_path, s, new_size, ratio):
    im = transformsXY_im(image_path,new_size)
    fig, ax = plt.subplots()
    ax.imshow(im)
    x_list, y_list, _ = get_referencepoints(image_path, s)
    ax.scatter(x_list, y_list, s=10, c='b')
    for i in range(len(BBOXs)):
        rect = patches.Rectangle((BBOXs[i].arr[0], BBOXs[i].arr[1]), BBOXs[i].arr[2]-BBOXs[i].arr[0],
                                 BBOXs[i].arr[3]-BBOXs[i].arr[1], linewidth=0.1, edgecolor=get_random_color(), facecolor='none')
        ax.add_patch(rect)
    plt.show()


def get_referencepoints(im_shape, s):
    #im = cv2.imread(str(img_path))

    width = 128
    height = 128

    print('Width: ', width)
    print('Heigth: ', height)

    width_incr = int(width//s)
    height_incr = int(height//s)
    x = []
    y = []

    for i in range(height_incr//2, height, width_incr):
        for j in range(width_incr//2, width, height_incr):
            x.append(i)
            y.append(j)
    return x, y, s

def get_random_color():
    h, s, l = random.uniform(0, 1), 1, 0.5
    return colorsys.hls_to_rgb(h, l, s)


def transformsXY(path, bb, new_size, ratio):
    x = cv2.imread(str(path)).astype(np.float32)
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)/255
    x = cv2.resize(x,(new_size,new_size) ) 
    bb[0] = int(bb[0]/ratio)
    bb[1] = int(bb[1]/ratio)
    bb[2] = int(bb[2]/ratio)
    bb[3] = int(bb[3]/ratio)
    return x, bb 

def transformsXY_im(path, new_size):
    x = cv2.imread(str(path)).astype(np.float32)
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)/255
    x = cv2.resize(x,(new_size,new_size) ) 
    return x
