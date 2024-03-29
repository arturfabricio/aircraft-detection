import colorsys
import random
import cv2
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
import skimage.io
from typing import List
#import model

IMAGE_SIZE = (256,256)

class BBOX():
    def __init__(self, min_x: int, min_y: int, width: int, height: int):
        self.arr = np.array([np.clip(min_x, 0, IMAGE_SIZE[0]), np.clip(min_y, 0, IMAGE_SIZE[1]), np.clip(
            min_x+width, 0, IMAGE_SIZE[0]), np.clip(min_y+height, 0, IMAGE_SIZE[1])])

    def __str__(self):
        return "BBOX: % s" % (self.arr)


def contains(list, filter):
    for x in list:
        if filter(x):
            return True
    return False


def generate(s, bbox_count, max_size, im_shape) -> List[BBOX]:
    scaling = 1.0/s+2

    bbox_final = []
    x_list, y_list, _ = get_referencepoints(im_shape, s)

    for idx in range(len(x_list)):
        ref_x, ref_y = x_list[idx], y_list[idx]

        for i in list(np.linspace(max_size, 0, num=bbox_count, endpoint=False)):
            bbox_final.append(
                BBOX(ref_x - (i // 2),   ref_y - (i // 2),   i,   i))
            bbox_final.append(
                BBOX(ref_x - (i*2 // 2), ref_y - (i // 2),   i*2, i))
            bbox_final.append(
                BBOX(ref_x - (i // 2),   ref_y - (i*2 // 2), i,   i*2))

    return bbox_final


def display(BBOXs: List[BBOX], image_path, s, new_size, ratio):
    im = transformsXY_im(image_path, new_size)
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

    width = IMAGE_SIZE[0]
    height = IMAGE_SIZE[1]

    # print('Width: ', width)
    # print('Heigth: ', height)

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


def transformsImg(path, new_size):
    x = cv2.imread(str(path))
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    x = cv2.resize(x, (new_size, new_size))

    return x


def transformsBbox(bboxs, ratio):
    return list(map(lambda bbox: [int(bbox[0]/ratio), int(bbox[1]/ratio), int(bbox[0]/ratio)+int(bbox[2]/ratio), int(bbox[1]/ratio)+int(bbox[3]/ratio)], bboxs))

# def rotationImg(img):
