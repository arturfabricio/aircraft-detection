import colorsys
import random
from PIL import Image
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


def generate(s, n, increment, image_path) -> list[BBOX]:
    scaling = 1.0/s+2
    bbox_final = []

    x_list, y_list, _ = get_referencepoints(image_path, s)

    for x in x_list:
        for y in y_list:

            for i in range(0, n, increment):
                bbox_final.append(BBOX(x - (i // 2),   y - (i // 2),   i,   i))
                bbox_final.append(BBOX(x - (i*2 // 2), y - (i // 2),   i*2, i))
                bbox_final.append(
                    BBOX(x - (i // 2),   y - (i*2 // 2), i,   i*2))

    return bbox_final


def display(BBOXs: list[BBOX], image_path, s):
    im = Image.open(image_path)
    fig, ax = plt.subplots()
    ax.imshow(im)

    x_list, y_list, _ = get_referencepoints(image_path, s)
    ax.scatter(x_list, y_list, s=10, c='b')

    for i in range(len(BBOXs)):
        rect = patches.Rectangle((BBOXs[i].arr[0], BBOXs[i].arr[1]), BBOXs[i].arr[2]-BBOXs[i].arr[0],
                                 BBOXs[i].arr[3]-BBOXs[i].arr[1], linewidth=0.1, edgecolor=get_random_color(), facecolor='none')
        ax.add_patch(rect)

    plt.show()


def get_referencepoints(img_path, s):
    im = cv2.imread(str(img_path))
    width, height, *_ = im.shape
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
