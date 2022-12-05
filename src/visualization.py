import numpy as np
import matplotlib.pyplot as plt
from data_aug import *
from bbox_aug import *


def display_bboxs(img: np.array, bboxs: np.array):
    plotted_img = draw_rect(img, bboxs)
    plt.imshow(plotted_img)
    plt.show()


def display_bbox_target_vector(img, target, np_bboxs, threshhold):
    to_draw = np_bboxs[target >= threshhold]
    display_bboxs(img, to_draw)
