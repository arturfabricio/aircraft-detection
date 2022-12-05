import numpy as np
import matplotlib.pyplot as plt
from data_aug import *
from bbox_aug import *


def plot(img: np.array, bboxs: np.array):
    plotted_img = draw_rect(img, bboxs)
    plt.imshow(plotted_img)
    plt.show()
