import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple
from torch import Tensor, tensor
import cv2
import matplotlib.patches as patches

# functions to show an image


def show_image(img: Tensor, bboxs: np.ndarray, normalization_used: Tuple[Tensor, Tensor] = (tensor([0, 0, 0]), tensor([1, 1, 1]))):
    img = img.clone()
    img[0] = img[0] * normalization_used[1][0] + normalization_used[0][0]
    img[1] = img[1] * normalization_used[1][1] + normalization_used[0][1]
    img[2] = img[2] * normalization_used[1][2] + normalization_used[0][2]
    npimg = img.numpy()

    fig, ax = plt.subplots()
    ax.imshow(np.transpose(npimg, (1, 2, 0)))
    for bbox in bboxs:
        start_point = (int(bbox[0]), int(bbox[1]))
        width, height = int(bbox[2]) - int(bbox[0]), int(bbox[3]) - int(bbox[1])
        rect = patches.Rectangle(start_point, width, height, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    plt.show()

# def show_bboxs(img: Tensor, bboxs: Tensor):
