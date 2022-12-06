import torch.nn as nn
import torch
import numpy as np
import model

def loss_fn(output, target):
    return nn.MSELoss()(output, target)
    # return nn.BCELoss()(output, target)
    #return nn.CrossEntropyLoss()(output, target)
    scale = 100
    # print('newly loaded 3')
    # print('output', output)
    # print('target', target)
    diff = torch.sub(target, output)
    # print('difference ', diff)
    scaled = torch.mul(diff, scale)
    # print('Scaled ', scaled)
    squared = torch.square(scaled)
    # print('squared ', squared)
    sum_loss = torch.sum(squared)
    # print('sum_loss', sum_loss)
    loss = torch.div(sum_loss, torch.numel(target))
    # print('loss', loss)
    plane_count = torch.sum(target)

    return loss
    # print('plane_count', plane_count)
    if plane_count == 0:
        return loss
    else:
        loss = torch.divide(loss, plane_count)
        #print('after divide', loss)
        return loss


def calculate_iou(bboxs: list[list[float]]):
    target_vector = np.zeros(len(model.np_bboxs))
    for bbox in bboxs:
        xA = np.maximum(model.np_bboxs[:, 0], bbox[0])
        yA = np.maximum(model.np_bboxs[:, 1], bbox[1])
        xB = np.minimum(model.np_bboxs[:, 2], bbox[2])
        yB = np.minimum(model.np_bboxs[:, 3], bbox[3])
        interArea = np.maximum(0, xB - xA + 1) * np.maximum(0, yB - yA + 1)
        boxAArea = (model.np_bboxs[:, 2] - model.np_bboxs[:, 0] + 1) * \
            (model.np_bboxs[:, 3] - model.np_bboxs[:, 1] + 1)
        boxBArea = (bbox[2] - bbox[0] + 1) * \
            (bbox[3] - bbox[1] + 1)
        iou = np.divide(interArea, (np.subtract(
            np.add(boxAArea, boxBArea), interArea)))
        b = np.zeros_like(iou)

        arg_best_match = np.argmax(iou, axis=0)
        if iou[arg_best_match] != 0:
            b[arg_best_match] = 1
        target_vector += b

    return target_vector