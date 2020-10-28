import numpy as np


def calculate_iou(box1, box2):
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    y1 = np.maximum(box1[0], box2[0])
    x1 = np.maximum(box1[1], box2[1])
    y2 = np.minimum(box1[2], box2[2])
    x2 = np.minimum(box1[3], box2[3])

    print(y1, y2)
    print(x1, x2)

    iou = 0
    if y1 < y2 and x1 < x2:
        inter = (y2 - y1) * (x2 - x1)
        union = area1 + area2 - inter
        iou = inter / union

    return iou


def calculate_ious(box1, box2):
    ious = np.zeros((box1.shape[0], box2.shape[0]), dtype=np.float32)

    for i_1 in range(len(box1)):
        b1 = box1[i_1]
        b1_area = (b1[2] - b1[0]) * (b1[3] - b1[1])
        for i_2 in range(len(box2)):
            b2 = box2[i_2]
            b2_area = (b2[2] - b2[0]) * (b2[3] - b2[1])

            inter_y1 = max(b1[0], b2[0])
            inter_x1 = max(b1[1], b2[1])
            inter_y2 = min(b1[2], b1[2])
            inter_x2 = min(b1[3], b1[3])

            if inter_y1 < inter_y2 and inter_x1 < inter_x2:
                inter_area = (inter_y2 - inter_y1) * (inter_x2 - inter_x1)
                union_area = b1_area + b2_area - inter_area
                iou = inter_area / union_area
            else:
                iou = 0

            ious[i_1, i_2] = iou

    return ious


a_ = np.array([150, 241, 112, 63])
b_ = np.array([160, 251, 90, 70])

ay1 = int(a_[0] - .5 * a_[2])
ax1 = int(a_[1] - .5 * a_[3])
ay2 = int(ay1 + a_[2])
ax2 = int(ax1 + a_[3])

by1 = int(b_[0] - .5 * b_[2])
bx1 = int(b_[1] - .5 * b_[3])
by2 = int(by1 + b_[2])
bx2 = int(bx1 + b_[3])

a = np.array([ay1, ax1, ay2, ax2])
b = np.array([by1, bx1, by2, bx2])

aa_ = np.array([a_])
bb_ = np.array([b_])

iou = calculate_iou(a, b)
print(iou)

import matplotlib.pyplot as plt
import cv2 as cv

img = cv.imread('samples/dogs.jpg')
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
cv.rectangle(img, (a[1], a[0]), (a[3], a[2]), (0, 0, 255), 3)
cv.rectangle(img, (b[1], b[0]), (b[3], b[2]), (0, 255, 0), 3)
plt.imshow(img)
plt.show()
