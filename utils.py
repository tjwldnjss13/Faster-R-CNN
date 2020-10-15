import torch
import numpy as np


def crop(in_, out_size):
    in_size = (in_.shape[1], in_.shape[2])
    drop_size = (in_size[0] - out_size[0], in_size[1] - out_size[1])
    drop_size = (drop_size[0] // 2, drop_size[1] // 2)
    out_ = in_[:, drop_size[0]:drop_size[0] + out_size[0], drop_size[1]:drop_size[1] + out_size[1]]

    return out_


def pad_4dim(x, ref, cuda=True):
    zeros = torch.zeros(x.shape[0], x.shape[1], 1, x.shape[3])
    if cuda:
        zeros = zeros.cuda()
    while x.shape[2] < ref.shape[2]:
        x = torch.cat([x, zeros], dim=2)
    zeros = torch.zeros(x.shape[0], x.shape[1], x.shape[2], 1)
    if cuda:
        zeros = zeros.cuda()
    while x.shape[3] < ref.shape[3]:
        x = torch.cat([x, zeros], dim=3)

    return x


def pad_3dim(x, ref_size):
    zeros = torch.zeros(x.shape[0], 1, x.shape[2]).cuda()
    while x.shape[1] < ref_size[0]:
        x = torch.cat([x, zeros], dim=1)
    zeros = torch.zeros(x.shape[0], x.shape[1], 1).cuda()
    while x.shape[2] < ref_size[1]:
        x = torch.cat([x, zeros], dim=2)

    return x


def pad_2dim(x, ref_size):
    zeros = torch.zeros(1, x.shape[1], dtype=torch.long).cuda()
    while x.shape[0] < ref_size[0]:
        x = torch.cat([x, zeros], dim=0)
    zeros = torch.zeros(x.shape[0], 1, dtype=torch.long).cuda()
    while x.shape[1] < ref_size[1]:
        x = torch.cat([x, zeros], dim=1)

    return x


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


def calculate_iou(box1, box2):
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    y1 = np.maximum(box1[0], box2[0])
    x1 = np.maximum(box1[1], box2[1])
    y2 = np.minimum(box1[2], box2[2])
    x2 = np.minimum(box1[3], box2[3])

    iou = 0
    if y1 < y2 and x1 < x2:
        inter = (y2 - y1) * (x2 - x1)
        union = area1 + area2 - inter
        iou = inter / union

    return iou


def mean_iou_segmentation(output, predict):
    a, b = (output[:, 1, :, :] > 0), (predict > 0)

    a_area = len(a.nonzero())
    b_area = len(b.nonzero())
    union = a_area + b_area
    inter = len((a & b).nonzero())
    iou = inter / (union - inter)

    return iou


def nms(boxes, ground_truth, threshold):
    n_gt = len(ground_truth)
    ious_gt = calculate_ious(boxes, ground_truth)

    max_iou_gt_args = [np.argmax(ious_gt[:, i]) for i in range(n_gt)]

    for i in range(n_gt):
        max_iou_gt_arg = max_iou_gt_args[i]
        max_iou_gt_box = boxes[max_iou_gt_arg]
        for j in range(len(boxes)):
            iou = calculate_iou(max_iou_gt_box, boxes[j])
            if iou <




def time_calculator(sec):
    if sec < 60:
        return 0, 0, sec
    if sec < 3600:
        M = sec // 60
        S = sec % M
        return 0, M, S
    H = sec // 3600
    sec = sec % 3600
    M = sec // 60
    S = sec % 60
    return int(H), int(M), S


if __name__ == '__main__':

    from rpn import anchor_box_generator
    import cv2 as cv
    import matplotlib.pyplot as plt
    import copy

    ratios = [.5, 1, 2]
    scales = [128, 256, 512]
    anchor_boxes = anchor_box_generator(ratios, scales, (600, 1000), 16)

    img_pth = 'samples/dogs.jpg'
    img = cv.imread(img_pth)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img_h_og, img_w_og, _ = img.shape
    img = cv.resize(img, (1000, 600))

    bbox = np.array([[120, 70, 570, 280], [220, 270, 580, 450], [30, 440, 570, 700]])
    bbox[:, 0] = bbox[:, 0] * (600 / img_h_og)
    bbox[:, 1] = bbox[:, 1] * (1000 / img_w_og)
    bbox[:, 2] = bbox[:, 2] * (600 / img_h_og)
    bbox[:, 3] = bbox[:, 3] * (1000 / img_w_og)

    img_copy = copy.deepcopy(img)

    for i, box in enumerate(anchor_boxes):
        y1, x1, y2, x2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        cv.rectangle(img_copy, (x1, y1), (x2, y2), (0, 0, 255), 3)

    # plt.figure(figsize=(15, 9))
    # plt.imshow(img_copy)
    # plt.show()

    # rpn = RPN(512, 256, (60, 40), 9).cuda()
    # from torchsummary import summary
    # summary(rpn, (512, 60, 40))

    nms(anchor_boxes, bbox, .7)

