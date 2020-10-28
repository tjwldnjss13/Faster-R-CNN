import numpy as np
import torch
import torch.nn as nn

from utils import calculate_ious

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class RPN(nn.Module):
    def __init__(self, in_dim, out_dim, in_size, n_anchor):
        super(RPN, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.in_size = in_size
        self.conv = nn.Conv2d(in_dim, out_dim, 3, 1, 1)
        self.conv.weight.data.normal_(0, .01)
        self.conv.bias.data.zero_()
        self.reg_layer = nn.Conv2d(out_dim, n_anchor * 4, 1, 1, 0)
        # self.reg_layer = nn.Linear(out_dim * self.in_size[0] * in_size[1], n_anchor * 4)
        self.reg_layer.weight.data.normal_(0, .01)
        self.reg_layer.bias.data.zero_()
        self.cls_layer = nn.Conv2d(out_dim, n_anchor * 2, 1, 1, 0)
        # self.cls_layer = nn.Linear(out_dim * self.in_size[0] * self.in_size[1], n_anchor * 2)
        self.cls_layer.weight.data.normal_(0, .01)
        self.cls_layer.bias.data.zero_()
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        x = self.conv(x)

        reg = self.reg_layer(x)
        cls = self.cls_layer(x)

        reg = reg.permute(0, 2, 3, 1).contiguous().view(reg.size(0), -1, 4)
        cls = cls.permute(0, 2, 3, 1).contiguous().view(cls.size(0), -1, 2)
        cls = self.softmax(cls)

        return reg, cls


def anchor_generator(feature_size, anchor_stride):
    feat_h, feat_w = feature_size[0], feature_size[1]
    ctr_x = np.arange(anchor_stride, (feat_w + 1) * anchor_stride, anchor_stride)
    ctr_y = np.arange(anchor_stride, (feat_h + 1) * anchor_stride, anchor_stride)

    anchors = np.zeros((len(ctr_x) * len(ctr_y), 2))

    c_i = 0
    for x in ctr_x:
        for y in ctr_y:
            anchors[c_i, 1] = x - feat_w // 2
            anchors[c_i, 0] = y - feat_h // 2
            c_i += 1

    print('Anchors generated')

    return anchors


def anchor_box_generator(ratios, scales, input_size, anchor_stride):
    # 50 = 800 // 16

    # ratios = [.5, 1, 2]
    # scales = [128, 256, 512]

    in_h, in_w = input_size[0], input_size[1]

    feat_h, feat_w = in_h // anchor_stride, in_w // anchor_stride
    anchors = anchor_generator((feat_h, feat_w), anchor_stride)
    anchor_boxes = np.zeros((len(anchors) * len(ratios) * len(scales), 4))

    anc_i = 0
    for anc in anchors:
        anc_y, anc_x = anc
        for r in ratios:
            for s in scales:
                # h = anchor_stride * s * np.sqrt(r)
                # w = anchor_stride * s * np.sqrt(1. / r)
                if r < 1:
                    h, w = s, s * (1. / r)
                elif r > 1:
                    h, w = s * (1. / r), s
                else:
                    h, w = s, s

                anchor_boxes[anc_i, 0] = anc_y - .5 * h
                anchor_boxes[anc_i, 1] = anc_x - .5 * w
                anchor_boxes[anc_i, 2] = anc_y + .5 * h
                anchor_boxes[anc_i, 3] = anc_x + .5 * w

                anc_i += 1

    # idx_valid = np.where((anchor_boxes[:, 0] >= 0) &
    #                      (anchor_boxes[:, 1] >= 0) &
    #                      (anchor_boxes[:, 2] <= in_h) &
    #                      (anchor_boxes[:, 3] <= in_w))[0]
    # anchor_boxes = anchor_boxes[idx_valid]

    print('Anchor boxes generated')

    return anchor_boxes


def anchor_boxes_generator_categorical(anchor_boxes, ground_truth):
    n_gt = ground_truth.shape[0]
    ious_anc_gt = calculate_ious(anchor_boxes, ground_truth)
    argmax_iou_anc_gt = np.argmax(ious_anc_gt, axis=1)

    anchor_boxes_cat = [[] for _ in range(n_gt)]
    for i, arg in enumerate(argmax_iou_anc_gt):
        anchor_boxes_cat[arg].append(anchor_boxes[i])

    # anchor_gts = ground_truth[argmax_iou_anc_gt]

    for i in range(n_gt):
        anchor_boxes_cat[i] = np.array(anchor_boxes_cat[i])
    # anchor_boxes_cat = np.array(anchor_boxes_cat)

    print('Categorical anchor boxes generated')

    return anchor_boxes_cat


def anchor_label_generator(anchor_boxes, ground_truth, pos_threshold, neg_threshold):
    ious_anc_gt = calculate_ious(anchor_boxes, ground_truth)

    pos_args_ious_anc_gt_1 = np.argmax(ious_anc_gt, axis=0)
    pos_args_ious_anc_gt_2 = np.where(ious_anc_gt >= pos_threshold)[0]
    pos_args_ious_anc_gt = np.append(pos_args_ious_anc_gt_1, pos_args_ious_anc_gt_2)
    pos_args_ious_anc_gt = np.array(list(set(pos_args_ious_anc_gt)))

    # anchor_labels = np.zeros(anchors.shape[0])
    anchor_labels = np.array([-1 for _ in range(anchor_boxes.shape[0])])
    anchor_labels[pos_args_ious_anc_gt] = 1

    non_pos_args_labels = np.where(anchor_labels != 1)[0]
    for i in non_pos_args_labels:
        neg_f = False
        for j in range(len(ground_truth)):
            if ious_anc_gt[i, j] >= neg_threshold:
                break
            neg_f = True
        if neg_f:
            anchor_labels[i] = 0

    # neg_args_ious_anc_gt = np.where(anchor_labels == -1)[0]

    print('Anchor labels generated')

    return anchor_labels


def anchor_label_generatgor_2dim(anchor_labels):
    anchor_labels2 = np.zeros((anchor_labels.shape[0], 2))
    train_args = np.where(anchor_labels != -1)
    anchor_labels2[train_args, anchor_labels[train_args]] = 1

    print('2-dim anchor labels generated')

    return anchor_labels2


def anchor_ground_truth_generator(anchor_boxes, ground_truth):
    ious_anc_gt = calculate_ious(anchor_boxes, ground_truth)
    argmax_iou_anc_gt = np.argmax(ious_anc_gt, axis=1)

    anchor_gts = ground_truth[argmax_iou_anc_gt]

    print('Anchor ground truth generated')

    return anchor_gts


def loc_delta_generator(bbox, anchor_box):
    assert bbox.shape == anchor_box.shape

    bbox_h, bbox_w = bbox[:, 2] - bbox[:, 0], bbox[:, 3] - bbox[:, 1]
    bbox_cy, bbox_cx = bbox[:, 0] + .5 * bbox_h, bbox[:, 1] + .5 * bbox_w

    anc_h, anc_w = anchor_box[:, 2] - anchor_box[:, 0], anchor_box[:, 3] - anchor_box[:, 1]
    anc_cy, anc_cx = anchor_box[:, 0] + .5 * anc_h, anchor_box[:, 1] + .5 * anc_w

    loc_h, loc_w = torch.log(bbox_h / anc_h), torch.log(bbox_w / anc_w)
    loc_cy, loc_cx = (bbox_cy - anc_cy) / anc_h, (bbox_cx - anc_cx) / anc_w

    locs = np.zeros(bbox.shape)
    locs[:, 0], locs[:, 1], locs[:, 2], locs[:, 3] = loc_cy, loc_cx, loc_h, loc_w

    return locs


# def loc_delta_generator(predicted, target):
#     pred = predicted
#     tar = target
#
#     h_pred = pred[:, 2] - pred[:, 0]
#     w_pred = pred[:, 3] - pred[:, 1]
#     cy_pred = pred[:, 0] + .5 + h_pred
#     cx_pred = pred[:, 1] + .5 * w_pred
#
#     h_tar = tar[:, 2] - tar[:, 0]
#     w_tar = tar[:, 3] - tar[:, 1]
#     cy_tar = tar[:, 0] + .5 * h_tar
#     cx_tar = tar[:, 1] + .5 * w_tar
#
#     # eps = np.finfo(h_pred.dtype).eps
#
#     h_pred = np.maximum(0., h_pred)
#     w_pred = np.maximum(0., w_pred)
#
#     dy = (cy_tar - cy_pred) / h_pred
#     dx = (cx_tar - cx_pred) / w_pred
#     dh = np.log(h_tar / h_pred)
#     dw = np.log(w_tar / w_pred)
#
#     loc_deltas = np.vstack((dy, dx, dh, dw)).transpose()
#
#     return loc_deltas


if __name__ == '__main__':
    import cv2 as cv
    import matplotlib.pyplot as plt
    import copy

    ratios = [.5, 1, 2]
    scales = [128, 256, 512]
    in_size = (600, 1000)
    anchor_boxes = anchor_box_generator(ratios, scales, in_size, 16)

    img_pth = 'samples/dogs.jpg'
    img = cv.imread(img_pth)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img_h_og, img_w_og, _ = img.shape
    img = cv.resize(img, (in_size[1], in_size[0]))

    bbox = np.array([[120, 70, 570, 280], [220, 270, 580, 450], [30, 440, 570, 700]])
    bbox[:, 0] = bbox[:, 0] * (in_size[0] / img_h_og)
    bbox[:, 1] = bbox[:, 1] * (in_size[1] / img_w_og)
    bbox[:, 2] = bbox[:, 2] * (in_size[0] / img_h_og)
    bbox[:, 3] = bbox[:, 3] * (in_size[1] / img_w_og)

    img_copy = copy.deepcopy(img)

    for i, box in enumerate(anchor_boxes):
        y1, x1, y2, x2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        cv.rectangle(img_copy, (x1, y1), (x2, y2), (0, 0, 255), 3)

    for i, gt in enumerate(bbox):
        y1, x1, y2, x2 = int(gt[0]), int(gt[1]), int(gt[2]), int(gt[3])
        cv.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 3)

    plt.figure(figsize=(10, 6))
    plt.imshow(img_copy)
    plt.show()