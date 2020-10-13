import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import copy
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from PIL import Image

from rpn import RPN

class FasterRCNN(nn.Module):
    def __init__(self, in_size, visualize=False):
        super(FasterRCNN, self).__init__()
        self.in_size = in_size
        self.visualize = visualize
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.backbone = self.build_backbone(self.in_size)
        self.rpn = RPN(512, 512, 9)

    def build_backbone(self, in_size):
        model = models.vgg16(pretrained=True).to(self.device)
        features = list(model.features)

        dummy_img = torch.zeros((1, 3, in_size, in_size)).float()
        req_features = []
        dummy = dummy_img.clone().to(self.device)

        for feature in features:
            dummy = feature(dummy)

            if dummy.size()[2] < 800 // 16:
                break
            req_features.append(feature)
            out_dim = dummy.size()[1]

        return nn.Sequential(*req_features)

    def build_detector(self, in_, anchor_box, roi, gt):
        ious = iou(anchor_box, gt)

    def resize_img_bbox(self, img, bbox, img_to_input=True):
        if img_to_input:
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            img_og = img
            img = cv.resize(img, dsize=(800, 800), interpolation=cv.INTER_CUBIC)

            ratio_h = self.in_size / img_og.shape[0]
            ratio_w = self.in_size / img_og.shape[1]
            ratio_hw = [ratio_h, ratio_w, ratio_h, ratio_w]

            bbox_warp = []
            for box in bbox:
                box = [int(a * b) for a, b in zip(box, ratio_hw)]
                bbox_warp.append(box)
            bbox = np.array(bbox_warp)

            if self.visualize:
                img_clone = copy.deepcopy(img)
                for i in range(len(bbox)):
                    cv.rectangle(img_clone, (bbox[i][1], bbox[i][0]), (bbox[i][3], bbox[i][2]), color=(0, 255, 0), thickness=5)
                title = 'Bounding box'
                plt.figure(title)
                plt.title(title)
                plt.imshow(img_clone)
                plt.show()

        return img, bbox

    def feature_map_extractor(self, img, bbox):
        img, bbox = self.resize_img_bbox(img, bbox)

        transform = transforms.Compose([transforms.ToTensor()])
        img_tensor = transform(img).to(self.device)
        img_tensor = img_tensor.unsqueeze(0)

        feature_maps = self.backbone(img_tensor)

        if self.visualize:
            img_arr = feature_maps.detach().cpu().numpy().squeeze(0)
            fig = plt.figure(figsize=(12, 4))
            num_fig = 1

            for i in range(5):
                fig.add_subplot(1, 5, num_fig)
                plt.imshow(img_arr[i], cmap='gray')
                num_fig += 1
            plt.show()

        return feature_maps

    def forward(self, in_, gt, label):
        feature_map = self.backbone(in_)
        reg_pred, cls_pred = self.rpn(feature_map)


def anchor_generator(feature_size, anchor_stride):
    ctr_x = np.arange(anchor_stride, (feature_size + 1) * anchor_stride, anchor_stride)
    ctr_y = np.arange(anchor_stride, (feature_size + 1) * anchor_stride, anchor_stride)

    ctr = np.zeros((len(ctr_x) * len(ctr_y), 2))

    c_i = 0
    for x in ctr_x:
        for y in ctr_y:
            ctr[c_i, 1] = x - 8
            ctr[c_i, 0] = y - 8
            c_i += 1

    return ctr


def anchor_box_generator(ratios, scales, anchors, input_size, anchor_stride):
    # 50 = 800 // 16

    # ratios = [.5, 1, 2]
    # scales = [8, 16, 32]

    feature_size = input_size // anchor_stride
    anchors = anchor_generator(feature_size)
    anchor_boxes = np.zeros((feature_size * feature_size * len(ratios) * len(scales), 4))

    anc_i = 0
    for anc in anchors:
        anc_y, anc_x = anc
        for r in ratios:
            for s in scales:
                h = anchor_stride * s * np.sqrt(r)
                w = anchor_stride * np.sqrt(1. / r)

                anchor_boxes[anc_i, 0] = anc_y - .5 * h
                anchor_boxes[anc_i, 1] = anc_x - .5 * w
                anchor_boxes[anc_i, 2] = anc_y + .5 * h
                anchor_boxes[anc_i, 3] = anc_x + .5 * w

                anc_i += 1

    idx_valid = np.where((anchor_boxes[:, 0] >= 0) &
                         (anchor_boxes[:, 1] >= 0) &
                         (anchor_boxes[:, 2] <= 800) &
                         (anchor_boxes[:, 3] <= 800))[0]
    anchor_boxes = anchor_boxes[idx_valid]

    return anchor_boxes


def iou(box1, box2):
    ious = np.zeros((len(box1), len(box2)), dtype=np.float32)

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


def loc_delta_generator(predicted, target):
    pred = predicted
    tar = target

    h_pred = pred[:, 2] - pred[:, 0]
    w_pred = pred[:, 3] - pred[:, 1]
    cy_pred = pred[:, 0] + .5 + h_pred
    cx_pred = pred[:, 1] + .5 * w_pred

    h_tar = tar[:, 2] - tar[:, 0]
    w_tar = tar[:, 3] - tar[:, 1]
    cy_tar = tar[:, 0] + .5 * h_tar
    cx_tar = tar[:, 1] + .5 * w_tar

    eps = np.finfo(h_pred.dtype).eps

    h_pred = np.maximum(0., h_pred)
    w_pred = np.maximum(0., w_pred)

    dy = (cy_tar - cy_pred) / h_pred
    dx = (cx_tar - cx_pred) / w_pred
    dh = np.log(h_tar / h_pred)
    dw = np.log(w_tar / w_pred)

    loc_deltas = np.vstack((dy, dx, dh, dw)).transpose()

    return loc_deltas


if __name__ == '__main__':
    frcnn = FasterRCNN(800, False)

    img = cv.imread('dogs.jpg')
    bbox = np.array([[120, 70, 570, 280], [220, 270, 580, 450], [30, 440, 570, 700]])

    feature_maps = frcnn.feature_map_extractor(img, bbox)
    frcnn.anchor_generator(feature_maps.shape[2])



