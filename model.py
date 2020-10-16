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
from vgg import VGG
from utils import calculate_ious


class FasterRCNN(nn.Module):
    def __init__(self, in_size, num_classes, visualize=False):
        super(FasterRCNN, self).__init__()
        self.in_size = in_size
        self.num_classes = num_classes
        self.visualize = visualize
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # self.backbone = self.build_backbone()
        self.backbone = VGG('A', self.num_classes).to(self.device)
        self.rpn = RPN(512, 512, self.in_size, 9).to(self.device)

    def build_backbone(self):
        in_h, in_w = self.in_size[0], self.in_size[1]
        model = VGG('A', self.num_classes).to(self.device)
        # model = models.vgg16(pretrained=True).to(self.device)
        # features = list(model.features)
        #
        # dummy_img = torch.zeros((1, 3, in_h, in_w)).float()
        # req_features = []
        # dummy = dummy_img.clone().to(self.device)
        #
        # for feature in features:
        #     dummy = feature(dummy)
        #
        #     if dummy.size()[2] < 800 // 16:
        #         break
        #     req_features.append(feature)
        #     out_dim = dummy.size()[1]

        # return nn.Sequential(*req_features)
        return model

    def build_detector(self, in_, anchor_box, roi, gt):
        ious = calculate_ious(anchor_box, gt)
        pass

    def resize_img_bbox(self, img, bbox, img_to_input=True):
        if img_to_input:
            in_h, in_w = self.in_size[0], self.in_size[1]
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            img_og = img
            img = cv.resize(img, dsize=(in_w, in_h), interpolation=cv.INTER_CUBIC)

            ratio_h = in_h / img_og.shape[0]
            ratio_w = in_w / img_og.shape[1]
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
        pass



if __name__ == '__main__':
    # bbox = np.array([[120, 70, 570, 280], [220, 270, 580, 450], [30, 440, 570, 700]])

    model = FasterRCNN((600, 1000), 10)



