import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from PIL import Image

device = torch.device('cuda:0:' if torch.cuda.is_available() else 'cpu')

# Load image
imgO = cv.imread('dogs.jpg')
imgO = cv.cvtColor(imgO, cv.COLOR_BGR2RGB)

# print(imgO.shape)

# Set bounding boxes
bboxO = np.array([[120, 70, 570, 280], [220, 270, 580, 450], [30, 440, 570, 700]])
labels = np.array([1, 1, 1])

# Visualize bounding boxes
for i in range(len(bboxO)):
    cv.rectangle(imgO, (bboxO[i][1], bboxO[i][0]), (bboxO[i][3], bboxO[i][2]), color=(0, 255, 0), thickness=5)
# plt.imshow(imgO)
# plt.show()

# Resize image and bounding boxes
img = cv.resize(imgO, dsize=(800, 800), interpolation=cv.INTER_CUBIC)
Wratio = 800 / imgO.shape[1]
Hratio = 800 / imgO.shape[0]
ratioList = [Hratio, Wratio, Hratio, Wratio]
bbox = []
for box in bboxO:
    box = [int(a * b) for a, b in zip(box, ratioList)]
    bbox.append(box)
bbox = np.array(bbox)

# print(bbox)

# Visualize bounding boxes
for i in range(len(bbox)):
    cv.rectangle(img, (bbox[i][1], bbox[i][0]), (bbox[i][3], bbox[i][2]), color=(0, 255, 0), thickness=5)
# plt.imshow(img)
# plt.show()


# Load pretrained VGG16 model
model = torchvision.models.vgg16(pretrained=True).to(device)
fe = list(model.features)

dummy_img = torch.zeros((1, 3, 800, 800)).float()

# Make (512, 50, 50) features with VGG16
req_features = []
k = dummy_img.clone().to(device)
for feature in fe:
    k = feature(k)
    if k.size()[2] < 800 // 16:
        break
    req_features.append(feature)
    out_channels = k.size()[1]

faster_rcnn_fe_extractor = nn.Sequential(*req_features)

transform = transforms.Compose([transforms.ToTensor()])
imgTensor = transform(img).to(device)
imgTensor = imgTensor.unsqueeze(0)
out_map = faster_rcnn_fe_extractor(imgTensor)

# Visualize first 5 of (512, 50, 50) features
# imgArray = out_map.detach().cpu().numpy().squeeze(0)
# fig = plt.figure(figsize=(12, 4))
# figNo = 1
# for i in range(5):
#     fig.add_subplot(1, 5, figNo)
#     plt.imshow(imgArray[i], cmap='gray')
#     figNo += 1
# plt.show()

# Generate 2500 anchors
fe_size = 800 // 16
ctr_x = np.arange(16, (fe_size + 1) * 16, 16)
ctr_y = np.arange(16, (fe_size + 1) * 16, 16)

index = 0
ctr = np.zeros((2500, 2))
for x in range(len(ctr_x)):
    for y in range(len(ctr_y)):
        ctr[index, 1] = ctr_x[x] - 8
        ctr[index, 0] = ctr_y[y] - 8
        index += 1

# Visualize anchors
# img_clone = np.copy(img)
# plt.figure(figsize=(9, 6))
# for i in range(ctr.shape[0]):
#     cv.circle(img_clone, (int(ctr[i][1]), int(ctr[i][0])), radius=1, color=(255, 0, 0), thickness=2)
# plt.imshow(img_clone)
# plt.show()

# With 2500 anchors, generate 2500 * 9 anchor boxes
# 9 : len(ratios) * len(scales)
ratios = [.5, 1, 2]
scales = [8, 16, 32]
sub_sample = 16
anchor_boxes = np.zeros((fe_size * fe_size * 9, 4))
index = 0
for c in ctr:
    ctr_y, ctr_x = c
    for i in range(len(ratios)):
        for j in range(len(scales)):
            h = sub_sample * scales[j] * np.sqrt(ratios[i])
            w = sub_sample * scales[j] * np.sqrt(1. / ratios[i])
            anchor_boxes[index, 0] = ctr_y - h / 2.
            anchor_boxes[index, 1] = ctr_x - w / 2.
            anchor_boxes[index, 2] = ctr_y + h / 2.
            anchor_boxes[index, 3] = ctr_x + w / 2.
            index += 1

# Visualize 9 anchor boxes of one anchor
# img_clone = np.copy(img)
# for i in range(9 * 1225, 9 * 1225 + 9):
#     y0 = int(anchor_boxes[i][0])
#     x0 = int(anchor_boxes[i][1])
#     y1 = int(anchor_boxes[i][2])
#     x1 = int(anchor_boxes[i][3])
#     cv.rectangle(img_clone, (x0, y0), (x1, y1), color=(0, 0, 255), thickness=3)
#
# for i in range(len(bbox)):
#     cv.rectangle(img_clone, (bbox[i][1], bbox[i][0]), (bbox[i][3], bbox[i][2]), color=(0, 255, 0), thickness=3)

# plt.imshow(img_clone)
# plt.show()

# Take valid anchor boxes
index_inside = np.where(
    (anchor_boxes[:, 0] >= 0) &
    (anchor_boxes[:, 1] >= 0) &
    (anchor_boxes[:, 2] <= 800) &
    (anchor_boxes[:, 3] <= 800)
)[0]
valid_anchor_boxes = anchor_boxes[index_inside]

# Calculate IoU
# Each valid anchor box with bounding box ground truth
ious = np.empty((len(valid_anchor_boxes), len(bbox)), dtype=np.float32)
ious.fill(0)
for num1, i in enumerate(valid_anchor_boxes):
    ya1, xa1, ya2, xa2 = i
    anchor_area = (ya2 - ya1) * (xa2 - xa1)
    for num2, j in enumerate(bbox):
        yb1, xb1, yb2, xb2 = j
        box_area = (yb2 - yb1) * (xb2 - xb1)
        inter_y1 = max(ya1, yb1)
        inter_x1 = max(xa1, xb1)
        inter_y2 = min(ya2, yb2)
        inter_x2 = min(xa2, xb2)
        if inter_x1 < inter_x2 and inter_y1 < inter_y2:
            iter_area = (inter_y2 - inter_y1) * (inter_x2 - inter_x1)
            iou = iter_area / (anchor_area + box_area - iter_area)
        else:
            iou = 0
        ious[num1, num2] = iou

# 각 ground truth 별로 iou가 가장 놓은 anchor box의 index들
gt_argmax_ious = ious.argmax(axis=0)

# 각 ground truth 별로 가장 높은 iou들
gt_max_ious = ious[gt_argmax_ious, np.arange(ious.shape[1])]

# 각 ground truth 별로 가장 높은 iou를 가지는 anchor box의 index들
gt_argmax_ious = np.where(ious == gt_max_ious)[0]

# 각 anchor box 별로 가장 높은 iou를 가지는 ground truth의 index들
argmax_ious = ious.argmax(axis=1)

# 각 anchor box 별로 가장 높은 iou들
max_ious = ious[np.arange(len(index_inside)), argmax_ious]

# Set label matrix of valid anchor boxes
# 1: object, 0: background, -1: ignore
label = np.empty((len(index_inside), ), dtype=np.int32)
label.fill(-1)

pos_iou_threshold = .7
neg_iou_threshold = .3
label[gt_argmax_ious] = 1
label[max_ious >= pos_iou_threshold] = 1
label[max_ious < neg_iou_threshold] = 0

n_sample = 256
pos_ratio = .5
n_pos = pos_ratio * n_sample

pos_index = np.where(label == 1)[0]
if len(pos_index) > n_pos:
    disable_index = np.random.choice(pos_index, size=(len(pos_index) - int(n_pos)), replace=False)
    label[disable_index] = -1

n_neg = n_sample * np.sum(label == 1)
neg_index = np.where(label == 0)[0]
if len(neg_index) > n_neg:
    disable_index = np.random.choice(neg_index, size=(len(neg_index) - int(n_neg)), replace=False)
    label[disable_index] = -1

# 각 anchor box 별로 가장 높은 iou를 가지는 bounding box
max_iou_bbox = bbox[argmax_ious]

# valid anchor boxes의 h, w, cy, cx
height = valid_anchor_boxes[:, 2] - valid_anchor_boxes[:, 0]
width = valid_anchor_boxes[:, 3] - valid_anchor_boxes[:, 1]
ctr_y = valid_anchor_boxes[:, 0] + .5 * height
ctr_x = valid_anchor_boxes[:, 1] + .5 + width

base_height = max_iou_bbox[:, 2] - max_iou_bbox[:, 0]
base_width = max_iou_bbox[:, 3] - max_iou_bbox[:, 1]
base_ctr_y = max_iou_bbox[:, 0] + .5 * base_height
base_ctr_x = max_iou_bbox[:, 1] + .5 * base_width

# Get valid anchor boxes' deltas (loc = (y-ya)/ha, (x-xa)/wa, log(h/ha), log(w/wa))
eps = np.finfo(height.dtype).eps
height = np.maximum(height, eps)
width = np.maximum(width, eps)
dy = (base_ctr_y - ctr_y) / height
dx = (base_ctr_x - ctr_x) / width
dh = np.log(base_height / height)
dw = np.log(base_width / width)
anchor_locs = np.vstack((dy, dx, dh, dw)).transpose()
# print(anchor_locs.shape)

# Set label matrix of all anchor boxes
anchor_labels = np.empty((len(anchor_boxes), ), dtype=label.dtype)
anchor_labels.fill(-1)
anchor_labels[index_inside] = label

# Set location matrix of all anchor boxes
anchor_locations = np.empty((len(anchor_boxes), ) + anchor_boxes.shape[1:], dtype=anchor_locs.dtype)
anchor_locations.fill(-1)
anchor_locations[index_inside, :] = anchor_locs

in_channels = 512
mid_channels = 512
n_anchor = 9

conv1 = nn.Conv2d(in_channels, mid_channels, 3, 1, 1).to(device)
conv1.weight.data.normal_(0, .01)
conv1.bias.data.zero_()

reg_layer = nn.Conv2d(mid_channels, n_anchor * 4, 1, 1, 0).to(device)
reg_layer.weight.data.normal_(0, .01)
reg_layer.bias.data.zero_()

cls_layer = nn.Conv2d(mid_channels, n_anchor * 2, 1, 1, 0).to(device)
cls_layer.weight.data.normal_(0, .01)
cls_layer.bias.data.zero_()

x = conv1(out_map.to(device))
pred_anchor_locs = reg_layer(x)
pred_cls_scores = cls_layer(x)

# Change formats of anchor boxes
# [1, 36(9*4), 50, 50] => [1, 22500(50*50*9), 4] (dy, dx, dh, dw)
# [1, 18(9*2), 50, 50] => [1, 2500, 2] (1, 0)
pred_anchor_locs = pred_anchor_locs.permute(0, 2, 3, 1).contiguous().view(1, -1, 4)
pred_cls_scores = pred_cls_scores.permute(0, 2, 3, 1).contiguous()
objectness_score = pred_cls_scores.view(1, 50, 50, 9, 2)[:, :, :, :, 1].contiguous().view(1, -1)
pred_cls_scores = pred_cls_scores.view(1, -1, 2)

## Calculate RPN loss(classification scores, bounding boxes)
rpn_loc = pred_anchor_locs[0]
rpn_score = pred_cls_scores[0]

gt_rpn_loc = torch.from_numpy(anchor_locations)
gt_rpn_score = torch.from_numpy(anchor_labels)

# Cross entropy loss for classification
rpn_cls_loss = F.cross_entropy(rpn_score, gt_rpn_score.long().to(device), ignore_index=-1)

# Smooth L1 loss for regression (Fast RCNN paper)
pos = gt_rpn_score > 0
mask = pos.unsqueeze(1).expand_as(rpn_loc)

# Take bounding boxes which have positive labels
mask_loc_preds = rpn_loc[mask].view(-1, 4)
mask_loc_targets = gt_rpn_loc[mask].view(-1, 4)

x = torch.abs(mask_loc_targets.cpu() - mask_loc_preds.cpu())
rpn_loc_loss = ((x < 1).float() * .5 * x ** 2) + ((x >= 1).float() * (x - .5))

# Combine rpn_cls_loss and rpn_reg_loss
rpn_lambda = 10.
N_reg = (gt_rpn_score > 0).float().sum()
rpn_loc_loss = rpn_loc_loss.sum() / N_reg
rpn_loss = rpn_cls_loss + (rpn_lambda * rpn_loc_loss)

# NMS
# Reduce the number of rois from 22500 to 2000
nms_thresh = .7
n_train_pre_nms = 12000
n_train_post_nms = 2000
n_test_pre_nms = 6000
n_test_post_nms = 300
min_size = 16

# Convert 22500 labeled anchor boxes from [y0, x0, y1, x1] to [ctr_x, cty_y, h, w]
height_anchor = anchor_boxes[:, 2] - anchor_boxes[:, 0]
width_anchor = anchor_boxes[:, 3] - anchor_boxes[:, 1]
ctr_y_anchor = anchor_boxes[:, 0] + .5 * height_anchor
ctr_x_anchor = anchor_boxes[:, 1] + .5 * width_anchor

# Predict 22500 anchor boxes locations and labels with RPN
# (dy, dx, dh, dw)
pred_anchor_locs_numpy = pred_anchor_locs[0].cpu().data.numpy()
objectness_score_numpy = objectness_score[0].cpu().data.numpy()
dy = pred_anchor_locs_numpy[:, 0::4]
dx = pred_anchor_locs_numpy[:, 1::4]
dh = pred_anchor_locs_numpy[:, 2::4]
dw = pred_anchor_locs_numpy[:, 3::4]

ctr_y = dy * height_anchor[:, np.newaxis] + ctr_y_anchor[:, np.newaxis]
ctr_x = dx * width_anchor[:, np.newaxis] + ctr_x_anchor[:, np.newaxis]
h = np.exp(dh) * height_anchor[:, np.newaxis]
w = np.exp(dw) * width_anchor[:, np.newaxis]

# Calculate RoIs using labelled anchor boxes that is predicted with RPN.
roi = np.zeros(pred_anchor_locs_numpy.shape, dtype=anchor_locs.dtype)
roi[:, 0::4] = ctr_y - .5 * h
roi[:, 1::4] = ctr_x - .5 * w
roi[:, 2::4] = ctr_y + .5 * h
roi[:, 3::4] = ctr_x + .5 * w

# Clip the predicted boxes to the image
img_size = (800, 800)
roi[:, slice(0, 4, 2)] = np.clip(roi[:, slice(0, 4, 2)], 0, img_size[0])
roi[:, slice(1, 4, 2)] = np.clip(roi[:, slice(1, 4, 2)], 0, img_size[1])

# Remove predicted boxes with either height or width < threshold
hs = roi[:, 2] - roi[:, 0]
ws = roi[:, 3] - roi[:, 1]
keep = np.where((hs >= min_size) & (ws >= min_size))[0]
roi = roi[keep, :]
score = objectness_score_numpy[keep]

# Sort all (proposal, score) pairs by score in descending order
order = score.ravel().argsort()[::-1]

# Take top n_train_pre_nms scores (e.g.12000)
order = order[:n_train_pre_nms]
roi = roi[order, :]

# Take all the roi boxes
y1 = roi[:, 0]
x1 = roi[:, 1]
y2 = roi[:, 2]
x2 = roi[:, 3]

# Find the areas of all RoI boxes.
areas = (x2 - x1 + 1) * (y2 - y1 + 1)

# Take final region proposals.
order = order.argsort()[::-1]
keep = []

while order.size > 0:
    i = order[0] #take the 1st elt in order and append to keep
    keep.append(i)

    xx1 = np.maximum(x1[i], x1[order[1:]])
    yy1 = np.maximum(y1[i], y1[order[1:]])
    xx2 = np.minimum(x2[i], x2[order[1:]])
    yy2 = np.minimum(y2[i], y2[order[1:]])

    w = np.maximum(0.0, xx2 - xx1 + 1)
    h = np.maximum(0.0, yy2 - yy1 + 1)

    inter = w * h
    ovr = inter / (areas[i] + areas[order[1:]] - inter)
    inds = np.where(ovr <= nms_thresh)[0]
    order = order[inds + 1]

keep = keep[:n_train_post_nms] # while training/testing , use accordingly
roi = roi[keep] # the final region proposals

print(len(keep))
print(roi.shape)

import copy

img_clone = copy.deepcopy(img)

for i in range(len(roi)):
    y1, x1, y2, x2 = roi[i].astype(int)
    cv.rectangle(img_clone, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=1)

plt.figure(figsize=(8, 8))
plt.imshow(img_clone)
plt.show()

# Find the iou of each ground truth object with the region proposals,
ious = np.empty((len(roi), len(bbox)), dtype=np.float32)
ious.fill(0)
for num1, i in enumerate(roi):
    ya1, xa1, ya2, xa2 = i
    anchor_area = (ya2 - ya1) * (xa2 - xa1)
    for num2, j in enumerate(bbox):
        yb1, xb1, yb2, xb2 = j
        box_area = (yb2 - yb1) * (xb2 - xb1)
        inter_x1 = max([xb1, xa1])
        inter_y1 = max([yb1, ya1])
        inter_x2 = min([xb2, xa2])
        inter_y2 = min([yb2, ya2])
        if (inter_x1 < inter_x2) and (inter_y1 < inter_y2):
            iter_area = (inter_y2 - inter_y1) * (inter_x2 - inter_x1)
            iou = iter_area / (anchor_area + box_area - iter_area)
        else:
            iou = 0.
        ious[num1, num2] = iou
print('ious.shape ', ious.shape)

faults = []
for i in range(len(ious)):
    for j in range(3):
        if ious[i, j] > 1:
            faults.append([ious[i, j], i, j])

print('[Faults]')
print(len(faults))
print(faults)

## NMS for 2000 rois
n_sample = 128  # Number of samples from roi
pos_ratio = 0.25 # Number of positive examples out of the n_samples
pos_iou_thresh = 0.5  # Min iou of region proposal with any groundtruth object to consider it as positive label
neg_iou_thresh_hi = 0.5  # iou 0~0.5 is considered as negative (0, background)
neg_iou_thresh_lo = 0.0

# Find out which ground truth has high IoU for each region proposal, Also find the maximum IoU
gt_assignment = ious.argmax(axis=1)
max_iou = ious.max(axis=1)
print(gt_assignment)
print(max_iou)

# Assign the labels to each proposal
gt_roi_label = labels[gt_assignment]
print(gt_roi_label)

# Select the foreground rois as per the pos_iou_thesh and
# n_sample x pos_ratio (128 x 0.25 = 32) foreground samples.
pos_roi_per_image = 32
pos_index = np.where(max_iou >= pos_iou_thresh)[0]
pos_roi_per_this_image = int(min(pos_roi_per_image, pos_index.size))
if pos_index.size > 0:
    pos_index = np.random.choice(
        pos_index, size=pos_roi_per_this_image, replace=False)
print('pos_roi_per_this_image ', pos_roi_per_this_image)
print('pos_index ', pos_index)

# Similarly we do for negitive (background) region proposals
neg_index = np.where((max_iou < neg_iou_thresh_hi) &
                             (max_iou >= neg_iou_thresh_lo))[0]
neg_roi_per_this_image = n_sample - pos_roi_per_this_image
neg_roi_per_this_image = int(min(neg_roi_per_this_image, neg_index.size))
if  neg_index.size > 0 :
    neg_index = np.random.choice(
        neg_index, size=neg_roi_per_this_image, replace=False)
print('neg_roi_per_this_image ', neg_roi_per_this_image)
print('neg_index ', neg_index)

# display ROI samples with postive
img_clone = np.copy(img)
for i in range(pos_roi_per_this_image):
    y0, x0, y1, x1 = roi[pos_index[i]].astype(int)
    cv.rectangle(img_clone, (x0, y0), (x1, y1), color=(255, 0, 0), thickness=3)

for i in range(len(bbox)):
    cv.rectangle(img_clone, (bbox[i][1], bbox[i][0]), (bbox[i][3], bbox[i][2]), color=(0, 255, 0),
                  thickness=3)  # Draw Rectangle

plt.imshow(img_clone)
plt.show()

# display ROI samples with negative
img_clone = np.copy(img)
plt.figure(figsize=(9, 6))
for i in range(neg_roi_per_this_image):
    y0, x0, y1, x1 = roi[neg_index[i]].astype(int)
    cv.rectangle(img_clone, (x0, y0), (x1, y1), color=(255, 0, 0), thickness=3)

for i in range(len(bbox)):
    cv.rectangle(img_clone, (bbox[i][1], bbox[i][0]), (bbox[i][3], bbox[i][2]), color=(0, 255, 0),
                  thickness=3)  # Draw Rectangle

plt.imshow(img_clone)
plt.show()

# Now we gather positve samples index and negitive samples index,
# their respective labels and region proposals

keep_index = np.append(pos_index, neg_index)
gt_roi_labels = gt_roi_label[keep_index]
gt_roi_labels[pos_roi_per_this_image:] = 0  # negative labels --> 0
sample_roi = roi[keep_index]
print(sample_roi.shape)

# Pick the ground truth objects for these sample_roi and
# later parameterize as we have done while assigning locations to anchor boxes in section 2.
bbox_for_sampled_roi = bbox[gt_assignment[keep_index]]
print('bbox_for_sampled_roi.shape ', bbox_for_sampled_roi.shape)

height = sample_roi[:, 2] - sample_roi[:, 0]
width = sample_roi[:, 3] - sample_roi[:, 1]
ctr_y = sample_roi[:, 0] + 0.5 * height
ctr_x = sample_roi[:, 1] + 0.5 * width

base_height = bbox_for_sampled_roi[:, 2] - bbox_for_sampled_roi[:, 0]
base_width = bbox_for_sampled_roi[:, 3] - bbox_for_sampled_roi[:, 1]
base_ctr_y = bbox_for_sampled_roi[:, 0] + 0.5 * base_height
base_ctr_x = bbox_for_sampled_roi[:, 1] + 0.5 * base_width

eps = np.finfo(height.dtype).eps
height = np.maximum(height, eps)
width = np.maximum(width, eps)

dy = (base_ctr_y - ctr_y) / height
dx = (base_ctr_x - ctr_x) / width
dh = np.log(base_height / height)
dw = np.log(base_width / width)

gt_roi_locs = np.vstack((dy, dx, dh, dw)).transpose()
print(gt_roi_locs.shape)

## ROI Pooling
rois = torch.from_numpy(sample_roi).float()
roi_indices = 0 * np.ones((len(rois),), dtype=np.int32)
roi_indices = torch.from_numpy(roi_indices).float()
print(rois.shape, roi_indices.shape)

indices_and_rois = torch.cat([roi_indices[:, None], rois], dim=1)
xy_indices_and_rois = indices_and_rois[:, [0, 2, 1, 4, 3]]
indices_and_rois = xy_indices_and_rois.contiguous()
print(xy_indices_and_rois.shape)

size = (7, 7)
adaptive_max_pool = nn.AdaptiveMaxPool2d(size[0], size[1])

output = []
rois = indices_and_rois.data.float()
rois[:, 1:].mul_(1/16.0) # Subsampling ratio
rois = rois.long()
num_rois = rois.size(0)
for i in range(num_rois):
    roi = rois[i]
    im_idx = roi[0]
    im = out_map.narrow(0, im_idx, 1)[..., roi[2]:(roi[4]+1), roi[1]:(roi[3]+1)]
    tmp = adaptive_max_pool(im)
    output.append(tmp[0])
output = torch.cat(output, 0)
print(output.size())

# Visualize the first 5 ROI's feature map (for each feature map, only show the 1st channel of d=512)
fig=plt.figure(figsize=(12, 4))
figNo = 1
for i in range(5):
    roi = rois[i]
    im_idx = roi[0]
    im = out_map.narrow(0, im_idx, 1)[..., roi[2]:(roi[4]+1), roi[1]:(roi[3]+1)]
    tmp = im[0][0].detach().cpu().numpy()
    fig.add_subplot(1, 5, figNo)
    plt.imshow(tmp, cmap='gray')
    figNo +=1
plt.show()

# Visualize the first 5 ROI's feature maps after ROI pooling (for each feature map, only show the 1st channel of d=512)
fig=plt.figure(figsize=(12, 4))
figNo = 1
for i in range(5):
    roi = rois[i]
    im_idx = roi[0]
    im = out_map.narrow(0, im_idx, 1)[..., roi[2]:(roi[4]+1), roi[1]:(roi[3]+1)]
    tmp = adaptive_max_pool(im)[0]
    tmp = tmp[0][0].detach().cpu().numpy()
    fig.add_subplot(1, 5, figNo)
    plt.imshow(tmp, cmap='gray')
    figNo +=1
plt.show()

# Reshape the tensor so that we can pass it through the feed forward layer.
k = output.view(output.size(0), -1)
print(k.shape) # 25088 = 7*7*512

## 128 ROI samples' boxes + features (7x7x512) detection network bounding box class
## Classifier
roi_head_classifier = nn.Sequential(*[nn.Linear(25088, 4096), nn.Linear(4096, 4096)]).to(device)
cls_loc = nn.Linear(4096, 2 * 4).to(device) # (1 classes 安全帽 + 1 background. Each will have 4 co-ordinates)
cls_loc.weight.data.normal_(0, 0.01)
cls_loc.bias.data.zero_()

score = nn.Linear(4096, 2).to(device) # (1 classes, 安全帽 + 1 background)

# passing the output of roi-pooling to ROI head
k = roi_head_classifier(k.to(device))
roi_cls_loc = cls_loc(k)
roi_cls_score = score(k)
print(roi_cls_loc.shape, roi_cls_score.shape)

## 128 ROI ground truth bounding boxes, features (h, w, d=512), loss
# predicted
print(roi_cls_loc.shape)
print(roi_cls_score.shape)

#actual
print(gt_roi_locs.shape)
print(gt_roi_labels.shape)

# Converting ground truth to torch variable
gt_roi_loc = torch.from_numpy(gt_roi_locs)
gt_roi_label = torch.from_numpy(np.float32(gt_roi_labels)).long()
print(gt_roi_loc.shape, gt_roi_label.shape)

#Classification loss
roi_cls_loss = F.cross_entropy(roi_cls_score.cpu(), gt_roi_label.cpu(), ignore_index=-1)
print(roi_cls_loss.shape)

# Regression loss
n_sample = roi_cls_loc.shape[0]
roi_loc = roi_cls_loc.view(n_sample, -1, 4)
print(roi_loc.shape)

roi_loc = roi_loc[torch.arange(0, n_sample).long(), gt_roi_label]
print(roi_loc.shape)

# For Regression we use smooth L1 loss as defined in the Fast RCNN paper
pos = gt_roi_label > 0
mask = pos.unsqueeze(1).expand_as(roi_loc)
print(mask.shape)

# take those bounding boxes which have positve labels
mask_loc_preds = roi_loc[mask].view(-1, 4)
mask_loc_targets = gt_roi_loc[mask].view(-1, 4)
print(mask_loc_preds.shape, mask_loc_targets.shape)

x = torch.abs(mask_loc_targets.cpu() - mask_loc_preds.cpu())
roi_loc_loss = ((x < 1).float() * 0.5 * x**2) + ((x >= 1).float() * (x-0.5))
print(roi_loc_loss.sum())

roi_lambda = 10.
roi_loss = roi_cls_loss + (roi_lambda * roi_loc_loss)
print(roi_loss)

total_loss = rpn_loss + roi_loss
print(total_loss)