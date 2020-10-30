import os
import torch
import torch.utils.data as data
from pycocotools.coco import COCO
from PIL import Image


class Dataset(data.Dataset):
    def __init__(self, root, annotation, transforms=None):
        self.root = root
        self.annotation = annotation
        self.coco = COCO(annotation)
        self.ids = list(self.coco.imgs.keys())

    def __getitem__(self, index):
        coco = self.coco

        img_id = self.ids[index]
        ann_id = coco.getAnnIds(imgIds=img_id)
        img_path = coco.loadImgs(img_id)['file_name']
        img = Image.open(os.path.join(self.root, img_path))
        ann = coco.loadAnns(ann_id)

        num_objs = len(ann)

        boxes = []
        for i in range(num_objs):
            x_min = ann[i]['bbox'][0]
            y_min = ann[i]['bbox'][1]
            x_max = x_min + ann[i]['bbox'][2]
            y_max = y_min + ann[i]['bbox'][3]
            boxes.append([x_min, y_min, x_max, y_max])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((num_objs, ), dtype=torch.int32)
        iscrowd = torch.zeros((num_objs, ), dtype=torch.int32)

        areas = []
        for i in range(num_objs):
            areas.append(ann[i]['area'])
        areas = torch.as_tensor(areas, dtype=torch.float32)

        my_ann = {}
        my_ann['bbox'] = boxes
        my_ann['label'] = labels
        my_ann['iscrowd'] = iscrowd
        my_ann['area'] = areas

        return img, my_ann

    def __len__(self):
        return len(self.ids)