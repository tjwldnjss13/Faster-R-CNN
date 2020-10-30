from coco_dataset import COCODataset, collate_fn
from torch.utils.data import DataLoader

import os
import torchvision.transforms as transforms

batch_size = 2


root = 'C://DeepLearningData/COCOdataset2017/'
root_train = os.path.join(root, 'images', 'train')
root_val = os.path.join(root, 'images', 'val')
ann_train = os.path.join(root, 'annotations', 'instances_train2017.json')
ann_val = os.path.join(root, 'annotations', 'instances_val2017.json')

dset_train = COCODataset(root_train, ann_train, categorical=True, transforms=transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()]), instance_seg=False)
dset_val = COCODataset(root_val, ann_val, categorical=True, transforms=transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()]), instance_seg=False)

num_train_images = dset_train.__len__()
num_val_images = dset_val.__len__()

train_data_loader = DataLoader(dset_train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_data_loader = DataLoader(dset_val, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

for i, (images, anns) in enumerate(train_data_loader):
    # label = anns[0]['category_id']
    print(anns)

    if i > 5:
        break
