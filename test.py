import pandas as pd
import numpy as np
import cv2
import os
import re

from PIL import Image

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import torch
import torchvision

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from dataset.wheat import WheatDataset,WheatTestDataset
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SequentialSampler

from matplotlib import pyplot as plt
import Weighted_Boxes_Fusion.ensemble_boxes

DIR_INPUT = '/data1/jliang_data/dataset/wheat'
DIR_TRAIN = f'{DIR_INPUT}/train'
DIR_TEST = f'{DIR_INPUT}/test'
WEIGHTS_FILE = f'/data1/jliang_data/competition/first/global_wheat_detection/new_model/fasterrcnn_resnet152_fpn-30.pth'
test_df = pd.read_csv(f'{DIR_INPUT}/sample_submission.csv')

os.environ["CUDA_VISIBLE_DEVICES"] = '3'

# Albumentations
def get_test_transform():
    return A.Compose([
        # A.Resize(512, 512),
        ToTensorV2(p=1.0)
    ])

from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
def fasterrcnn_resnet101_fpn(pretrained=False, progress=True,
                            num_classes=91, pretrained_backbone=False,
                             trainable_backbone_layers=3, **kwargs):
    assert trainable_backbone_layers <= 5 and trainable_backbone_layers >= 0
    # dont freeze any layers if pretrained model or backbone is not used
    if not (pretrained or pretrained_backbone):
        trainable_backbone_layers = 5 #TODO: whta's mean of this trainable_backbone_layers
    if pretrained:
        # no need to download the backbone if pretrained is set
        pretrained_backbone = False
    backbone = resnet_fpn_backbone('resnet152', pretrained_backbone)
    model = FasterRCNN(backbone, num_classes, **kwargs)
    return model

def initialize_model():
    model = fasterrcnn_resnet101_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    return model

# load a model; pre-trained on COCO
# model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=False)
model = initialize_model()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

num_classes = 2  # 1 class (wheat) + background

# get number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features

# replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# Load the trained weights
model.load_state_dict(torch.load(WEIGHTS_FILE))
model.eval()

x = model.to(device)

def collate_fn(batch):
    return tuple(zip(*batch))

test_dataset = WheatTestDataset(DIR_INPUT, get_test_transform())

test_data_loader = DataLoader(
    test_dataset,
    batch_size=4,
    shuffle=False,
    num_workers=0,
    drop_last=False,
    collate_fn=collate_fn
)

def format_prediction_string(boxes, scores):
    pred_strings = []
    for j in zip(scores, boxes):
        pred_strings.append("{0:.4f} {1} {2} {3} {4}".format(j[0], j[1][0], j[1][1], j[1][2], j[1][3]))

    return " ".join(pred_strings)


detection_threshold = 0.5
results = []

for images, image_ids in test_data_loader:

    images = list(image.to(device) for image in images)
    outputs = model(images)

    for i, image in enumerate(images):
        boxes = outputs[i]['boxes'].data.cpu().numpy()
        scores = outputs[i]['scores'].data.cpu().numpy()

        boxes = boxes[scores >= detection_threshold].astype(np.int32)
        scores = scores[scores >= detection_threshold]
        image_id = image_ids[i]

        boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
        boxes[:, 3] = boxes[:, 3] - boxes[:, 1]

        result = {
            'image_id': image_id,
            'PredictionString': format_prediction_string(boxes, scores)
        }

        results.append(result)

test_df = pd.DataFrame(results, columns=['image_id', 'PredictionString'])
sample = images[1].permute(1,2,0).cpu().numpy()
boxes = outputs[1]['boxes'].data.cpu().numpy()
scores = outputs[1]['scores'].data.cpu().numpy()

boxes = boxes[scores >= detection_threshold].astype(np.int32)

for box in boxes:
    cv2.rectangle(sample,
                  (box[0], box[1]),
                  (box[2], box[3]),
                  (220, 0, 0), 2)

sample = cv2.cvtColor(sample, cv2.COLOR_RGB2BGR).astype(np.float32)
cv2.imwrite('demo.jpg', sample*255)
test_df.to_csv('submission.csv', index=False)


class BaseWheatTTA:
    """ author: @shonenkov """
    image_size = 1024

    def augment(self, image):
        raise NotImplementedError

    def batch_augment(self, images):
        raise NotImplementedError

    def deaugment_boxes(self, boxes):
        raise NotImplementedError


class TTAHorizontalFlip(BaseWheatTTA):
    """ author: @shonenkov """

    def augment(self, image):
        return image.flip(1)

    def batch_augment(self, images):
        return images.flip(2)

    def deaugment_boxes(self, boxes):
        boxes[:, [1, 3]] = self.image_size - boxes[:, [3, 1]]
        return boxes


class TTAVerticalFlip(BaseWheatTTA):
    """ author: @shonenkov """

    def augment(self, image):
        return image.flip(2)

    def batch_augment(self, images):
        return images.flip(3)

    def deaugment_boxes(self, boxes):
        boxes[:, [0, 2]] = self.image_size - boxes[:, [2, 0]]
        return boxes


class TTARotate90(BaseWheatTTA):
    """ author: @shonenkov """

    def augment(self, image):
        return torch.rot90(image, 1, (1, 2))

    def batch_augment(self, images):
        return torch.rot90(images, 1, (2, 3))

    def deaugment_boxes(self, boxes):
        res_boxes = boxes.copy()
        res_boxes[:, [0, 2]] = self.image_size - boxes[:, [1, 3]]
        res_boxes[:, [1, 3]] = boxes[:, [2, 0]]
        return res_boxes


class TTARotate180(BaseWheatTTA):
    """ author: @shonenkov """

    def augment(self, image):
        return torch.rot90(image, 2, (1, 2))

    def batch_augment(self, images):
        return torch.rot90(images, 2, (2, 3))

    def deaugment_boxes(self, boxes):
        boxes[:, [0, 1, 2, 3]] = self.image_size - boxes[:, [2, 3, 0, 1]]
        return boxes


class TTARotate270(BaseWheatTTA):
    """ author: @shonenkov """

    def augment(self, image):
        return torch.rot90(image, 3, (1, 2))

    def batch_augment(self, images):
        return torch.rot90(images, 3, (2, 3))

    def deaugment_boxes(self, boxes):
        res_boxes = boxes.copy()
        res_boxes[:, [0, 2]] = boxes[:, [1, 3]]
        res_boxes[:, [1, 3]] = self.image_size - boxes[:, [2, 0]]
        return res_boxes


class TTACompose(BaseWheatTTA):
    """ author: @shonenkov """

    def __init__(self, transforms):
        self.transforms = transforms

    def augment(self, image):
        for transform in self.transforms:
            image = transform.augment(image)
        return image

    def batch_augment(self, images):
        for transform in self.transforms:
            images = transform.batch_augment(images)
        return images

    def prepare_boxes(self, boxes):
        result_boxes = boxes.copy()
        result_boxes[:, 0] = np.min(boxes[:, [0, 2]], axis=1)
        result_boxes[:, 2] = np.max(boxes[:, [0, 2]], axis=1)
        result_boxes[:, 1] = np.min(boxes[:, [1, 3]], axis=1)
        result_boxes[:, 3] = np.max(boxes[:, [1, 3]], axis=1)
        return result_boxes

    def deaugment_boxes(self, boxes):
        for transform in self.transforms[::-1]:
            boxes = transform.deaugment_boxes(boxes)
        return self.prepare_boxes(boxes)

def process_det(index, det, score_threshold=0.5):
    boxes = det[index]['boxes'].detach().cpu().numpy()
    scores = det[index]['scores'].detach().cpu().numpy()
    # boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
    # boxes[:, 3] = boxes[:, 3] - boxes[:, 1]
    boxes = (boxes).clip(min=0, max=1023).astype(int) #TODO:511?
    indexes = np.where(scores>score_threshold)
    boxes = boxes[indexes]
    scores = scores[indexes]
    return boxes, scores


# you can try own combinations:
transform = TTACompose([
    TTARotate90(),
    TTAVerticalFlip(),
])

fig, ax = plt.subplots(1, 3, figsize=(16, 6))

image, image_id = test_dataset[8]

numpy_image = image.permute(1, 2, 0).cpu().numpy().copy()

ax[0].imshow(numpy_image)
ax[0].set_title('original')

tta_image = transform.augment(image)
tta_image_numpy = tta_image.permute(1, 2, 0).cpu().numpy().copy()

det = model(tta_image.unsqueeze(0).float().cuda())
boxes, scores = process_det(0, det)

for box in boxes:
    cv2.rectangle(tta_image_numpy, (box[0], box[1]), (box[2], box[3]), (0, 1, 0), 2)

ax[1].imshow(tta_image_numpy)
ax[1].set_title('tta')

boxes = transform.deaugment_boxes(boxes)

for box in boxes:
    cv2.rectangle(numpy_image, (box[0], box[1]), (box[2], box[3]), (0, 1, 0), 2)

ax[2].imshow(numpy_image)
ax[2].set_title('deaugment predictions')
plt.savefig('tta_demo.jpg')
