import pandas as pd
import numpy as np
import cv2
import os
import re
import math
from PIL import Image
from numpy import random
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
import torch.nn.functional as F
from matplotlib import pyplot as plt
import Weighted_Boxes_Fusion.ensemble_boxes as ensemble_boxes

from itertools import product

DIR_INPUT = '/data1/jliang_data/dataset/wheat'
DIR_TRAIN = f'{DIR_INPUT}/train'
DIR_TEST = f'{DIR_INPUT}/test'
WEIGHTS_FILE = f'/data1/jliang_data/competition/first/global_wheat_detection/new_model/fasterrcnn_resnet152_fpn-30.pth'
test_df = pd.read_csv(f'{DIR_INPUT}/sample_submission.csv')

os.environ["CUDA_VISIBLE_DEVICES"] = '4'

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
    batch_size=2,
    shuffle=False,
    num_workers=0,
    drop_last=False,
    collate_fn=collate_fn
)

#ref:https://www.kaggle.com/shonenkov/wbf-over-tta-single-model-efficientdet
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

#ref:https://www.kaggle.com/wasupandceacar/yolov5-single-model-with-more-tta-lb-0-745
class TTAScale(BaseWheatTTA):
    def __init__(self, ratio=1.0, same_shape=False):
        self.ratio = ratio #[0.83, 0.67]
        self.same_shape = same_shape

    def augment(self, image):# img(16,3,256,416), r=ratio
        # scales img(bs,3,y,x) by ratio
        h, w = image.shape[1:]
        s = (int(h * self.ratio), int(w * self.ratio))  # new size
        img = F.interpolate(image.unsqueeze(0), size=s, mode='bilinear', align_corners=False)  # resize
        if not self.same_shape:  # pad/crop img
            gs = 32  # (pixels) grid size
            h, w = [math.ceil(x * self.ratio / gs) * gs for x in (h, w)]
        return F.pad(img, [0, w - s[1], 0, h - s[0]], value=0.447).squeeze(0)  # value = imagenet mean

    def deaugment_boxes(self, boxes):
        boxes /= self.ratio
        return boxes

class PhotoMetricDistortion(BaseWheatTTA):
    def __init__(self,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def augment(self, image):
        img = image.permute(1, 2, 0).cpu().numpy()*255.0
        # random brightness
        if random.randint(2):
            delta = random.uniform(-self.brightness_delta,
                                   self.brightness_delta)
            img += delta

        # mode == 0 --> do random contrast first
        # mode == 1 --> do random contrast last
        mode = random.randint(2)
        if mode == 1:
            if random.randint(2):
                alpha = random.uniform(self.contrast_lower,
                                       self.contrast_upper)
                img *= alpha

        # convert color from BGR to HSV
        # img = mmcv.bgr2hsv(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # random saturation
        if random.randint(2):
            img[..., 1] *= random.uniform(self.saturation_lower,
                                          self.saturation_upper)

        # random hue
        if random.randint(2):
            img[..., 0] += random.uniform(-self.hue_delta, self.hue_delta)
            img[..., 0][img[..., 0] > 360] -= 360
            img[..., 0][img[..., 0] < 0] += 360

        # convert color from HSV to BGR
        # img = mmcv.hsv2bgr(img)
        img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

        # random contrast
        if mode == 0:
            if random.randint(2):
                alpha = random.uniform(self.contrast_lower,
                                       self.contrast_upper)
                img *= alpha

        # randomly swap channels
        if random.randint(2):
            img = img[..., random.permutation(3)]

        img = torch.from_numpy(img/255.0).permute(2, 0, 1).cuda()
        return img

    def deaugment_boxes(self, boxes):
        return boxes


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

def format_prediction_string(boxes, scores):
    pred_strings = []
    for j in zip(scores, boxes):
        pred_strings.append("{0:.4f} {1} {2} {3} {4}".format(j[0], j[1][0], j[1][1], j[1][2], j[1][3]))

    return " ".join(pred_strings)

def run_wbf(predictions, image_size=1024, iou_thr=0.44, skip_box_thr=0.43, weights=None): #iou_thr=0.41, skip_box_thr=0.4 #iou_thr=0.4, skip_box_thr=0.34
    boxes = [(prediction/(image_size-1)).tolist() for prediction in predictions['boxes']]
    scores = [prediction.tolist() for prediction in predictions['scores']]
    labels = [np.ones(len(prediction)).astype(int).tolist() for prediction in predictions['scores']]
    boxes, scores, labels = ensemble_boxes.ensemble_boxes_wbf.weighted_boxes_fusion(boxes, scores, labels, weights=None, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
    boxes = boxes*(image_size-1)
    return boxes, scores, labels

tta_transforms = []
for tta_combination in product(
        # [TTAHorizontalFlip(), None],
        # [TTAVerticalFlip(), None],
        # [TTARotate90(), None]
        # [TTAScale(ratio=0.8), None],
        [PhotoMetricDistortion(), None]
):
    tta_transforms.append(TTACompose([tta_transform for tta_transform in tta_combination if tta_transform]))
detection_threshold = 0.25 #TODO: how to set
results = []
for images, image_ids in test_data_loader:

    images = list(image.to(device) for image in images)

    for i, image in enumerate(images):
        predictions = {'image_id':image_ids[i], 'boxes':[], 'scores':[]}
        for tta_transform in tta_transforms:
            outputs = model([tta_transform.augment(image.clone())])
            boxes = outputs[0]['boxes'].data.cpu().numpy()
            scores = outputs[0]['scores'].data.cpu().numpy()

            boxes = boxes[scores >= detection_threshold]
            scores = scores[scores >= detection_threshold]
            boxes = tta_transform.deaugment_boxes(boxes.copy()).astype(np.int32)

            predictions['boxes'].append(boxes)
            predictions['scores'].append(scores)

        # sample = image.permute(1, 2, 0).cpu().numpy()
        # aug_boxes = predictions['boxes']
        # for boxes in aug_boxes:
        #     for box in boxes:
        #         cv2.rectangle(sample,
        #                       (box[0], box[1]),
        #                       (box[2], box[3]),
        #                       (220, 0, 0), 2)
        # sample = cv2.cvtColor(sample, cv2.COLOR_RGB2BGR).astype(np.float32)
        # cv2.imwrite('demo.jpg', sample * 255)

        wbf_boxes, wbf_scores, _ = run_wbf(predictions)
        wbf_boxes = wbf_boxes.astype(np.int32)#TODO: .clip(min=0, max=1024)?
        sample = image.permute(1, 2, 0).cpu().numpy()
        for box in wbf_boxes:
            cv2.rectangle(sample,
                          (box[0], box[1]),
                          (box[2], box[3]),
                          (220, 0, 0), 2)
        sample = cv2.cvtColor(sample, cv2.COLOR_RGB2BGR).astype(np.float32)
        cv2.imwrite('./img/'+image_ids[i]+'.jpg', sample * 255)
        wbf_boxes[:, 2] = wbf_boxes[:, 2] - wbf_boxes[:, 0]
        wbf_boxes[:, 3] = wbf_boxes[:, 3] - wbf_boxes[:, 1]
        result = {
            'image_id': image_ids[i],
            'PredictionString': format_prediction_string(wbf_boxes, wbf_scores)
        }

        results.append(result)

test_df = pd.DataFrame(results, columns=['image_id', 'PredictionString'])
test_df.to_csv('submission.csv', index=False)