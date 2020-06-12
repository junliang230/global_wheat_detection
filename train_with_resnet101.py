import os
import time
import torch
import argparse
import torchvision
import pandas as pd
import numpy as np
import albumentations as A
from dataset.transform import PhotoMetricDistortion, MixUp, Mosaic
from loss.averager import Averager
from dataset.wheat import WheatDataset,WheatTestDataset
from utils.Network_utils import get_logger,summary_args,Timer,wrap_color,info

from torch.utils.data import DataLoader, Dataset
from albumentations.pytorch.transforms import ToTensorV2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.rpn import AnchorGenerator
from torch.utils.data.sampler import SequentialSampler

DIR_INPUT = '/data1/jliang_data/dataset/wheat'
DIR_TRAIN = f'{DIR_INPUT}/train'
DIR_TEST = f'{DIR_INPUT}/test'

os.environ["CUDA_VISIBLE_DEVICES"] = '2'

# Albumentations
def get_train_transform():
    train_pipline = [
        PhotoMetricDistortion(
            brightness_delta=32,
            contrast_range=(0.5, 1.5),
            saturation_range=(0.5, 1.5),
            hue_delta=18),
        # MixUp(p=0.5, mode=1),
        # Mosaic(p=0.2),
        A.Compose([
            A.Flip(p=0.5),
            A.ToGray(p=0.01),
            A.Cutout(num_holes=8, max_h_size=64, max_w_size=64, fill_value=0, p=0.5),
            A.RandomCrop(height=1000, width=1000, p=0.5),
            ToTensorV2(p=1.0)
        ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
    ]

    return train_pipline


def get_valid_transform():
    return A.Compose([
        ToTensorV2(p=1.0)
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})


train_dataset = WheatDataset(DIR_INPUT, get_train_transform())


def collate_fn(batch):
    return tuple(zip(*batch))

def fasterrcnn_resnet101_fpn(pretrained=False, progress=True,
                            num_classes=91, pretrained_backbone=True,
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
    model = fasterrcnn_resnet101_fpn(pretrained=False, min_size=[512, 800, 1024], max_size=1024,
                                    image_mean=[123.675, 116.28, 103.53], image_std=[58.395, 57.12, 57.375])
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    return model

def train(args):
    t = time.strftime("-%Y-%m-%d-%H-%M-%S", time.localtime())
    name = 'Log' + t
    logger = get_logger('log', name)

    summary_args(logger, vars(args), 'green')

    train_data_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,  # 16
        shuffle=args.shuffle,  # set it to True??
        num_workers=4,
        collate_fn=collate_fn  # any diff with default???
    )

    model = initialize_model()
    # # load a model; pre-trained on COCO
    # model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True,
    #                                                              min_size=[512, 800, 1024], max_size=1024,
    #                                                              image_mean=[123.675, 116.28, 103.53], image_std=[58.395, 57.12, 57.375])
    # num_classes = 2  # 1 class (wheat) + background
    #
    # # get number of input features for the classifier
    # in_features = model.roi_heads.box_predictor.cls_score.in_features
    #
    # # replace the pre-trained head with a new one
    # model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[25, 29], gamma=0.1)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    # lr_scheduler = None

    num_epochs = args.num_epoch
    loss_hist = Averager()
    itr = 1

    for epoch in range(num_epochs):
        loss_hist.reset()

        Timer.record()
        for images, targets, image_ids in train_data_loader:

            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)

            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()

            loss_hist.send(loss_value)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            if itr % 50 == 0:
                Timer.record()
                # print(f"Iteration #{itr} loss: {loss_value}")
                now_lr = optimizer.state_dict()['param_groups'][0]['lr']

                msg = 'Epoch={}, Batch={}, lr={}, loss={:.4f}, speed={:.1f} b/s'
                msg = msg.format(epoch, itr, now_lr, loss_value, 50 / Timer.interval())
                info(logger, msg)

            itr += 1

        # update the learning rate
        if lr_scheduler is not None:
            lr_scheduler.step()

        print(f"Epoch #{epoch} loss: {loss_hist.value}")
    torch.save(model.state_dict(), 'fasterrcnn_resnet152_fpn' + t + '.pth')


if __name__ == "__main__":
    parse = argparse.ArgumentParser()

    # LR setting
    parse.add_argument('--lr', type=float, default=0.00125)
    parse.add_argument('--momentum', type=float, default=0.9)
    parse.add_argument('--weight-decay', type=float, default=0.0001)

    # Train setting
    parse.add_argument('--num-epoch', type=int, default=30)
    parse.add_argument('--batch-size', type=int, default=4)
    parse.add_argument('--shuffle', type=bool, default=True)

    args = parse.parse_args()

    train(args)
