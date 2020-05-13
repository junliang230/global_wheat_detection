# ref:https://www.kaggle.com/pestipeti/pytorch-starter-fasterrcnn-train
import os
import time
import torch
import argparse
import torchvision
import pandas as pd
import numpy as np
import albumentations as A

from loss.averager import Averager
from dataset.wheat import WheatDataset,WheatTestDataset
from utils.Network_utils import get_logger,summary_args,Timer,wrap_color,info

from torch.utils.data import DataLoader, Dataset
from albumentations.pytorch.transforms import ToTensorV2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torch.utils.data.sampler import SequentialSampler

DIR_INPUT = '/media/data1/jliang_data/dataset/wheat'
WEIGHTS_FILE = '/media/data2/wrye/gittemp/global_wheat_detection/fasterrcnn_resnet50_fpn_19-2020-05-13-08-36-29.pth'

os.environ["CUDA_VISIBLE_DEVICES"] = '2'

# Albumentations
def get_train_transform():
    return A.Compose([
        A.Flip(0.5),
        ToTensorV2(p=1.0)
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})


def get_valid_transform():
    return A.Compose([
        ToTensorV2(p=1.0)
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})

def get_test_transform():
    return A.Compose([
        # A.Resize(512, 512),
        ToTensorV2(p=1.0)
    ])

def collate_fn(batch):
    return tuple(zip(*batch))

def loadmodel():

    # load a model; pre-trained on COCO
    # model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True,
    #                                                              min_size=1024, max_size=1024,
    #                                                              image_mean=[123.675, 116.28, 103.53], image_std=[58.395, 57.12, 57.375])

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False,pretrained_backbone=False,
                                                                 min_size=1024, max_size=1024,
                                                                 image_mean=[123.675, 116.28, 103.53],
                                                                 image_std=[58.395, 57.12, 57.375])

    num_classes = 2  # 1 class (wheat) + background

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model.to(device)

    return model,device


def pseudo_test(model,device):

    test_dataset = WheatTestDataset(DIR_INPUT, get_test_transform())

    test_data_loader = DataLoader(
        test_dataset,
        # batch_size=1,
        batch_size=4,
        shuffle=True,
        # num_workers=1,
        num_workers=4,
        drop_last=False,
        collate_fn=collate_fn
    )

    model.load_state_dict(torch.load(WEIGHTS_FILE))
    model.eval()

    detection_threshold = 0.5
    testdf_psuedo = []

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

            for box in boxes:
                # print(box)
                result = {
                    'image_id': 'nvnn' + image_id,
                    'width': 1024,
                    'height': 1024,
                    'source': 'nvnn',
                    'x': box[0],
                    'y': box[1],
                    'w': box[2],
                    'h': box[3]
                }
                testdf_psuedo.append(result)

    test_df_pseudo = pd.DataFrame(testdf_psuedo, columns=['image_id', 'width', 'height', 'source', 'x', 'y', 'w', 'h'])
    # test_df_pseudo.head()

    return test_df_pseudo

def train(args,model,device,test_df):
    t = time.strftime("-%Y-%m-%d-%H-%M-%S", time.localtime())
    name = 'Log' + t
    logger = get_logger('log', name)

    summary_args(logger, vars(args), 'green')

    train_dataset = WheatDataset(DIR_INPUT, get_train_transform(),test_df)

    train_data_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,  # 16
        shuffle=args.shuffle, # set it to True??
        # num_workers=1,
        num_workers=4,
        collate_fn=collate_fn  # any diff with default???
    )

    model.train()

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[16, 19], gamma=0.1)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
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

        if epoch==9 or epoch==19:
            torch.save(model.state_dict(), 'fasterrcnn_resnet50_fpn_' + str(epoch) + t + '.pth')

        # update the learning rate
        if lr_scheduler is not None:
            lr_scheduler.step()

        print(f"Epoch #{epoch} loss: {loss_hist.value}")
    torch.save(model.state_dict(), 'fasterrcnn_resnet50_fpn_'+ str(num_epochs)+ t + '.pth')

if __name__ == "__main__":
    parse = argparse.ArgumentParser()

    # LR setting
    parse.add_argument('--lr', type=float, default=0.01)
    parse.add_argument('--momentum', type=float, default=0.9)
    parse.add_argument('--weight-decay', type=float, default=0.0001)

    # Train setting
    parse.add_argument('--num-epoch', type=int, default=20)
    parse.add_argument('--batch-size', type=int, default=8)
    parse.add_argument('--shuffle', type=bool, default=True)

    args = parse.parse_args()

    model,device=loadmodel()
    test_df=pseudo_test(model,device)
    train(args,model,device,test_df)
