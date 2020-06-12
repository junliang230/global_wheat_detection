from torch.utils.data import DataLoader, Dataset
import cv2
import torch
import numpy as np
import re
import pandas as pd

DIR_INPUT = '/data1/jliang_data/dataset/wheat'
DIR_TEST = f'{DIR_INPUT}/test'

class WheatDataset(Dataset):

    def __init__(self, DIR_INPUT,transforms=None,test_df=None):
        super().__init__()

        train_df = pd.read_csv(f'{DIR_INPUT}/train.csv')

        train_df['x'] = -1
        train_df['y'] = -1
        train_df['w'] = -1
        train_df['h'] = -1
        train_df[['x', 'y', 'w', 'h']] = np.stack(train_df['bbox'].apply(lambda x: self.expand_bbox(x)))
        train_df.drop(columns=['bbox'], inplace=True)
        train_df['x'] = train_df['x'].astype(np.float)
        train_df['y'] = train_df['y'].astype(np.float)
        train_df['w'] = train_df['w'].astype(np.float)
        train_df['h'] = train_df['h'].astype(np.float)
        image_ids = train_df['image_id'].unique()
        valid_ids = image_ids[-665:]
        # train_ids = image_ids[:-2000]
        valid_df = train_df[train_df['image_id'].isin(valid_ids)]
        # train_df = train_df[train_df['image_id'].isin(train_ids)]

        #incorporate train_df and test_df
        frames = [train_df, test_df]
        train_df = pd.concat(frames)
        # train_df.tail()

        self.image_ids = train_df['image_id'].unique()
        self.df = train_df
        self.image_dir = f'{DIR_INPUT}/train'
        self.transforms = transforms

    def expand_bbox(self, x):
        r = np.array(re.findall("([0-9]+[.]?[0-9]*)", x))
        if len(r) == 0:
            r = [-1, -1, -1, -1]
        return r

    def __getitem__(self, index: int):
        image_id = self.image_ids[index]
        records = self.df[self.df['image_id'] == image_id]


        if 'nvnn' in image_id:
            image_id = image_id[4:]
            image = cv2.imread(f'{DIR_TEST}/{image_id}.jpg', cv2.IMREAD_COLOR)
        else:
            image = cv2.imread(f'{self.image_dir}/{image_id}.jpg', cv2.IMREAD_COLOR)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        # image /= 255.0

        boxes = records[['x', 'y', 'w', 'h']].values
        boxes[:, 0] = np.where(boxes[:, 0] < 0, 0, boxes[:, 0])
        boxes[:, 1] = np.where(boxes[:, 1] < 0, 0, boxes[:, 1])
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
        boxes[:, 2] = np.where(boxes[:, 2] > 1024, 1024, boxes[:, 2]) #TODO: change 1024 to image width
        boxes[:, 3] = np.where(boxes[:, 3] > 1024, 1024, boxes[:, 3])

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        area = torch.as_tensor(area, dtype=torch.float32)

        # there is only one class
        labels = torch.ones((records.shape[0],), dtype=torch.int64)

        # suppose all instances are not crowd
        iscrowd = torch.zeros((records.shape[0],), dtype=torch.int64)

        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        # target['masks'] = None
        target['image_id'] = torch.tensor([index])
        target['area'] = area
        target['iscrowd'] = iscrowd

        if self.transforms:
            results = {
                'image': image,
                'bboxes': target['boxes'],
                'labels': labels,
                'idx': index,
                'image_ids': self.image_ids,
                'df': self.df,
            }
            for t in self.transforms[:-1]:
                while True:
                    new_results = t(results)
                    if len(new_results['bboxes']) != 0:
                        results = new_results
                        break

            if type(results['labels']) is np.ndarray:
                results['labels'] = torch.from_numpy(results['labels']).type(torch.int64)
            sample = {
                'image': results['image'],
                'bboxes': results['bboxes'],
                'labels': results['labels'],
            }
            for t in [self.transforms[-1]]: #TODO: diff with self.transforms[-1]? datatype of label will change?
                while True:
                    new_sample = t(**sample)
                    if len(new_sample['bboxes']) != 0:
                        sample = new_sample
                        break
            image = sample['image']
            target['boxes'] = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1, 0)
            # torch.tensor(sample['bboxes'])
            target['labels'] = torch.stack(sample['labels'], 0)

        return image, target, image_id

    def __len__(self) -> int:
        return self.image_ids.shape[0]

class WheatTestDataset(Dataset):

    def __init__(self, DIR_INPUT, transforms=None):
        super().__init__()

        dataframe = pd.read_csv(f'{DIR_INPUT}/sample_submission.csv')

        self.image_ids = dataframe['image_id'].unique()
        self.df = dataframe
        self.image_dir = f'{DIR_INPUT}/test'
        self.transforms = transforms

    def __getitem__(self, index: int):

        image_id = self.image_ids[index]
        records = self.df[self.df['image_id'] == image_id]

        image = cv2.imread(f'{self.image_dir}/{image_id}.jpg', cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0

        if self.transforms:
            sample = {
                'image': image,
            }
            sample = self.transforms(**sample)
            image = sample['image']

        return image, image_id

    def __len__(self) -> int:
        return self.image_ids.shape[0]