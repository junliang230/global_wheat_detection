#ref:https://github.com/open-mmlab/mmdetection
from numpy import random
import torch
import cv2
import numpy as np
import math
import albumentations as A

class PhotoMetricDistortion(object):
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.

    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)
    8. randomly swap channels

    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(self,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def __call__(self, results):
        img = results['image']
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

        results['image'] = img
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += ('(brightness_delta={}, contrast_range={}, '
                     'saturation_range={}, hue_delta={})').format(
                         self.brightness_delta, self.contrast_range,
                         self.saturation_range, self.hue_delta)
        return repr_str

class Mosaic(object):
    def __init__(self, p):
        self.p = p

    def load_image(self, index, image_ids, df):
        image_id = image_ids[index]
        DIR_INPUT = '/data1/jliang_data/dataset/wheat'
        imgpath = f'{DIR_INPUT}/train'
        img = cv2.imread(f'{imgpath}/{image_id}.jpg', cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        records = df[df['image_id'] == image_id]
        boxes = records[['x', 'y', 'w', 'h']].values
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
        boxes[:, 2] = np.where(boxes[:, 2] > 1024, 1024, boxes[:, 2]) #TODO: change 1024 to image width
        boxes[:, 3] = np.where(boxes[:, 3] > 1024, 1024, boxes[:, 3])
        im_w = 1024
        im_h = 1024
        boxesyolo = []
        for box in boxes:
            x1, y1, x2, y2 = box
            xc, yc, w, h = 0.5 * x1 / im_w + 0.5 * x2 / im_w, 0.5 * y1 / im_h + 0.5 * y2 / im_h, abs(
                x2 / im_w - x1 / im_w), abs(y2 / im_h - y1 / im_h)
            boxesyolo.append([1, xc, yc, w, h]) #TODO: label:0?
        bbox = np.array(boxesyolo)
        return img, bbox, img.shape[:2]

    def augment_hsv(self, results, hgain=0.5, sgain=0.5, vgain=0.5):
        img = results['image']
        r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
        hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
        dtype = img.dtype  # uint8

        x = np.arange(0, 256, dtype=np.int16)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
        cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed
        results['image'] = img

        return results

    def __call__(self, results):
        # loads images in a mosaic

        if np.random.rand() > self.p:
            return results

        index = results['idx']
        image_ids = results['image_ids']
        df = results['df']
        labels4 = []
        s = 1024
        xc, yc = [int(random.uniform(s * 0.5, s * 1.5)) for _ in range(2)]  # mosaic center x, y
        indices = [index] + [random.randint(0, len(image_ids) - 1) for _ in range(3)]  # 3 additional image indices
        for i, index in enumerate(indices):
            # Load image
            img, bbox, (h, w) = self.load_image(index, image_ids, df)

            # place img in img4
            if i == 0:  # top left
                img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            # print("{},{},{},{}".format(x1a,x2a,y1a,y2a))

            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            padw = x1a - x1b
            padh = y1a - y1b

            # Labels
            labels = bbox.copy()
            if bbox.size > 0:  # Normalized xywh to pixel xyxy format
                labels[:, 1] = w * (bbox[:, 1] - bbox[:, 3] / 2) + padw
                labels[:, 2] = h * (bbox[:, 2] - bbox[:, 4] / 2) + padh
                labels[:, 3] = w * (bbox[:, 1] + bbox[:, 3] / 2) + padw
                labels[:, 4] = h * (bbox[:, 2] + bbox[:, 4] / 2) + padh
            labels4.append(labels)

        # Concat/clip labels
        if len(labels4):
            labels4 = np.concatenate(labels4, 0)
            # np.clip(labels4[:, 1:] - s / 2, 0, s, out=labels4[:, 1:])  # use with center crop
            np.clip(labels4[:, 1:], 0, 2 * s, out=labels4[:, 1:])  # use with random_affine

        # Augment
        # img4 = img4[s // 2: int(s * 1.5), s // 2:int(s * 1.5)]  # center crop (WARNING, requires box pruning)
        # img4, labels4 = random_affine(img4, labels4,
        #                               degrees=1.98 * 2,
        #                               translate=0.05 * 2,
        #                               scale=0.05 * 2,
        #                               shear=0.641 * 2,
        #                               border=-s // 2)  # border to remove
        random_affine = RandomAffine(degrees=1.98 * 2,
                                      translate=0.05 * 2,
                                      scale=0.05 * 2,
                                      shear=0.641 * 2,
                                      border=-s // 2,
                                      p = 1)  # border to remove
        results['image'] = img4
        results['bboxes'] = labels4[:, 1:]
        results['labels'] = labels4[:, 0]
        results = random_affine(results)
        # results = self.augment_hsv(results, hgain=0.0138, sgain=0.678, vgain=0.36) #TODO: The validity has not been verified

        new_boxes = []
        new_labels = []
        for box in results['bboxes']:
            x1, y1, x2, y2 = box
            if x1 >= x2 or y1 >= y2:
                continue
            new_boxes.append(box)
            new_labels.append(1)
        results['image'] = results['image'].astype(np.float32)
        results['bboxes'] = np.array(new_boxes)
        results['labels'] = torch.from_numpy(np.array(new_labels))

        return results

class RandomAffine(object):
    def __init__(self,
                 degrees=1.98 * 2,
                 translate=0.05 * 2,
                 scale=0.05 * 2,
                 shear=0.641 * 2,
                 border=0,
                 p=0.5):
        self.degrees = float(degrees)
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.border = border
        self.p = p

    def __call__(self, results):
        # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
        # https://medium.com/uruvideo/dataset-augmentation-with-random-homographies-a8f4b44830d4

        if np.random.rand() > self.p:
            return results
        if type(results['labels']) is not np.ndarray:
            results['labels'] = results['labels'].numpy()
        targets = np.concatenate((np.expand_dims(results['labels'], 1), results['bboxes']), 1)
        img = results['image']

        if targets is None:  # targets = [cls, xyxy]
            targets = []
        height = img.shape[0] + self.border * 2
        width = img.shape[1] + self.border * 2

        # Rotation and Scale
        R = np.eye(3)
        a = random.uniform(-self.degrees, self.degrees)
        # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
        s = random.uniform(1 - self.scale, 1 + self.scale)
        R[:2] = cv2.getRotationMatrix2D(angle=a, center=(img.shape[1] / 2, img.shape[0] / 2), scale=s)

        # Translation
        T = np.eye(3)
        T[0, 2] = random.uniform(-self.translate, self.translate) * img.shape[0] + self.border  # x translation (pixels)
        T[1, 2] = random.uniform(-self.translate, self.translate) * img.shape[1] + self.border  # y translation (pixels)

        # Shear
        S = np.eye(3)
        S[0, 1] = math.tan(random.uniform(-self.shear, self.shear) * math.pi / 180)  # x shear (deg)
        S[1, 0] = math.tan(random.uniform(-self.shear, self.shear) * math.pi / 180)  # y shear (deg)

        # Combined rotation matrix
        M = S @ T @ R  # ORDER IS IMPORTANT HERE!!
        if (self.border != 0) or (M != np.eye(3)).any():  # image changed
            img = cv2.warpAffine(img, M[:2], dsize=(width, height), flags=cv2.INTER_LINEAR, borderValue=(114, 114, 114))

        # Transform label coordinates
        n = len(targets)
        if n:
            # warp points
            xy = np.ones((n * 4, 3))
            xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
            xy = (xy @ M.T)[:, :2].reshape(n, 8)

            # create new boxes
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

            # # apply angle-based reduction of bounding boxes
            # radians = a * math.pi / 180
            # reduction = max(abs(math.sin(radians)), abs(math.cos(radians))) ** 0.5
            # x = (xy[:, 2] + xy[:, 0]) / 2
            # y = (xy[:, 3] + xy[:, 1]) / 2
            # w = (xy[:, 2] - xy[:, 0]) * reduction
            # h = (xy[:, 3] - xy[:, 1]) * reduction
            # xy = np.concatenate((x - w / 2, y - h / 2, x + w / 2, y + h / 2)).reshape(4, n).T

            # reject warped points outside of image
            xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
            xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)
            w = xy[:, 2] - xy[:, 0]
            h = xy[:, 3] - xy[:, 1]
            area = w * h
            area0 = (targets[:, 3] - targets[:, 1]) * (targets[:, 4] - targets[:, 2])
            ar = np.maximum(w / (h + 1e-16), h / (w + 1e-16))  # aspect ratio
            i = (w > 4) & (h > 4) & (area / (area0 * s + 1e-16) > 0.2) & (ar < 10)

            targets = targets[i]
            targets[:, 1:5] = xy[i]

        results['image'] = img
        results['bboxes'] = targets[:, 1:]
        results['labels'] = targets[:, 0]

        return results

class MixUp(object):
    #ref: https://www.kaggle.com/shonenkov/oof-evaluation-mixup-efficientdet/
    def __init__(self, p=0.5, mode=1):
        self.p = p
        self.mode = mode

    def load_image(self, index, image_ids, df):
        image_id = image_ids[index]
        DIR_INPUT = '/data1/jliang_data/dataset/wheat'
        imgpath = f'{DIR_INPUT}/train'
        img = cv2.imread(f'{imgpath}/{image_id}.jpg', cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        records = df[df['image_id'] == image_id]
        boxes = records[['x', 'y', 'w', 'h']].values
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
        boxes[:, 2] = np.where(boxes[:, 2] > 1024, 1024, boxes[:, 2]) #TODO: change 1024 to image width
        boxes[:, 3] = np.where(boxes[:, 3] > 1024, 1024, boxes[:, 3])

        return img, boxes

    def __call__(self, results):
        if np.random.rand() > self.p:
            return results


        img = results['image']
        boxes = results['bboxes']
        image_ids = results['image_ids']
        df = results['df']
        indices = random.randint(0, len(image_ids) - 1)
        new_img, new_bbox = self.load_image(indices, image_ids, df)

        if self.mode == 1:
            mixup_image = (img + new_img) / 2
            mixup_boxes = np.concatenate((boxes, new_bbox), 0)
            mixup_label = torch.ones((results['labels'].shape[0]+new_bbox.shape[0],), dtype=torch.int64)
        elif self.mode == 2:
            r_image, r_boxes = new_img, new_bbox
            mixup_image = img.copy()

            imsize = img.shape[0]
            x1, y1 = [int(random.uniform(imsize * 0.0, imsize * 0.45)) for _ in range(2)]
            x2, y2 = [int(random.uniform(imsize * 0.55, imsize * 1.0)) for _ in range(2)]

            mixup_boxes = r_boxes.copy()
            mixup_boxes[:, [0, 2]] = mixup_boxes[:, [0, 2]].clip(min=x1, max=x2)
            mixup_boxes[:, [1, 3]] = mixup_boxes[:, [1, 3]].clip(min=y1, max=y2)

            mixup_boxes = mixup_boxes.astype(np.int32)
            mixup_boxes = mixup_boxes[
                np.where((mixup_boxes[:, 2] - mixup_boxes[:, 0]) * (mixup_boxes[:, 3] - mixup_boxes[:, 1]) > 0)]

            mixup_image[y1:y2, x1:x2] = (mixup_image[y1:y2, x1:x2] + r_image[y1:y2, x1:x2]) / 2
            mixup_boxes = np.concatenate((boxes, mixup_boxes), 0)
            mixup_label = torch.ones((mixup_boxes.shape[0],), dtype=torch.int64)

        results['image'] = mixup_image
        results['bboxes'] = mixup_boxes
        results['labels'] = mixup_label

        return results


class Resize(object):
    def __init__(self, img_scale, multiscale_mode='range'):
        self.img_scale = img_scale
        self.multiscale_mode = multiscale_mode

    def random_sample(self, img_scales):
        img_scale_long = [max(s) for s in img_scales]
        img_scale_short = [min(s) for s in img_scales]
        long_edge = np.random.randint(
            min(img_scale_long),
            max(img_scale_long) + 1)
        short_edge = np.random.randint(
            min(img_scale_short),
            max(img_scale_short) + 1)
        img_scale = (long_edge, short_edge)
        return img_scale, None

    def random_select(self, img_scales):
        scale_idx = np.random.randint(len(img_scales))
        img_scale = img_scales[scale_idx]
        return img_scale, scale_idx

    def __call__(self, results):
        sample = {
            'image': results['image'],
            'bboxes': results['bboxes'],
            'labels': results['labels'],
        }
        if self.multiscale_mode == 'range':
            scale, scale_idx = self.random_sample(self.img_scale)
        elif self.multiscale_mode == 'value':
            scale, scale_idx = self.random_select(self.img_scale)

        t = A.Compose([
            A.Resize(height=scale[0], width=scale[1], p=1)
        ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
        new_sample = t(**sample)

        results['image'] = new_sample['image']
        results['bboxes'] = np.stack(new_sample['bboxes'])
        results['labels'] = torch.stack(new_sample['labels'], 0)

        return results
