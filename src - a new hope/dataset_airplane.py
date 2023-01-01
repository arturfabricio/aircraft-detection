from torch.utils.data import Dataset
from skimage import io
import numpy as np
import torch
import pandas as pd
import json
from pathlib import Path
from torchvision.transforms import Normalize, CenterCrop
from torch.nn.utils.rnn import pad_sequence


from data_aug import Resize, RandomHSV, RandomRotate, RandomScale, RandomTranslate, RandomShear, RandomHorizontalFlip


class RareplanesDataset(Dataset):
    def __init__(self, json_file: str, root_dir: str, transform=None):
        with open(json_file) as json_data:
            self.dataframe = pd.DataFrame(json.load(json_data)['categories'])
        self.dataframe = self.dataframe[['image_fname', 'bbox']]
        self.dataframe = self.dataframe.groupby(['image_fname']).agg(list).reset_index()
        self.dataframe.rename(columns={"image_fname": "image_filename", "bbox": "bboxs"}, inplace=True)

        # add a class to bboxes, data aug require a class
        for row in self.dataframe.itertuples():
            [bbox.append(0) for bbox in row.bboxs]
        # to numpy
        self.dataframe['bboxs'] = self.dataframe['bboxs'].apply(np.array)
        # convert bboxs from xywh to xyxy
        for row in self.dataframe.itertuples():
            row.bboxs[:, 2] = row.bboxs[:, 2] + row.bboxs[:, 0]
            row.bboxs[:, 3] = row.bboxs[:, 3] + row.bboxs[:, 1]

        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image_path = Path(self.root_dir, self.dataframe['image_filename'][idx])
        image = io.imread(image_path)

        sample = {'image': image, 'bboxs': self.dataframe['bboxs'][idx]}

        if self.transform:
            sample = self.transform(sample)

        return sample

    @staticmethod
    def collate_fn(batch):
        image = [sample['image'] for sample in batch]
        image = torch.stack(image)

        has_plane = [sample['has_plane'] for sample in batch]
        has_plane = torch.tensor(has_plane, dtype=torch.float32).unsqueeze(dim=1)

        # bboxs = [torch.tensor(sample['bboxs']) for sample in batch]
        # bboxs = pad_sequence(bboxs, batch_first=True)

        return {
            'image': image,
            # 'bboxs': bboxs,
            'has_plane': has_plane
        }


class ToClassificationTask(object):
    def __call__(self, sample):
        sample['has_plane'] = sample['bboxs'].size != 0
        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        sample['image'] = torch.tensor(sample['image'].transpose((2, 0, 1)) / 255, dtype=torch.float32)
        return sample


class Resize(Resize):
    def __init__(self, inp_dim):
        super(Resize, self).__init__(inp_dim)

    def __call__(self, sample):
        sample['image'], sample['bboxs'] = super(Resize, self).__call__(sample['image'], sample['bboxs'])
        return sample


class RandomTranslate(RandomTranslate):
    def __init__(self, translate=0.2, diff=False):
        super(RandomTranslate, self).__init__(translate, diff)

    def __call__(self, sample):
        sample['image'], sample['bboxs'] = super(RandomTranslate, self).__call__(sample['image'], sample['bboxs'])
        return sample


class Normalize(Normalize):
    def __init__(self, mean, std, inplace=False):
        super(Normalize, self).__init__(mean, std, inplace)

    def __call__(self, sample):
        sample['image'] = super(Normalize, self).__call__(sample['image'])
        return sample


class RandomHSV(RandomHSV):
    def __init__(self, hue=None, saturation=None, brightness=None):
        super(RandomHSV, self).__init__(hue, saturation, brightness)

    def __call__(self, sample):
        sample['image'], sample['bboxs'] = super(RandomHSV, self).__call__(sample['image'], sample['bboxs'])
        return sample


class RandomRotate(RandomRotate):
    def __init__(self, angle=10):
        super(RandomRotate, self).__init__(angle)

    def __call__(self, sample):
        sample['image'], sample['bboxs'] = super(RandomRotate, self).__call__(sample['image'], sample['bboxs'])
        return sample


class RandomScale(RandomScale):
    def __init__(self, scale=0.2, diff=False):
        super(RandomScale, self).__init__(scale, diff)

    def __call__(self, sample):
        sample['image'], sample['bboxs'] = super(RandomScale, self).__call__(sample['image'], sample['bboxs'])
        return sample


class RandomShear(RandomShear):
    def __init__(self, shear_factor=0.2):
        super(RandomShear, self).__init__(shear_factor)

    def __call__(self, sample):
        sample['image'], sample['bboxs'] = super(RandomShear, self).__call__(sample['image'], sample['bboxs'])
        return sample


class RandomHorizontalFlip(RandomHorizontalFlip):
    def __init__(self, p=0.5):
        super(RandomHorizontalFlip, self).__init__(p)

    def __call__(self, sample):
        sample['image'], sample['bboxs'] = super(RandomHorizontalFlip, self).__call__(sample['image'], sample['bboxs'])
        return sample


class CenterCrop(CenterCrop):
    def __init__(self, size):
        super(CenterCrop, self).__init__(size)

    def __call__(self, sample):
        size_before = sample['image'].shape
        sample['image'] = super(CenterCrop, self).__call__(sample['image'])
        size_after = sample['image'].shape
        x_diff, y_diff = (size_before[1] - size_after[1])//2, (size_before[2] - size_after[2])//2

        original_area = np.multiply(sample['bboxs'][:, 2] - sample['bboxs'][:, 0], sample['bboxs'][:, 3] - sample['bboxs'][:, 1])

        sample['bboxs'][:, [0, 2]] = np.clip(sample['bboxs'][:, [0, 2]], x_diff, size_before[1] - x_diff) - x_diff
        sample['bboxs'][:, [1, 3]] = np.clip(sample['bboxs'][:, [1, 3]], y_diff, size_before[2] - y_diff) - y_diff

        new_area = np.multiply(sample['bboxs'][:, 2] - sample['bboxs'][:, 0], sample['bboxs'][:, 3] - sample['bboxs'][:, 1])

        sample['bboxs'] = sample['bboxs'][(new_area / original_area > 0.15)]

        return sample
