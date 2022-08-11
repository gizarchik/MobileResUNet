from PIL import Image
from torch.utils.data import Dataset
import scipy.io

import torch
from torchvision import transforms

import os

from utils.one_hot_encoder import one_hot_encode


class COCOStuff10kDataset(Dataset):
    """
    COCOStuff10k features dataset.
    """

    def __init__(self, dataset_path: str, input_size: tuple = (160, 240),
                 split: str = 'train') -> None:
        """
        Args:
            data_path: путь до изображений.
            mask_path: путь до масок изображений.
        """

        self.data_path = dataset_path + 'images/'
        self.mask_path = dataset_path + 'annotations/'
        self.split_info_path = dataset_path + 'imageLists/'
        self.label_values = torch.arange(183)

        # Файлы с изображениями и масками
        if split == 'train':
            filenames = open(self.split_info_path + "train.txt", "r").readlines()
        elif split == 'test':
            filenames = open(self.split_info_path + "test.txt", "r").readlines()
        else:
            raise ValueError("argument split must be 'test' or 'train'")

        self.files = list(map(lambda x: x[:-1] + '.jpg', filenames))
        self.mask_files = list(map(lambda x: x[:-1] + '.mat', filenames))

        # Кэш файлы с изображениями и масками
        self.cache_image = [0] * len(self.files)
        self.cache_mask = [0] * len(self.mask_files)
        self.is_cached = [False] * len(self.files)

        assert len(self.files) == len(self.mask_files)

        # Преобразование для изображений
        self.transform = transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
        ])
        # Преобразование для масок
        self.transform_mask = transforms.Compose([
            transforms.ToPILImage(),
            self.transform,
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if not self.is_cached[idx]:
            # Загружаем изображение и маску
            file_name = os.path.join(self.data_path, self.files[idx])
            mask_name = os.path.join(self.mask_path, self.mask_files[idx])

            # Пробразуем изображение и маску
            input = self.transform(Image.open(file_name).convert('RGB'))
            target = (self.transform_mask(scipy.io.loadmat(mask_name)['S']).squeeze() * 255).int()
            one_hot_target = one_hot_encode(target, self.label_values).permute((2, 0, 1))

            # Кэшируем
            self.cache_image[idx] = input
            self.cache_mask[idx] = target
            self.is_cached[idx] = True
        else:
            input = self.cache_image[idx]
            target = self.cache_mask[idx]
            one_hot_target = one_hot_encode(target, self.label_values).permute((2, 0, 1))

        return input, one_hot_target
