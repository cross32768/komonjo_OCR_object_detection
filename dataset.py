import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset

import config
from utils import normalize_dir_name


class OCRDataset(Dataset):
    def __init__(self, image_dir, annotation_list, transform=None):
        self.image_dir = normalize_dir_name(image_dir)
        self.annotation_list = annotation_list
        self.transform = transform

        self.image_size = 320

    def __len__(self):
        return len(self.annotation_list)

    def __getitem__(self, idx):
        annotation = self.annotation_list[idx]
        image_name = annotation['image_name']
        original_image_size = annotation['image_size']
        annotation_data = annotation['annotation_data']

        image = Image.open(self.image_dir + image_name)
        if self.transform is not None:
            image = self.transform(image)

        n_label_channel = 5 * config.N_KINDS_OF_CHARACTERS
        assert self.image_size % config.FE_STRIDE == 0, \
            "self.image_size must be multiple of feature extractor's stride."
        n_grid = int(self.image_size / config.FE_STRIDE)
        label = torch.zeros(n_label_channel, n_grid, n_grid)

        char_index = annotation_data[:, 0]
        min_x = annotation_data[:, 1]
        min_y = annotation_data[:, 2]
        width = annotation_data[:, 3]
        height = annotation_data[:, 4]

        center_x = min_x + 0.5*width
        center_y = min_y + 0.5*height

        shrink_rate = [self.image_size / original_image_size['Width'],
                       self.image_size / original_image_size['Height']]
        center_x = center_x * shrink_rate[0]
        center_y = center_y * shrink_rate[1]
        width = width * shrink_rate[0]
        height = height * shrink_rate[1]

        center_x_normalized = center_x / config.FE_STRIDE
        center_y_normalized = center_y / config.FE_STRIDE
        width_normalized = width / self.image_size
        height_normalized = height / self.image_size

        center_x_grid = center_x_normalized.astype(np.int16)
        center_y_grid = center_y_normalized.astype(np.int16)
        center_x_offset = center_x_normalized - center_x_grid
        center_y_offset = center_y_normalized - center_y_grid

        center_x_offset = torch.from_numpy(center_x_offset).float()
        center_y_offset = torch.from_numpy(center_y_offset).float()
        width_normalized = torch.from_numpy(width_normalized).float()
        height_normalized = torch.from_numpy(height_normalized).float()

        label[char_index+0, center_y_grid, center_x_grid] = 1.0
        label[char_index+1, center_y_grid, center_x_grid] = center_x_offset
        label[char_index+2, center_y_grid, center_x_grid] = center_y_offset
        label[char_index+3, center_y_grid, center_x_grid] = width_normalized
        label[char_index+4, center_y_grid, center_x_grid] = height_normalized

        return image, label
