import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset

import config


class OCRDataset(Dataset):
    def __init__(self, annotation_list, transform=None):
        self.annotation_list = annotation_list
        self.transform = transform

        self.image_size = 0

    def __len__(self):
        return len(self.annotation_list)

    def __getitem__(self, idx):
        annotation = self.annotation_list[idx]
        original_image_path = annotation['image_path']
        original_image_size = annotation['image_size']
        annotation_data = annotation['annotation_data']

        splited_original_image_path = original_image_path.split('/')
        image_path = ('/'.join(splited_original_image_path[:-1])
                      + '_resized_'
                      + str(self.image_size)
                      + '/'
                      + splited_original_image_path[-1])
        image = Image.open(image_path)
        if self.transform is not None:
            image = self.transform(image)

        shrink_rate = [self.image_size / original_image_size['Width'],
                       self.image_size / original_image_size['Height']]
        label = self.anno2label(annotation_data, shrink_rate)

        return image, label

    def anno2label(self, annotation_data, shrink_rate):
        n_label_channel = 5 * config.N_KINDS_OF_CHARACTERS
        assert self.image_size % config.FE_STRIDE == 0, \
            "self.image_size must be multiple of feature extractor's stride."
        n_grid = int(self.image_size / config.FE_STRIDE)
        label = torch.zeros(n_label_channel, n_grid, n_grid)
        if len(annotation_data) == 0:
            return label

        char_index = annotation_data[:, 0]
        min_x = annotation_data[:, 1]
        min_y = annotation_data[:, 2]
        width = annotation_data[:, 3]
        height = annotation_data[:, 4]

        center_x = min_x + 0.5*width
        center_y = min_y + 0.5*height

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

        label[5*char_index + 0, center_y_grid, center_x_grid] = 1.0
        label[5*char_index + 1, center_y_grid, center_x_grid] = center_x_offset
        label[5*char_index + 2, center_y_grid, center_x_grid] = center_y_offset
        label[5*char_index + 3, center_y_grid, center_x_grid] = width_normalized
        label[5*char_index + 4, center_y_grid, center_x_grid] = height_normalized

        return label

    def update_image_size(self, candidate_list):
        new_image_size = np.random.choice(candidate_list)
        self.image_size = new_image_size

    def label2bboxes(self, label, confidence_border=0.5):
        bboxes = [None] * config.N_KINDS_OF_CHARACTERS
        for char_index in range(config.N_KINDS_OF_CHARACTERS):
            label_per_class = label[5*char_index:5*char_index + 5]
            n_grid = label_per_class.size(1)
            corresponding_image_size = n_grid * config.FE_STRIDE

            valid_index = label_per_class[0] > confidence_border
            confidence = label_per_class[0][valid_index]

            grid_index_for_x = torch.arange(n_grid).expand(n_grid, n_grid).float()
            grid_index_for_y = grid_index_for_x.transpose(0, 1)
            label_center_x_normalized = label_per_class[1] + grid_index_for_x
            label_center_y_normalized = label_per_class[2] + grid_index_for_y
            center_x_normalized = label_center_x_normalized[valid_index]
            center_y_normalized = label_center_y_normalized[valid_index]

            width_normalized = label_per_class[3][valid_index]
            height_normalized = label_per_class[4][valid_index]

            center_x = center_x_normalized * config.FE_STRIDE
            center_y = center_y_normalized * config.FE_STRIDE
            width = width_normalized * corresponding_image_size
            height = height_normalized * corresponding_image_size

            bbox = np.array([confidence.numpy(),
                             center_x.numpy(), center_y.numpy(),
                             width.numpy(), height.numpy()])
            bboxes[char_index] = bbox

        return bboxes


class OCRDatasetWithAnchor(Dataset):
    def __init__(self, annotation_list, transform=None):
        self.annotation_list = annotation_list
        self.transform = transform

        self.image_size = 0
        self.anchor_boxes = self.calculate_anchor_boxes()

    def __len__(self):
        return len(self.annotation_list)

    def __getitem__(self, idx):
        annotation = self.annotation_list[idx]
        original_image_path = annotation['image_path']
        original_image_size = annotation['image_size']
        annotation_data = annotation['annotation_data']

        splited_original_image_path = original_image_path.split('/')
        image_path = ('/'.join(splited_original_image_path[:-1])
                      + '_resized_'
                      + str(self.image_size)
                      + '/'
                      + splited_original_image_path[-1])
        image = Image.open(image_path)
        if self.transform is not None:
            image = self.transform(image)

        label = self.anno2label(annotation_data, original_image_size)

        return image, label

    def calculate_anchor_boxes(self):
        anchor_boxes = [None] * config.N_KINDS_OF_CHARACTERS
        for i in range(config.N_KINDS_OF_CHARACTERS):
            concated_annotation_data_for_index = np.array([]).reshape(0, 2)
            for annotation in self.annotation_list:
                annotation_data = annotation['annotation_data']
                original_image_size = annotation['image_size']
                if len(annotation_data) != 0:
                    annotation_data_for_index = annotation_data[annotation_data[:, 0] == i]
                    anchor_info = np.zeros(2, len(annotation_data_for_index))
                    anchor_info[0] = annotation_data_for_index[0] / original_image_size['Width']
                    anchor_info[1] = annotation_data_for_index[1] / original_image_size['Height']
                    concated_annotation_data_for_index = np.concatenate([concated_annotation_data_for_index, 
                                                                         anchor_info])
            widths = concated_annotation_data_for_index[0]
            heights = concated_annotation_data_for_index[1]
            width_mean = widths.mean()
            height_mean = heights.mean()
            anchor_boxes[i] = [width_mean, height_mean]
        return anchor_boxes

    def anno2label(self, annotation_data, shrink_ * shrink_rate[0]
        height = height * shrink_rate[1]rate):
        n_label_channel = 5 * config.N_KINDS_OF_C * shrink_rate[0]
        height = height * shrink_rate[1]HARACTERS
        assert self.image_size % config.FE_STRIDE * shrink_rate[0]
        height = height * shrink_rate[1] == 0, \
            "self.image_size must be multiple of  * shrink_rate[0]
        height = height * shrink_rate[1]feature extractor's stride."
        n_grid = int(self.image_size / config.FE_STRIDE)
        label = torch.zeros(n_label_channel, n_grid, n_grid)
        if len(annotation_data) == 0:
            return label

        char_index = annotation_data[:, 0]
        min_x = annotation_data[:, 1]
        min_y = annotation_data[:, 2]
        width = annotation_data[:, 3]
        height = annotation_data[:, 4]

        center_x = min_x + 0.5*width
        center_y = min_y + 0.5*height

        center_x = center_x * shrink_rate[0]
        center_y = center_y * shrink_rate[1]
        width = width
        height = height
        anchor_width = self.anchor_boxes[char_index][:, 0]
        anchor_height = self.anchor_boxes[char_index][:, 1]

        center_x_normalized = center_x / config.FE_STRIDE
        center_y_normalized = center_y / config.FE_STRIDE

        center_x_grid = center_x_normalized.astype(np.int16)
        center_y_grid = center_y_normalized.astype(np.int16)
        center_x_offset = center_x_normalized - center_x_grid
        center_y_offset = center_y_normalized - center_y_grid

        width_change_from_anchor = np.log(width / anchor_width)
        height_change_from_anchor = np.log(height / anchor_height)

        center_x_offset = torch.from_numpy(center_x_offset).float()
        center_y_offset = torch.from_numpy(center_y_offset).float()
        width_change_from_anchor = torch.from_numpy(width_change_from_anchor).float()
        height_change_from_anchor = torch.from_numpy(height_change_from_anchor).float()

        label[5*char_index + 0, center_y_grid, center_x_grid] = 1.0
        label[5*char_index + 1, center_y_grid, center_x_grid] = center_x_offset
        label[5*char_index + 2, center_y_grid, center_x_grid] = center_y_offset
        label[5*char_index + 3, center_y_grid, center_x_grid] = width_change_from_anchor
        label[5*char_index + 4, center_y_grid, center_x_grid] = height_change_from_anchor

        return label

    def update_image_size(self, candidate_list):
        new_image_size = np.random.choice(candidate_list)
        self.image_size = new_image_size

    def label2bboxes(self, label, original_image_size, confidence_border=0.5):
        bboxes = [None] * config.N_KINDS_OF_CHARACTERS
        for char_index in range(config.N_KINDS_OF_CHARACTERS):
            label_per_class = label[5*char_index:5*char_index + 5]
            n_grid = label_per_class.size(1)
            corresponding_image_size = n_grid * config.FE_STRIDE

            valid_index = label_per_class[0] > confidence_border
            confidence = label_per_class[0][valid_index]

            grid_index_for_x = torch.arange(n_grid).expand(n_grid, n_grid).float()
            grid_index_for_y = grid_index_for_x.transpose(0, 1)
            label_center_x_normalized = label_per_class[1] + grid_index_for_x
            label_center_y_normalized = label_per_class[2] + grid_index_for_y
            center_x_normalized = label_center_x_normalized[valid_index]
            center_y_normalized = label_center_y_normalized[valid_index]

            width_change_from_anchor = label_per_class[3][valid_index]
            height_change_from_anchor = label_per_class[4][valid_index]

            anchor_width = self.anchor_boxes[char_index][:, 0]
            anchor_height = self.anchor_boxes[char_index][:, 1]

            width = np.exp(width_change_from_anchor) * anchor_width
            height = np.exp(height_change_from_anchor) * anchor_height

            center_x = center_x_normalized * config.FE_STRIDE
            center_y = center_y_normalized * config.FE_STRIDE
            width = width * corresponding_image_size / original_image_size['Width']
            height = height * corresponding_image_size / original_image_size['Height']

            bbox = np.array([confidence.numpy(),
                             center_x.numpy(), center_y.numpy(),
                             width.numpy(), height.numpy()])
            bboxes[char_index] = bbox

        return bboxes
