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
        1+1