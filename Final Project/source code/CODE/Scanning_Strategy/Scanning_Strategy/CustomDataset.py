import os

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import cv2
import utils


class VCDataset(Dataset):
    def __init__(self, img_dir, crop_size=64):

        self.img_dir = img_dir
        self.img_names = []
        for root, _, files in os.walk(img_dir):
            for filename in files:
                file_path = os.path.join(root, filename)
                self.img_names.append(file_path)

        if crop_size: self.crop = transforms.RandomCrop((64, 64))
        else: self.crop = None

    def __getitem__(self, index):
        
        img_path = self.img_names[index]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)

        if self.crop:
            img = self.crop(img)

        return img

    def __len__(self):
        return len(self.img_names)
    

class VCDataset2(Dataset):
    def __init__(self, img_dir, crop_size=64):

        self.img_dir = img_dir
        self.img_names = []
        for root, _, files in os.walk(img_dir):
            for filename in files:
                file_path = os.path.join(root, filename)
                self.img_names.append(file_path)

        self.img_names = [img_name for img_name in self.img_names if not img_name.endswith('7.png')]

        if crop_size: self.crop = transforms.RandomCrop((64, 64))
        else: self.crop = None

    def __getitem__(self, index):
        
        img_path = self.img_names[index]
        frame_number = int(os.path.basename(img_path)[2])
        img1 = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(img_path.replace(f'{frame_number}.png',f'{frame_number+1}.png'), cv2.IMREAD_GRAYSCALE)

        img1 = torch.tensor(img1, dtype=torch.float32).unsqueeze(0)
        img2 = torch.tensor(img2, dtype=torch.float32).unsqueeze(0)

        img = img1 - img2

        if self.crop:
            img = self.crop(img)

        return img

    def __len__(self):
        return len(self.img_names)