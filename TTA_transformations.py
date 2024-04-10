from PIL import Image
import numpy as np
import pandas as pd 
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import v2
import torch
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import argparse



class CricDataset(Dataset):
    def __init__(self, img_dir, annotation_file, TTA_number, img_transform = None):
        self.img_dir = img_dir
        self.img_labels = pd.read_csv(annotation_file)
        self.img_transform = img_transform
        self.TTA_number = TTA_number
        # Define additional transformations
        self.transforms = transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),  # Apply horizontal flip with probability 0.5
    transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.2)),  # Apply scaling between 0.8 and 1.2
    transforms.ColorJitter(hue=0.1, saturation=0.1, brightness=0.2, contrast=0.5)  # Convert PIL image to PyTorch tensor
    ])

    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = Image.open(img_path)
        if self.img_transform is not None:
            image = self.transforms(image)
        image = np.array(image)
        image = np.transpose(image, (2, 0, 1))
        image = image / 255.0
        torch_tensor = torch.tensor(image)
        
        #temp_path = '/scratch/shubham.ojha/TTA/1'
        os.makedirs(os.path.join('/scratch/shubham.ojha/TTA', str(self.TTA_number)), exist_ok=True)
        temp_path = os.path.join('/scratch/shubham.ojha/TTA', str(self.TTA_number))

        torch.save(torch_tensor, os.path.join(temp_path, str(self.img_labels.iloc[idx, 0]))[:-3] + 'pt')

        
        label = self.img_labels.iloc[idx, 1]

        return image, label
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_idx', default=1, type=int, help='starting index')
    parser.add_argument('--end_idx', default=1, type=int, help='starting index')
    args = parser.parse_args()
    img_dir = '/scratch/shubham.ojha/100_nucleus'
    path = '/scratch/shubham.ojha/binary_labels.csv'

    for i in range(args.start_idx, args.end_idx):
        print(i)
        dataset = CricDataset(img_dir = img_dir, annotation_file = path, TTA_number =  i, img_transform = True)
        loader = DataLoader(dataset, batch_size = 256, shuffle = False, num_workers = 4)
        for data in loader:
            pass   
    

if __name__ == '__main__':
    main()
    

