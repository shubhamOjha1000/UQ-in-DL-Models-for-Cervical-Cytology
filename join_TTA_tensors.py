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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_idx', default=0, type=int, help='starting index')
    parser.add_argument('--end_idx', default=2000, type=int, help='starting index')
    parser.add_argument('--TTA_path', default='/path', type=str, help='path for TTA director')
    args = parser.parse_args()
    joined_path = os.path.join(args.TTA_path, 'joined_tensor')
    l = os.path.join(args.TTA_path, '1')
    for path in l[args.start_idx : args.end_idx]:
        print(path)
        List = []
        for idx in range(1, 51): 
            temp_path = os.path.join(args.TTA_path, str(idx), path)
            List.append(torch.load(temp_path))
        concatenated_tensor = torch.stack(List)
        torch.save(concatenated_tensor, os.path.join(joined_path, str(path)))


if __name__ == '__main__':
    main()