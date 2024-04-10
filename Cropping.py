import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
from PIL import Image
import argparse


def crop_subpart(image_path, center_x, center_y, width, height, file_name, cell_img_dir):
    # Open the image using PIL
    img = Image.open(image_path)

    # Calculate the coordinates for cropping
    left = center_x - width // 2
    upper = center_y - height // 2
    right = center_x + width // 2
    lower = center_y + height // 2

    # Crop the image using the calculated coordinates
    cropped_img = img.crop((left, upper, right, lower))

    # Show the cropped image (optional)
    #cropped_img.save('/scratch/shubham.ojha/1.png')
    cropped_img.save(os.path.join(cell_img_dir, str(file_name)+'.png'))



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='/path', type=str, help='path to the cell centre coordinates file')
    parser.add_argument('--img_dir', default='/path', type=str, help='path to the cric img patches directory')
    parser.add_argument('--cell_img_dir', default='/path', type=str, help='path to cell img directory')
    args = parser.parse_args()

    # loading the annotation file :-
    df = pd.read_csv(args.dataset)

    for index, row in df.iterrows():
        image_filename = row['image_filename']
    
        image_path = os.path.join(args.img_dir, row['image_filename'])
        center_x = row['nucleus_x']
        center_y = row['nucleus_y']
        width = 100
        height = 100
        file_name = row['cell_id']
    
        # Assuming crop_subpart() is a function that processes the image
        crop_subpart(image_path, center_x, center_y, width, height, file_name, args.cell_img_dir)



if __name__ == '__main__':
    main()