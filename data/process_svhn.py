import scipy.io
import numpy as np
from PIL import Image
import os

import argparse

from tqdm import tqdm

# Add arg parser so that we can specify the .mat, the output dir
parser = argparse.ArgumentParser(description='Convert SVHN .mat file to images')

parser.add_argument('--mat-file', type=str, required=True,
                    help='Path to the SVHN .mat file (e.g., train_32x32.mat)')
parser.add_argument('--output-dir', type=str, required=True,
                    help='Output directory to save the images')


def get_dir_file_names(number_path):
    if os.path.exists(number_path):
        existing_files = os.listdir(number_path)
        existing_numbers = [int(f.split('.')[0]) for f in existing_files if f.endswith('.png')]
        
        if len(existing_numbers) > 0:
            return max(existing_numbers) + 1
        else:
            return 1
    else:
        return 1

def create_dir_and_get_start_dict(path):
    start_dict = {}
    
    if os.path.exists(path):
        for i in range(10):
            start_dict[i] = get_dir_file_names(os.path.join(path, str(i)))
    else:
        os.makedirs(path)
        for i in range(10):
            start_dict[i] = 1
            os.makedirs(os.path.join(path, str(i)))

    return start_dict


if __name__ == "__main__":
    parsed_args = parser.parse_args()
    mat_file_path = parsed_args.mat_file
    output_dir = parsed_args.output_dir

    start_dict = create_dir_and_get_start_dict(output_dir)

    # Load the .mat file
    data = scipy.io.loadmat(mat_file_path)

    # Extract the images (the exact key name might vary)
    # For the training set, it's typically 'X' for images
    images = data['X']  # This should be a 4D array (32, 32, 3, N)
    labels = data['y']  # 1D array (N,)

    # # Iterate over each image in the dataset
    num_images = images.shape[3]
    for i in tqdm(range(num_images), desc=f"Casting {mat_file_path} to images"):
        # Get the image data (the array is in shape (32, 32, 3) per image)
        img = images[:, :, :, i]
        label = labels[i].item()

        if label == 10:
            label = 0

        # Convert the image array to a PIL image (Pillow)
        img_pil = Image.fromarray(np.transpose(img, (1, 0, 2)))  # Transpose to correct format
        
        # Save the image as a PNG file (you can also use other formats like JPEG if preferred)
        start_dict[label] += 1

        # Set image in b&w
        img_pil = img_pil.convert('L')
        # Set size to 28x28
        img_pil = img_pil.resize((28, 28))

        img_pil.save(os.path.join(output_dir, str(label), f'{start_dict[label]}.png'))
