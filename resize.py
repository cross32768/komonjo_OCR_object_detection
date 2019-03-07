import argparse
import os

from PIL import Image
from tqdm import tqdm

from utils import normalize_dir_name


def main():

    parser = argparse.ArgumentParser(description='program to resize images')
    parser.add_argument('image_dir',
                        help='path to directory include original images')
    parser.add_argument('output_dir',
                        help='path to directory save resized images')
    parser.add_argument('resized_image_size',
                        help='resized image size (only square)',
                        type=int)
    args = parser.parse_args()

    image_dir = normalize_dir_name(args.image_dir)
    output_dir = normalize_dir_name(args.output_dir)
    resized_image_size = args.resized_image_size

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    assert len(os.listdir(output_dir)) == 0, 'Output directory is not empty.'

    image_name_list = os.listdir(image_dir)
    for image_name in tqdm(image_name_list):
        image = Image.open(image_dir + image_name)
        image = image.resize((resized_image_size, resized_image_size),
                             Image.LANCZOS)
        image.save(output_dir + image_name)

if __name__ == '__main__':
    main()
