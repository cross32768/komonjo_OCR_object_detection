import argparse
import os

from PIL import Image
from tqdm import tqdm

from utils import normalize_dir_name
import config


def resize_and_save(image_dir, resized_image_size):
    output_dir = image_dir[:-1] + '_resized_' + str(resized_image_size) + '/'

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    if len(os.listdir(output_dir)) != 0:
        print('The output directory is not empty.')
        return 0

    image_name_list = os.listdir(image_dir)
    for image_name in tqdm(image_name_list):
        image = Image.open(image_dir + image_name)
        image = image.resize((resized_image_size, resized_image_size),
                             Image.LANCZOS)
        image.save(output_dir + image_name)
    print(str(resized_image_size) + ' is completed.')
    return 1


def main():

    parser = argparse.ArgumentParser(description='program to resize images')
    parser.add_argument('image_dir',
                        help='path to directory include original images')
    args = parser.parse_args()

    image_dir = normalize_dir_name(args.image_dir)
    for resized_image_size in config.RESIZE_IMAGE_SIZE_CANDIDATES:
        resize_and_save(image_dir, resized_image_size)


if __name__ == '__main__':
    main()
