import pandas as pd
from PIL import Image


def normalize_dir_name(dir_name):
    if dir_name[-1] == '/':
        return dir_name
    else:
        return dir_name + '/'


def preprocess_annotation(path_to_annotation_csv, original_image_dir):
    annotation_data = pd.read_csv(path_to_annotation_csv)
    image_name_list = list(set(annotation_data.Image))
    processed_annotation_data = [None] * len(image_name_list)
    image_dir = normalize_dir_name(original_image_dir)

    for idx, image_name in enumerate(image_name_list):
        image_file_name = image_name + '.jpg'
        width, height = Image.open(image_dir + image_file_name).size[:2]

        annotation_data_for_an_image = annotation_data[annotation_data.Image == image_name]
        extracted_annotation_np = annotation_data_for_an_image[['Unicode', 'X', 'Y', 'Width', 'Height']].values

        annotation_dict = {}
        annotation_dict['image_name'] = image_file_name
        annotation_dict['image_size'] = {'Width': width, 'Height': height}
        annotation_dict['annotation_data'] = extracted_annotation_np
        processed_annotation_data[idx] = annotation_dict

    return processed_annotation_data
