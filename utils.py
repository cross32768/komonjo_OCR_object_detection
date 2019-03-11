from collections import Counter
from copy import deepcopy

import numpy as np
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


def count_characters(preprocessed_annotation):
    concated_char_kind_utf16 = np.array([])
    for anno in preprocessed_annotation:
        anno_data = anno['annotation_data']
        char_kind_utf16 = anno_data[:, 0]
        concated_char_kind_utf16 = np.concatenate([concated_char_kind_utf16, char_kind_utf16])

    counter_descending_order = Counter(concated_char_kind_utf16).most_common()
    return counter_descending_order


def make_maps_between_index_and_frequent_characters_utf16(preprocessed_annotation, n_kinds_of_characters):
    counter_descending_order = count_characters(preprocessed_annotation)
    utf16_to_index = {}
    index_to_utf16 = {}
    for index in range(n_kinds_of_characters):
        utf16_to_index[counter_descending_order[index][0]] = index
        index_to_utf16[index] = counter_descending_order[index][0]
    return utf16_to_index, index_to_utf16


def select_annotation_and_convert_ut16_to_index(preprocessed_annotation, utf16_to_index):
    selected_annotation = deepcopy(preprocessed_annotation)

    for i, anno in enumerate(preprocessed_annotation):
        selected_annotation_data = []
        annotation_data = anno['annotation_data']

        for char_anno in annotation_data:
            utf16 = char_anno[0]
            if utf16 in utf16_to_index.keys():
                updated_char_anno = [utf16_to_index[utf16], *char_anno[1:]]
                selected_annotation_data.append(updated_char_anno)

        selected_annotation_data = np.array(selected_annotation_data)
        selected_annotation[i]['annotation_data'] = selected_annotation_data
    return selected_annotation


def prepare_selected_annotation_from_dataset_indexes(dataset_index_list):
    1+1