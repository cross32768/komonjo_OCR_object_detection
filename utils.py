from collections import Counter
from copy import deepcopy

import numpy as np
import pandas as pd
from PIL import Image

import config


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
        image_path = image_dir + image_name + '.jpg'
        width, height = Image.open(image_path).size[:2]

        annotation_data_for_an_image = annotation_data[annotation_data.Image == image_name]
        extracted_annotation_np = annotation_data_for_an_image[['Unicode', 'X', 'Y', 'Width', 'Height']].values

        annotation_dict = {}
        annotation_dict['image_path'] = image_path
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
    preprocessed_annotations = []
    for dataset_index in dataset_index_list:
        dataset_dir = config.DATASET_DIR_LIST[dataset_index]
        path_to_annotation_csv = dataset_dir + dataset_dir.split('/')[-2] + '_coordinate.csv'
        original_image_dir = dataset_dir + 'images/'
        preprocessed_annotations += preprocess_annotation(path_to_annotation_csv,
                                                          original_image_dir)
    utf16_to_index, index_to_utf16 = \
        make_maps_between_index_and_frequent_characters_utf16(preprocessed_annotations,
                                                              config.N_KINDS_OF_CHARACTERS)
    selected_annotation = select_annotation_and_convert_ut16_to_index(preprocessed_annotations,
                                                                      utf16_to_index)
    return selected_annotation, index_to_utf16


def compute_IOU(coordinates1, coordinates2):
    x_min1, y_min1, x_max1, y_max1 = coordinates1
    x_min2, y_min2, x_max2, y_max2 = coordinates2
    intersect_w = np.maximum(np.minimum(x_max1, x_max2) - np.maximum(x_min1, x_min2), 0)
    intersect_h = np.maximum(np.minimum(y_max1, y_max2) - np.maximum(y_min1, y_min2), 0)
    intersection = intersect_w * intersect_h

    union = (x_max1-x_min1)*(y_max1-y_min1) + (x_max2-x_min2)*(y_max2-y_min2) - intersection
    epsilon = 1e-6

    return intersection / (union + epsilon)


def NMS(bboxes, border=0.3):
    bbox_indexes = np.argsort(bboxes[0])[::-1]
    NMS_result_indexes = list()

    while len(bbox_indexes) != 0:
        NMS_result_indexes.append(bbox_indexes[0])
        IOUs = compute_IOU(bboxes[:, bbox_indexes[0]][1:],
                           bboxes[:, bbox_indexes][1:])
        bbox_indexes = bbox_indexes[IOUs < border]
    return NMS_result_indexes
