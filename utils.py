import pandas as pd


def preprocess_annotation(path_to_annotation_csv):
    annotation_data = pd.read_csv(path_to_annotation_csv)
    image_name_list = list(set(annotation_data.Image))
    processed_annotation_data = [None] * len(image_name_list)
    
    for idx, image_name in enumerate(image_name_list):
        annotation_data_for_an_image = annotation_data[annotation_data.Image == image_name]
        extracted_annotation_np = annotation_data_for_an_image[['Unicode', 'X', 'Y', 'Width', 'Height']].values
        
        annotation_dict = {}
        annotation_dict['image_name'] = image_name
        annotation_dict['annotation_data'] = extracted_annotation_np
        processed_annotation_data[idx] = annotation_dict
    
    return processed_annotation_data