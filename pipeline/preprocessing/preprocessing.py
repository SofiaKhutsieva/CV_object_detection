import os
import argparse
import shutil
import json
import ast
import random

import ruamel.yaml as yaml
import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from tqdm.auto import tqdm
import torch
import torchvision.ops.boxes as bops

import preprocessing_utils

# Loading config
parser = argparse.ArgumentParser(
    description='Переносит данные из исходного датасета в указанный в формате, пригодном для YOLOv5'
)
parser.add_argument('--config_file_path', type=str, help='Путь к файлу конфигурации')
args = parser.parse_args()

with open(args.config_file_path) as stream:
    try:
        config = yaml.safe_load(stream)

        if 'preprocessing' not in config:
            print('Preprocessing not configured')
            exit(1)

        preprocessing_config = config['preprocessing']
        common_config = config['common']
    except yaml.YAMLError as exc:
        exit(1)

# Setting random state
random.seed(config['random_state'])
np.random.seed(config['random_state'])
torch.manual_seed(config['random_state'])
torch.backends.cudnn.deterministic = True

# Loading dataset descriptor
dataset_descriptor = pd.read_csv(preprocessing_config['source_dataset_descriptor_file_path'])
print(f'All images number: {len(dataset_descriptor)}')

img_name_2_img_shape = dict()

# Applying semi-supervised labeling
if preprocessing_config.get('semi_supervised', dict()).get('use_additional_labeling'):
    added_bboxes_counter = 0

    semi_supervised_config = preprocessing_config['semi_supervised']

    processed_sample = dataset_descriptor.copy()

    model = preprocessing_utils.load_model(
        common_config['model_repo'], semi_supervised_config['model_weights_path'],
        semi_supervised_config['confidence_threshold'], semi_supervised_config['nms_iou_threshold'],
    )

    for row_index, row_data in tqdm(processed_sample.iterrows()):

        video_id = row_data['video_id']
        video_frame = row_data['video_frame']
        annotations = eval(row_data['annotations'])

        if semi_supervised_config['processed_sample'] == 'images_without_bboxes' and annotations:
            continue

        image_name = f'{video_id}_{video_frame}.jpg'
        image_path = os.path.join(
            preprocessing_config['source_dataset_path'], 'train_images', f'video_{video_id}', f'{video_frame}.jpg'
        )

        img = cv2.imread(image_path)[..., ::-1]
        img_name_2_img_shape[image_name] = img.shape[:2]

        # predicting bboxes
        bboxes, confs = preprocessing_utils.predict(
            model, img, size=semi_supervised_config['image_size'], augment=semi_supervised_config['augment']
        )

        if len(bboxes):
            # formatting bboxes
            bboxes = [
                {'x': int(bbox[0]), 'y': int(bbox[1]), 'width': int(bbox[2] - bbox[0]),
                 'height': int(bbox[3] - bbox[1])}
                for bbox in bboxes
            ]

            # filtering bboxes
            for annotation in annotations:
                for bbox in list(bboxes):
                    compared_annotation = torch.tensor(
                        [[
                            annotation['x'], annotation['y'],
                            annotation['x'] + annotation['width'],
                            annotation['x'] + annotation['height']
                        ]], dtype=torch.float
                    )
                    compared_bbox = torch.tensor(
                        [[
                            bbox['x'], bbox['y'],
                            bbox['x'] + bbox['width'],
                            bbox['x'] + bbox['height']
                        ]], dtype=torch.float
                    )
                    iou = bops.box_iou(compared_annotation, compared_bbox)

                    if iou > semi_supervised_config['predictions_labels_iou_threshold']:
                        bboxes.remove(bbox)

            added_bboxes_counter += len(bboxes)

            # dumping bboxes
            annotations.extend(bboxes)
            annotations = json.dumps(annotations).replace('"', '\'')

            annotations_column_index = dataset_descriptor.columns.tolist().index('annotations')
            dataset_descriptor.iloc[
                (dataset_descriptor['video_id'] == video_id) & (dataset_descriptor['video_frame'] == video_frame),
                annotations_column_index
            ] = annotations

    print(f'Added bboxes: {added_bboxes_counter}')

# Adding additional fields

# Bboxes number
dataset_descriptor['num_bboxes'] = \
    dataset_descriptor['annotations'].apply(ast.literal_eval).apply(lambda item: len(item))

# Subsequences
subsequences = []

previous_bboxes_number = None

for _, row_data in dataset_descriptor.iterrows():
    current_bboxes_number = row_data['num_bboxes']

    if previous_bboxes_number is None:
        subsequences.append(0)
    elif (previous_bboxes_number == 0 and current_bboxes_number != 0) or \
            (previous_bboxes_number != 0 and current_bboxes_number == 0):
        new_bboxes_index = subsequences[-1] + 1
        subsequences.append(new_bboxes_index)
    else:
        current_subsequence_index = subsequences[-1]
        subsequences.append(current_subsequence_index)

    previous_bboxes_number = current_bboxes_number

dataset_descriptor['subsequence_index'] = subsequences

# dropping images without bboxes
df_without_bboxes = dataset_descriptor[dataset_descriptor.annotations == '[]']
print(f'Images without bboxes: {len(df_without_bboxes)}')

images_with_bboxes_number = len(dataset_descriptor[dataset_descriptor.annotations != '[]'])
enabled_images_without_bboxes_number = int(
    preprocessing_config['images_without_bboxes_part'] * images_with_bboxes_number
)
enabled_images_without_bboxes_number = min(enabled_images_without_bboxes_number, len(df_without_bboxes))

print(
    f'Saving {preprocessing_config["images_without_bboxes_part"] * 100}% images '
    f'without bboxes ({enabled_images_without_bboxes_number} items)'
)
drop_indices = np.random.choice(
    df_without_bboxes.index, len(df_without_bboxes) - enabled_images_without_bboxes_number, replace=False
)
df_without_bboxes = df_without_bboxes.drop(drop_indices)

dataset_descriptor = dataset_descriptor[dataset_descriptor.annotations != '[]']
dataset_descriptor = dataset_descriptor.append(df_without_bboxes, ignore_index=True)

print(f'Len of result dataset: {len(dataset_descriptor)}')

# Images and labels copying

# Copying images
print('Copying images')
source_images_folder_path = os.path.join(preprocessing_config['source_dataset_path'], 'train_images')
target_images_path = os.path.join(preprocessing_config['target_dataset_path'], 'images')
os.makedirs(target_images_path, exist_ok=True)

for _, row_data in tqdm(dataset_descriptor.iterrows(), total=len(dataset_descriptor)):
    video_id = row_data['video_id']
    video_frame = row_data['video_frame']

    image_name = f'{video_id}_{video_frame}.jpg'

    shutil.copyfile(
        os.path.join(source_images_folder_path, f'video_{video_id}', f'{video_frame}.jpg'),
        os.path.join(target_images_path, image_name)
    )

target_labels_path = os.path.join(preprocessing_config['target_dataset_path'], 'labels')
os.makedirs(target_labels_path, exist_ok=True)
label = 0

# Making labels
print('Making labels')

for _, row_data in tqdm(dataset_descriptor.iterrows(), total=len(dataset_descriptor)):
    video_id = row_data['video_id']
    video_frame = row_data['video_frame']

    image_name = f'{video_id}_{video_frame}.jpg'

    coco_bboxes = dataset_descriptor[
        (dataset_descriptor['video_id'] == video_id) & (dataset_descriptor['video_frame'] == video_frame)
        ]['annotations'].values[0]
    coco_bboxes = eval(coco_bboxes)
    coco_bboxes = np.array([
        [bbox['x'], bbox['y'], bbox['width'], bbox['height']] for bbox in coco_bboxes
    ])

    yolo_bboxes = list()

    for coco_bbox in coco_bboxes:
        height, width = img_name_2_img_shape[image_name]
        yolo_bbox = preprocessing_utils.convert_coco_labels_2_yolo_labels(coco_bbox, label, height, width)
        yolo_bbox = ' '.join(np.array(yolo_bbox).astype(str)).replace('0.0 ', '0 ')
        yolo_bboxes.append(yolo_bbox)

    yolo_bboxes = '\n'.join(yolo_bboxes)

    with open(os.path.join(target_labels_path, f'{video_id}_{video_frame}.txt'), 'w') as labels_file:
        labels_file.write(yolo_bboxes)

# Making train-test split
print('Making train-test split')

train_subset, validation_subset = None, None

if preprocessing_config['train_test_splitting_method'] == 'pseudo_stratified':
    train_subset, validation_subset = train_test_split(
        dataset_descriptor,
        test_size=preprocessing_config['test_size'],
        stratify=dataset_descriptor['num_bboxes'],
        random_state=config['random_state'],
    )

elif preprocessing_config['train_test_splitting_method'] == 'subsequence':
    gss = GroupShuffleSplit(
        n_splits=1,
        test_size=preprocessing_config['test_size'],
        random_state=config['random_state']
    )
    train_indexes, test_indexes = next(gss.split(dataset_descriptor, groups=dataset_descriptor['subsequence_index']))
    train_subset, validation_subset = dataset_descriptor.iloc[train_indexes], dataset_descriptor.iloc[test_indexes]

else:
    print('Train-test splitting method not defined')
    exit(1)

# saving train.txt
train_set_descriptor_path = os.path.join(preprocessing_config['target_dataset_path'], 'train.txt')
with open(train_set_descriptor_path, 'w') as train_images_file:
    for _, row_data in train_subset.iterrows():
        video_id = row_data['video_id']
        video_frame = row_data['video_frame']

        image_name = f'{video_id}_{video_frame}.jpg'

        train_images_file.write(
            f'{os.path.join(os.getcwd(), target_images_path, image_name)}\n'
        )

# saving val.txt
val_set_descriptor_path = os.path.join(preprocessing_config['target_dataset_path'], 'val.txt')
with open(val_set_descriptor_path, 'w') as val_images_file:
    for _, row_data in validation_subset.iterrows():
        video_id = row_data['video_id']
        video_frame = row_data['video_frame']

        image_name = f'{video_id}_{video_frame}.jpg'

        val_images_file.write(
            f'{os.path.join(os.getcwd(), target_images_path, image_name)}\n'
        )

dataset_config = {
    'path': os.path.join(os.getcwd(), preprocessing_config['target_dataset_path']),
    'train': os.path.basename(train_set_descriptor_path),
    'val': os.path.basename(val_set_descriptor_path),
    'test': None,
    'nc': 1,
    'names': ['COT'],
}

target_dataset_descriptor_file_path = preprocessing_config['target_dataset_descriptor_file_path']
os.makedirs(os.path.dirname(target_dataset_descriptor_file_path), exist_ok=True)

with open(target_dataset_descriptor_file_path, 'w') as dataset_config_file:
    yaml.dump(dataset_config, dataset_config_file)
