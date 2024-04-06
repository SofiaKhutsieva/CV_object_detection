import os
import subprocess
import argparse

import ruamel.yaml as yaml


# Loading config
parser = argparse.ArgumentParser(
    description='Управляет процессами, необходимыми для обучения YOLOv5'
)
parser.add_argument('--config_file_path', type=str, required=True, help='Путь к файлу конфигурации')
args = parser.parse_args()

config_file_path = os.path.abspath(args.config_file_path)

with open(config_file_path) as stream:
    try:
        config = yaml.safe_load(stream)

        if 'preprocessing' not in config:
            print('Preprocessing not configured')
            exit(1)

        preprocessing_config = config['preprocessing']
        modelling_config = config['modelling']
        common_config = config['common']
    except yaml.YAMLError as exc:
        exit(1)

# Preprocessing
preprocessing_script_path = os.path.join('preprocessing', 'preprocessing.py')
args = f'--config_file_path="{config_file_path}"'

command = f'python3 {preprocessing_script_path} {args}'

cp = subprocess.run(command, shell=True)
status_code = cp.returncode

if status_code != 0:
    print('Error while running preprocessing.py')
    exit(1)

# Modeling
modeling_script_path = os.path.join(common_config['model_repo'], 'train.py')
args = f'--img {modelling_config["image_size"]} ' \
       f'--batch {modelling_config["batch_size"]} ' \
       f'--epochs {modelling_config["epochs_number"]} ' \
       f'--data {preprocessing_config["target_dataset_descriptor_file_path"]} ' \
       f'--hyp {modelling_config["hyperparameters_file"]} ' \
       f'--weights {modelling_config["weights"]} ' \
       f'--project {modelling_config["project_name"]} ' \
       f'--name {modelling_config["name"]}'

command = f'python3 {modeling_script_path} {args}'

cp = subprocess.run(command, shell=True)
status_code = cp.returncode

if status_code != 0:
    print('Error while running preprocessing.py')
    exit(1)