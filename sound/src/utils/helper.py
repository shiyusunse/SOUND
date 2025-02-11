import shutil
from warnings import simplefilter

import os
import re
import numpy as np
import pickle

simplefilter(action='ignore', category=FutureWarning)

root_path = r'D:/line-level-defect-prediction-master/'
dataset_string = 'Dataset'
result_string = 'Result'

dataset_path = f'{root_path}/{dataset_string}/Bug-Info/'
file_level_path = f'{root_path}{dataset_string}/File-level/'
line_level_path = f'{root_path}{dataset_string}/Line-level/'
result_path = f'{root_path}{result_string}'
file_level_path_suffix = '_ground-truth-files_dataset.csv'
line_level_path_suffix = '_defective_lines_dataset.csv'


def read_file_level_dataset(release='', file_path=file_level_path):
    if release == '':
        return [], [], [], [], []
    path = f'{file_path}{release}{file_level_path_suffix}'
    with open(path, 'r', encoding='utf-8', errors='ignore') as file:
        lines = file.readlines()
        src_file_indices = [lines.index(line) for line in lines if r'.java,true,"' in line or r'.java,false,"' in line]
        src_files = [lines[index].split(',')[0] for index in src_file_indices]
        string_labels = [lines[index].split(',')[1] for index in src_file_indices]
        numeric_labels = [1 if label == 'true' else 0 for label in string_labels]

        texts_lines = []
        texts_lines_without_comments = []
        for i in range(len(src_file_indices)):
            s_index = src_file_indices[i]
            e_index = src_file_indices[i + 1] if i + 1 < len(src_file_indices) else len(lines)

            code_lines = [line.strip() for line in lines[s_index:e_index]]
            code_lines[0] = code_lines[0].split(',')[-1][1:]
            code_lines = code_lines[:-1]
            texts_lines.append(code_lines)

            enumerated_lines = list(enumerate(code_lines))
            new_lines = []
            for index, line in enumerated_lines:
                if not (line.startswith('/') or line.startswith('*')):
                    new_lines.append(line)

            code_lines_without_comments = [line.strip() for line in new_lines]
            texts_lines_without_comments.append(code_lines_without_comments)

        texts = [' '.join(line) for line in texts_lines]

        return texts, texts_lines, numeric_labels, src_files, texts_lines_without_comments


def read_line_level_dataset(release=''):
    if release == '':
        return dict()
    path = f'{line_level_path}{release}{line_level_path_suffix}'
    with open(path, 'r', encoding='utf-8', errors='ignore') as file:
        lines = file.readlines()
        file_buggy_lines_dict = {}
        for line in lines[1:]:
            temp = line.split(',', 2)
            file_name, buggy_line_number = temp[0], int(temp[1])
            if file_name not in file_buggy_lines_dict.keys():
                file_buggy_lines_dict[file_name] = [buggy_line_number]
            else:
                file_buggy_lines_dict[file_name].append(buggy_line_number)

    return file_buggy_lines_dict


def make_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


def save_csv_result(file_path, file_name, data):
    make_path(file_path)
    with open(f'{file_path}{file_name}', 'w', encoding='utf-8') as file:
        file.write(data)
    print(f'Result has been saved to {file_path}{file_name} successfully!')