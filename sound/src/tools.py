import os
import re


def read_java_file_without_comments(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        file_content = file.read()
        file_content = re.sub(r'//.*?\n|/\*.*?\*/', '', file_content, flags=re.DOTALL)
        return file_content


def get_java_files(folder_path, java_name):
    java_files = [f for f in os.listdir(folder_path) if f == java_name]
    return [os.path.join(folder_path, file) for file in java_files]
