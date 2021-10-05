import json
import os
import shutil as shutil
from typing import Dict, List, Optional


# File I/O
# ----------------------------------------------------------------------------------------------------------------------

def dir_up(path, n):  # here 'path' is your path, 'n' is number of dirs up you want to go
    for _ in range(n):
        path = dir_up(path.rpartition("\\")[0], 0)
        # second argument equal '0' ensures that the function iterates proper number of times
    return path


def find_first_file_with_extension(dir_path: str, extension: str) -> str:
    file_name = None
    if extension[0] != '.':
        extension = '.' + extension
    for fn in os.listdir(dir_path):
        if fn.endswith(extension):
            file_name = fn
            break
    return file_name


def find_all_files_with_extension(dir_path: str, extension: str) -> List[str]:
    file_names = []
    if extension[0] != '.':
        extension = '.' + extension
    for fn in os.listdir(dir_path):
        if fn.endswith(extension):
            file_names.append(fn)
    return file_names


def read_file(path: str, file_name: str = None) -> str:
    if path is None:
        file_path = file_name
    elif file_name is None:
        file_path = path
    else:
        file_path = os.path.join(path, file_name)
    with open(file_path, 'r') as f:
        text = f.read()
    return text


def read_file_lines(path: str, file_name: str = None) -> List[str]:
    if path is None:
        file_path = file_name
    elif file_name is None:
        file_path = path
    else:
        file_path = os.path.join(path, file_name)
    with open(file_path, 'r') as f:
        lines = f.readlines()
    return lines


def read_json_file(path: str, file_name: str = None):
    if path is None:
        file_path = file_name
    elif file_name is None:
        file_path = path
    else:
        file_path = os.path.join(path, file_name)
    with open(file_path, 'r') as f:
        o = json.load(f)
    return o


def write_file(dir_path: Optional[str],
               file_name: str,
               text: str,
               can_retry: bool = True):

    if dir_path is None:
        dir_path = os.getcwd()

    file_path = os.path.join(dir_path, file_name)

    is_looping = True

    while is_looping:

        try:
            with open(file_path, 'w') as f:
                f.write(text)
            is_looping = False

        except IOError as e:
            print(e)
            if can_retry:
                is_looping = retry_write_interface(file_path)


def write_json_file(dir_path: str, file_name: str, obj):
    with open(os.path.join(dir_path, file_name), 'w') as f:
        json.dump(obj, f, indent=4, sort_keys=True)


def retry_write_interface(file_path: str):
    retry: bool = True
    while True:
        prompt = "Encountered an error while saving file '{0}'. \nTry again? (Y/N) \n".format(file_path)
        option = str(input(prompt))
        if len(option) > 0:
            if option[0].lower() == 'y':
                break
            elif option[0].lower() == 'n':
                retry = False
                break
    return retry


def copy_file(source_dir_path: str, source_file_name: str, dest_dir_path: str, dest_file_name: str = None):
    source_file_path = os.path.join(source_dir_path, source_file_name)
    if dest_file_name is None:
        dest_file_path = os.path.join(dest_dir_path, source_file_name)
    else:
        dest_file_path = os.path.join(dest_dir_path, dest_file_name)
    shutil.copyfile(source_file_path, dest_file_path)


def move_file(source_dir_path: str, source_file_name: str, dest_dir_path: str, dest_file_name: str = None):
    source_file_path = os.path.join(source_dir_path, source_file_name)
    if dest_file_name is None:
        dest_file_path = os.path.join(dest_dir_path, source_file_name)
    else:
        dest_file_path = os.path.join(dest_dir_path, dest_file_name)
    shutil.move(source_file_path, dest_file_path)


# Strings
# ----------------------------------------------------------------------------------------------------------------------

def replace(text: str, word_map: Dict[str, str]) -> str:
    for word, replacement in word_map.items():
        text = text.replace(word, replacement)
    return text


def remove_escape_characters(word: str) -> str:
    esc_filter = ''.join([chr(i) for i in range(1, 32)])
    return word.translate(str.maketrans('', '', esc_filter))


# Collections
# ----------------------------------------------------------------------------------------------------------------------

def splice_list(l: list, remove_indices: list):
    return [i for j, i in enumerate(l) if j not in remove_indices]
