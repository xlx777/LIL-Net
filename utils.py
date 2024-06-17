import os


def get_files(path):
    file_info = os.walk(path)
    file_list = []
    for r, d, f in file_info:
        file_list += f
    return file_list


def get_dirs(path):
    file_info = os.walk(path)
    dirs = []
    for d, r, f in file_info:
        dirs.append(d)
    return dirs[1:]


def get_dirs_train(path):
    file_info = os.listdir(path)
    dirs = []
    for f in file_info:
        dirs.append(os.path.join(path, f))
    return dirs


