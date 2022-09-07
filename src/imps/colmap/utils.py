import os


def is_known_params(data_dir):
    return os.path.exists(os.path.join(data_dir, 'sparse/0/cameras.txt'))

def does_exist(data_dir):
    return os.path.exists(os.path.join(data_dir, 'sparse/0/cameras.bin'))